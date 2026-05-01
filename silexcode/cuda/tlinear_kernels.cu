#include "tlinear.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

__constant__ int8_t TRIT_LUT[256][5];

static constexpr uint8_t POW3[5] = {1, 3, 9, 27, 81};
static constexpr int WARP_SIZE = 32;
static constexpr int TILE_T = 4;
static constexpr int TILE_O = 4;
static constexpr int WARPS_PER_CTA = TILE_T * TILE_O;
static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * WARP_SIZE;
static constexpr int FAST_TILE_T = 8;
static constexpr int FAST_TILE_O = 16;
static constexpr int FAST_WARPS_PER_CTA = FAST_TILE_O;
static constexpr int FAST_THREADS_PER_CTA = FAST_WARPS_PER_CTA * WARP_SIZE;
static constexpr int TILE_T_DX = 4;
static constexpr int TILE_I_DX = 4;
static constexpr int WARPS_PER_CTA_DX = TILE_T_DX * TILE_I_DX;
static constexpr int THREADS_PER_CTA_DX = WARPS_PER_CTA_DX * WARP_SIZE;
static constexpr int FAST_DX_TILE_T = 8;
static constexpr int FAST_DX_TILE_I = 16;
static constexpr int FAST_DX_WARPS_PER_CTA = FAST_DX_TILE_I;
static constexpr int FAST_DX_THREADS_PER_CTA = FAST_DX_WARPS_PER_CTA * WARP_SIZE;

int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

int align_up_int(int x, int a) {
    return ((x + a - 1) / a) * a;
}

int row_stride_bytes(int d_in) {
    return align_up_int(ceil_div_int(d_in, 5), 32);
}

static void init_trit_lut_host(int8_t lut[256][5]) {
    for (int b = 0; b < 256; ++b) {
        for (int r = 0; r < 5; ++r) {
            if (b <= 242) {
                int c = (b / POW3[r]) % 3;
                lut[b][r] = static_cast<int8_t>(c - 1);
            } else {
                lut[b][r] = 0;
            }
        }
    }
}

void init_trit_lut_cuda() {
    static bool initialized = false;
    if (initialized) {
        return;
    }

    int8_t lut_host[256][5];
    init_trit_lut_host(lut_host);
    cudaMemcpyToSymbol(TRIT_LUT, lut_host, 256 * 5 * sizeof(int8_t));
    initialized = true;
}

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ void ternary_accumulate(
    float& acc,
    int8_t trit,
    float x
) {
    if (trit > 0) {
        acc += x;
    }
    if (trit < 0) {
        acc -= x;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;

    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);

    return v;
}

__device__ __forceinline__ int deterministic_zero_residue_for_o(int o, int layer, int matrix_id) {
    int v = 9 * (-(13 * ((o + 1) & 15)) - (matrix_id & 15) - (3 * (layer & 15)));
    return (v - 1) & 15;
}

__device__ __forceinline__ float hadamard_low_sign(int low, int residue) {
    return (__popc(static_cast<unsigned>((low & residue) & 15)) & 1) ? -1.0f : 1.0f;
}

__device__ void fwht_shared(float* data, int n) {
    for (int step = 1; step < n; step <<= 1) {
        int pairs = n >> 1;
        for (int idx = threadIdx.x; idx < pairs; idx += blockDim.x) {
            int group = idx / step;
            int j = idx - group * step;
            int a_idx = group * (step << 1) + j;
            int b_idx = a_idx + step;
            float a = data[a_idx];
            float b = data[b_idx];
            data[a_idx] = a + b;
            data[b_idx] = a - b;
        }
        __syncthreads();
    }
}

__device__ void fwht_high_bits_shared(float* data, int n) {
    for (int step = 16; step < n; step <<= 1) {
        int pairs = n >> 1;
        for (int idx = threadIdx.x; idx < pairs; idx += blockDim.x) {
            int group = idx / step;
            int j = idx - group * step;
            int a_idx = group * (step << 1) + j;
            int b_idx = a_idx + step;
            float a = data[a_idx];
            float b = data[b_idx];
            data[a_idx] = a + b;
            data[b_idx] = a - b;
        }
        __syncthreads();
    }
}

__global__ void tlinear_forward_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t*       __restrict__ Wpack,
    const __nv_bfloat16* __restrict__ alpha,
    __nv_bfloat16*       __restrict__ Y,
    int T,
    int d_in,
    int d_out,
    int row_stride_bytes
) {
    int thread = threadIdx.x;

    int warp_id = thread >> 5;
    int lane    = thread & 31;

    int local_t = warp_id / 4;
    int local_o = warp_id - local_t * 4;

    int t = blockIdx.x * 4 + local_t;
    int o = blockIdx.y * 4 + local_o;

    float acc = 0.0f;

    if (t < T && o < d_out) {
        const uint8_t* wrow = Wpack + o * row_stride_bytes;
        const __nv_bfloat16* xrow = X + t * d_in;

        for (int kb = lane; kb < row_stride_bytes; kb += 32) {
            uint8_t packed = wrow[kb];
            int base_i = kb * 5;

            #pragma unroll
            for (int r = 0; r < 5; ++r) {
                int i = base_i + r;
                if (i < d_in) {
                    int8_t trit = TRIT_LUT[packed][r];
                    float x = bf16_to_f32(xrow[i]);
                    ternary_accumulate(acc, trit, x);
                }
            }
        }

        acc = warp_sum(acc);

        if (lane == 0) {
            float a = bf16_to_f32(alpha[o]);
            Y[t * d_out + o] = f32_to_bf16(acc * a);
        }
    }
}

__global__ void tlinear_forward_tiled_t_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t*       __restrict__ Wpack,
    const __nv_bfloat16* __restrict__ alpha,
    __nv_bfloat16*       __restrict__ Y,
    int T,
    int d_in,
    int d_out,
    int row_stride_bytes
) {
    int thread = threadIdx.x;
    int warp_id = thread >> 5;
    int lane = thread & 31;

    int t_base = blockIdx.x * FAST_TILE_T;
    int o = blockIdx.y * FAST_TILE_O + warp_id;

    float acc[FAST_TILE_T];
    #pragma unroll
    for (int tt = 0; tt < FAST_TILE_T; ++tt) {
        acc[tt] = 0.0f;
    }

    if (o < d_out) {
        const uint8_t* wrow = Wpack + o * row_stride_bytes;

        for (int kb = lane; kb < row_stride_bytes; kb += 32) {
            uint8_t packed = wrow[kb];
            int base_i = kb * 5;

            #pragma unroll
            for (int r = 0; r < 5; ++r) {
                int i = base_i + r;
                if (i < d_in) {
                    int8_t trit = TRIT_LUT[packed][r];

                    #pragma unroll
                    for (int tt = 0; tt < FAST_TILE_T; ++tt) {
                        int t = t_base + tt;
                        if (t < T) {
                            ternary_accumulate(acc[tt], trit, bf16_to_f32(X[t * d_in + i]));
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int tt = 0; tt < FAST_TILE_T; ++tt) {
            acc[tt] = warp_sum(acc[tt]);
        }

        if (lane == 0) {
            float a = bf16_to_f32(alpha[o]);
            #pragma unroll
            for (int tt = 0; tt < FAST_TILE_T; ++tt) {
                int t = t_base + tt;
                if (t < T) {
                    Y[t * d_out + o] = f32_to_bf16(acc[tt] * a);
                }
            }
        }
    }
}

__global__ void tlinear_backward_input_kernel(
    const __nv_bfloat16* __restrict__ dY,
    const uint8_t*       __restrict__ Wpack,
    const __nv_bfloat16* __restrict__ alpha,
    float*               __restrict__ dX,
    int T,
    int d_in,
    int d_out,
    int row_stride_bytes
) {
    int thread = threadIdx.x;

    int warp_id = thread >> 5;
    int lane    = thread & 31;

    int local_t = warp_id / 4;
    int local_i = warp_id - local_t * 4;

    int t = blockIdx.x * 4 + local_t;
    int i = blockIdx.y * 4 + local_i;

    float acc = 0.0f;

    if (t < T && i < d_in) {
        int kb = i / 5;
        int r  = i - kb * 5;

        for (int o = lane; o < d_out; o += 32) {
            uint8_t packed = Wpack[o * row_stride_bytes + kb];
            int8_t trit = TRIT_LUT[packed][r];

            float dy = bf16_to_f32(dY[t * d_out + o]);
            float a  = bf16_to_f32(alpha[o]);
            float v = dy * a;

            if (trit > 0) {
                acc += v;
            }

            if (trit < 0) {
                acc -= v;
            }
        }

        acc = warp_sum(acc);

        if (lane == 0) {
            dX[t * d_in + i] = acc;
        }
    }
}

__global__ void tlinear_backward_input_tiled_t_kernel(
    const __nv_bfloat16* __restrict__ dY,
    const uint8_t*       __restrict__ Wpack,
    const __nv_bfloat16* __restrict__ alpha,
    float*               __restrict__ dX,
    int T,
    int d_in,
    int d_out,
    int row_stride_bytes
) {
    int thread = threadIdx.x;
    int warp_id = thread >> 5;
    int lane = thread & 31;

    int t_base = blockIdx.x * FAST_DX_TILE_T;
    int i = blockIdx.y * FAST_DX_TILE_I + warp_id;

    float acc[FAST_DX_TILE_T];
    #pragma unroll
    for (int tt = 0; tt < FAST_DX_TILE_T; ++tt) {
        acc[tt] = 0.0f;
    }

    if (i < d_in) {
        int kb = i / 5;
        int r = i - kb * 5;

        for (int o = lane; o < d_out; o += 32) {
            uint8_t packed = Wpack[o * row_stride_bytes + kb];
            int8_t trit = TRIT_LUT[packed][r];
            float a = bf16_to_f32(alpha[o]);

            #pragma unroll
            for (int tt = 0; tt < FAST_DX_TILE_T; ++tt) {
                int t = t_base + tt;
                if (t < T) {
                    float v = bf16_to_f32(dY[t * d_out + o]) * a;
                    if (trit > 0) {
                        acc[tt] += v;
                    }
                    if (trit < 0) {
                        acc[tt] -= v;
                    }
                }
            }
        }

        #pragma unroll
        for (int tt = 0; tt < FAST_DX_TILE_T; ++tt) {
            acc[tt] = warp_sum(acc[tt]);
        }

        if (lane == 0) {
            #pragma unroll
            for (int tt = 0; tt < FAST_DX_TILE_T; ++tt) {
                int t = t_base + tt;
                if (t < T) {
                    dX[t * d_in + i] = acc[tt];
                }
            }
        }
    }
}

__global__ void deterministic_tlinear_forward_fwht_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ alpha,
    __nv_bfloat16*       __restrict__ Y,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
) {
    extern __shared__ float smem[];
    float* full = smem;
    int sub_n = d_in >> 4;
    int t = blockIdx.x;
    if (t >= T) {
        return;
    }

    for (int i = threadIdx.x; i < d_in; i += blockDim.x) {
        full[i] = bf16_to_f32(X[t * d_in + i]);
    }
    __syncthreads();
    fwht_high_bits_shared(full, d_in);

    for (int o = threadIdx.x; o < d_out; o += blockDim.x) {
        int low = o & 15;
        int high = (o >> 4) & (sub_n - 1);
        int zr = deterministic_zero_residue_for_o(o, layer, matrix_id);
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r != zr) {
                acc += hadamard_low_sign(low, r) * full[(high << 4) + r];
            }
        }
        float a = bf16_to_f32(alpha[o]);
        Y[t * d_out + o] = f32_to_bf16(acc * a);
    }
}

__global__ void deterministic_tlinear_forward_multi_fwht_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ alpha0,
    const __nv_bfloat16* __restrict__ alpha1,
    const __nv_bfloat16* __restrict__ alpha2,
    const __nv_bfloat16* __restrict__ alpha3,
    __nv_bfloat16*       __restrict__ Y0,
    __nv_bfloat16*       __restrict__ Y1,
    __nv_bfloat16*       __restrict__ Y2,
    __nv_bfloat16*       __restrict__ Y3,
    int count,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id0
) {
    extern __shared__ float full[];
    int sub_n = d_in >> 4;
    int t = blockIdx.x;
    if (t >= T) {
        return;
    }

    for (int i = threadIdx.x; i < d_in; i += blockDim.x) {
        full[i] = bf16_to_f32(X[t * d_in + i]);
    }
    __syncthreads();
    fwht_high_bits_shared(full, d_in);

    for (int o = threadIdx.x; o < d_out; o += blockDim.x) {
        int low = o & 15;
        int high = (o >> 4) & (sub_n - 1);
        float parts[16];
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            parts[r] = hadamard_low_sign(low, r) * full[(high << 4) + r];
        }
        float total = 0.0f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            total += parts[r];
        }

        int zr0 = deterministic_zero_residue_for_o(o, layer, matrix_id0 + 0);
        Y0[t * d_out + o] = f32_to_bf16((total - parts[zr0]) * bf16_to_f32(alpha0[o]));
        if (count > 1) {
            int zr1 = deterministic_zero_residue_for_o(o, layer, matrix_id0 + 1);
            Y1[t * d_out + o] = f32_to_bf16((total - parts[zr1]) * bf16_to_f32(alpha1[o]));
        }
        if (count > 2) {
            int zr2 = deterministic_zero_residue_for_o(o, layer, matrix_id0 + 2);
            Y2[t * d_out + o] = f32_to_bf16((total - parts[zr2]) * bf16_to_f32(alpha2[o]));
        }
        if (count > 3) {
            int zr3 = deterministic_zero_residue_for_o(o, layer, matrix_id0 + 3);
            Y3[t * d_out + o] = f32_to_bf16((total - parts[zr3]) * bf16_to_f32(alpha3[o]));
        }
    }
}

__global__ void deterministic_tlinear_backward_input_fwht_kernel(
    const __nv_bfloat16* __restrict__ dY,
    const __nv_bfloat16* __restrict__ alpha,
    float*               __restrict__ dX,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
) {
    extern __shared__ float smem[];
    float* full = smem;
    int sub_n = d_in >> 4;
    int t = blockIdx.x;
    if (t >= T) {
        return;
    }

    for (int i = threadIdx.x; i < d_in; i += blockDim.x) {
        int q = i >> 4;
        int r = i & 15;
        float acc = 0.0f;
        for (int rep = 0; rep * d_in + (q << 4) < d_out; ++rep) {
            int base_o = rep * d_in + (q << 4);
            #pragma unroll
            for (int low = 0; low < 16; ++low) {
                int o = base_o + low;
                if (o < d_out && deterministic_zero_residue_for_o(o, layer, matrix_id) != r) {
                    acc += hadamard_low_sign(low, r) * bf16_to_f32(dY[t * d_out + o]) * bf16_to_f32(alpha[o]);
                }
            }
        }
        full[i] = acc;
    }
    __syncthreads();
    fwht_high_bits_shared(full, d_in);

    for (int i = threadIdx.x; i < d_in; i += blockDim.x) {
        dX[t * d_in + i] = full[i];
    }
}

__global__ void embedding_forward_kernel(
    const uint16_t*      __restrict__ token_ids,
    const uint8_t*       __restrict__ Epack,
    const __nv_bfloat16* __restrict__ alpha,
    __nv_bfloat16*       X,
    int T,
    int d,
    int row_stride_bytes
) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * d;
    if (linear >= total) {
        return;
    }

    int t = linear / d;
    int j = linear - t * d;
    uint16_t token = token_ids[t];
    if (token > 257) {
        return;
    }

    int kb = j / 5;
    int r = j - kb * 5;
    uint8_t packed = Epack[token * row_stride_bytes + kb];
    int8_t trit = TRIT_LUT[packed][r];
    float a = bf16_to_f32(alpha[token]);
    float value = 0.0f;
    ternary_accumulate(value, trit, a);
    X[linear] = f32_to_bf16(value);
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ int8_t deterministic_trit_device(int o, int i, int layer, int matrix_id, int d_in) {
    long long z = 1103515245ll * static_cast<long long>(o + 1)
        + 12345ll * static_cast<long long>(i + 1)
        + 97ll * static_cast<long long>(matrix_id)
        + 131ll * static_cast<long long>(layer);
    if ((z & 15ll) == 0ll) {
        return 0;
    }
    unsigned long long x = static_cast<unsigned long long>((o % d_in) & i);
    return (__popcll(x) & 1) == 0 ? static_cast<int8_t>(1) : static_cast<int8_t>(-1);
}

__global__ void deterministic_pack_kernel(
    uint8_t* __restrict__ Wpack,
    int d_out,
    int d_in,
    int stride,
    int layer,
    int matrix_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d_out * stride;
    if (idx >= total) {
        return;
    }
    int o = idx / stride;
    int kb = idx - o * stride;
    uint32_t byte_value = 0;

    #pragma unroll
    for (int r = 0; r < 5; ++r) {
        int i = kb * 5 + r;
        int c = 1;
        if (i < d_in) {
            c = static_cast<int>(deterministic_trit_device(o, i, layer, matrix_id, d_in)) + 1;
        }
        int p = (r == 0) ? 1 : ((r == 1) ? 3 : ((r == 2) ? 9 : ((r == 3) ? 27 : 81)));
        byte_value += static_cast<uint32_t>(c) * static_cast<uint32_t>(p);
    }
    Wpack[idx] = static_cast<uint8_t>(byte_value);
}

__global__ void deterministic_alpha_kernel(
    __nv_bfloat16* __restrict__ alpha,
    int d_out,
    int d_in,
    int layer,
    int matrix_id
) {
    int o = blockIdx.x;
    if (o >= d_out) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int count = 0;

    for (int i = threadIdx.x; i < d_in; i += blockDim.x) {
        count += deterministic_trit_device(o, i, layer, matrix_id, d_in) != 0 ? 1 : 0;
    }

    float sum = static_cast<float>(count);
    sum = warp_sum(sum);
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            float nonzero = fmaxf(1.0f, total);
            alpha[o] = f32_to_bf16(rsqrtf(nonzero));
        }
    }
}

__global__ void rms_norm_forward_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16*       __restrict__ y,
    int T,
    int d,
    float eps
) {
    int t = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    float sum = 0.0f;

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = bf16_to_f32(x[t * d + j]);
        sum += v * v;
    }

    sum = warp_sum(sum);
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(total / static_cast<float>(d) + eps);
        }
    }
    __syncthreads();

    float inv_rms = warp_sums[0];
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = bf16_to_f32(x[t * d + j]);
        float g = bf16_to_f32(gamma[j]);
        y[t * d + j] = f32_to_bf16(v * inv_rms * g);
    }
}

__global__ void rms_norm_backward_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const float*         __restrict__ grad_y,
    float*               __restrict__ grad_x,
    int T,
    int d,
    float eps
) {
    int t = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    float sum_x2 = 0.0f;
    float sum_dot = 0.0f;

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float xv = bf16_to_f32(x[t * d + j]);
        float gv = bf16_to_f32(gamma[j]);
        float dy = grad_y[t * d + j];
        sum_x2 += xv * xv;
        sum_dot += dy * gv * xv;
    }

    sum_x2 = warp_sum(sum_x2);
    sum_dot = warp_sum(sum_dot);
    __shared__ float warp_x2[8];
    __shared__ float warp_dot[8];
    if (lane == 0) {
        warp_x2[warp_id] = sum_x2;
        warp_dot[warp_id] = sum_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float total_x2 = threadIdx.x < 8 ? warp_x2[lane] : 0.0f;
        float total_dot = threadIdx.x < 8 ? warp_dot[lane] : 0.0f;
        total_x2 = warp_sum(total_x2);
        total_dot = warp_sum(total_dot);
        if (lane == 0) {
            float inv = rsqrtf(total_x2 / static_cast<float>(d) + eps);
            warp_x2[0] = inv;
            warp_dot[0] = total_dot * inv * inv * inv / static_cast<float>(d);
        }
    }
    __syncthreads();

    float inv = warp_x2[0];
    float dot_scale = warp_dot[0];
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float xv = bf16_to_f32(x[t * d + j]);
        float gv = bf16_to_f32(gamma[j]);
        float dy = grad_y[t * d + j];
        grad_x[t * d + j] = dy * gv * inv - xv * dot_scale;
    }
}

__global__ void activation_forward_kernel(
    const __nv_bfloat16* __restrict__ x,
    float*               __restrict__ y,
    int total,
    int activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float v = bf16_to_f32(x[idx]);
    if (activation == 0) {
        y[idx] = sigmoid_f32(v);
    } else {
        y[idx] = v * sigmoid_f32(v);
    }
}

__global__ void activation_backward_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float*         __restrict__ grad_out,
    float*               __restrict__ grad_x,
    int total,
    int activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float xv = bf16_to_f32(x[idx]);
    float s = sigmoid_f32(xv);
    float local;
    if (activation == 0) {
        local = s * (1.0f - s);
    } else {
        local = s * (1.0f + xv * (1.0f - s));
    }
    grad_x[idx] = grad_out[idx] * local;
}

__global__ void activation_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float*         __restrict__ grad_out,
    __nv_bfloat16*       __restrict__ grad_x,
    int total,
    int activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float xv = bf16_to_f32(x[idx]);
    float s = sigmoid_f32(xv);
    float local;
    if (activation == 0) {
        local = s * (1.0f - s);
    } else {
        local = s * (1.0f + xv * (1.0f - s));
    }
    grad_x[idx] = f32_to_bf16(grad_out[idx] * local);
}

__global__ void gated_silu_product_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16*       __restrict__ h,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float av = bf16_to_f32(a[idx]);
    float bv = bf16_to_f32(b[idx]);
    h[idx] = f32_to_bf16((av * sigmoid_f32(av)) * bv);
}

__global__ void swiglu_backward_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float*         __restrict__ grad_h,
    float*               __restrict__ grad_a,
    float*               __restrict__ grad_b,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float av = bf16_to_f32(a[idx]);
    float bv = bf16_to_f32(b[idx]);
    float s = sigmoid_f32(av);
    float silu = av * s;
    float dsilu = s * (1.0f + av * (1.0f - s));
    float gh = grad_h[idx];
    grad_a[idx] = gh * bv * dsilu;
    grad_b[idx] = gh * silu;
}

__global__ void swiglu_backward_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float*         __restrict__ grad_h,
    __nv_bfloat16*       __restrict__ grad_a,
    __nv_bfloat16*       __restrict__ grad_b,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float av = bf16_to_f32(a[idx]);
    float bv = bf16_to_f32(b[idx]);
    float s = sigmoid_f32(av);
    float silu = av * s;
    float dsilu = s * (1.0f + av * (1.0f - s));
    float gh = grad_h[idx];
    grad_a[idx] = f32_to_bf16(gh * bv * dsilu);
    grad_b[idx] = f32_to_bf16(gh * silu);
}

__global__ void residual_add_forward_kernel(
    const __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ ternary,
    const __nv_bfloat16* __restrict__ adapter,
    __nv_bfloat16*       __restrict__ out,
    int total,
    float rho
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float b = bf16_to_f32(base[idx]);
    float t = bf16_to_f32(ternary[idx]);
    float p = bf16_to_f32(adapter[idx]);
    out[idx] = f32_to_bf16(b + rho * (t + p));
}

__global__ void adapter_a_forward_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float*         __restrict__ A,
    float*               __restrict__ hidden,
    int T,
    int d,
    int r
) {
    int t = blockIdx.x;
    int p = blockIdx.y;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    float acc = 0.0f;

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        acc += bf16_to_f32(x[t * d + j]) * A[p * d + j];
    }

    acc = warp_sum(acc);
    __shared__ float warp_sums[8];
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp_id == 0) {
        total = threadIdx.x < 8 ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            hidden[t * r + p] = total;
        }
    }
}

__global__ void adapter_b_forward_kernel(
    const float*   __restrict__ hidden,
    const float*   __restrict__ B,
    __nv_bfloat16* __restrict__ out,
    int T,
    int d,
    int r
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * d;
    if (idx >= total) {
        return;
    }
    int t = idx / d;
    int j = idx - t * d;
    float acc = 0.0f;
    for (int p = 0; p < 64; ++p) {
        if (p < r) {
            acc += hidden[t * r + p] * B[j * r + p];
        }
    }
    out[idx] = f32_to_bf16(acc);
}

__global__ void recurrent_mixer_forward_kernel(
    const float*        __restrict__ i_gate,
    const float*        __restrict__ f_gate,
    const float*        __restrict__ v_val,
    const float*        __restrict__ r_gate,
    const __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ lambda_raw,
    const __nv_bfloat16* __restrict__ beta_raw,
    __nv_bfloat16*       __restrict__ g,
    __nv_bfloat16*       __restrict__ new_state,
    int T,
    int R,
    int d
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) {
        return;
    }

    float m[8];
    float lambda[8];
    float beta_exp[8];
    float beta_sum = 0.0f;

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            m[s] = bf16_to_f32(state[s * d + j]);
            lambda[s] = sigmoid_f32(bf16_to_f32(lambda_raw[s * d + j]));
            beta_exp[s] = expf(bf16_to_f32(beta_raw[s * d + j]));
            beta_sum += beta_exp[s];
        }
    }

    for (int t = 0; t < T; ++t) {
        float i_t = i_gate[t * d + j];
        float f_t = f_gate[t * d + j];
        float v_t = v_val[t * d + j];
        float r_t = r_gate[t * d + j];
        float write = i_t * v_t;
        float c = 0.0f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < R) {
                float mix = lambda[s] * f_t;
                m[s] = mix * m[s] + (1.0f - mix) * write;
                c += (beta_exp[s] / beta_sum) * m[s];
            }
        }

        g[t * d + j] = f32_to_bf16(r_t * c);
    }

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            new_state[s * d + j] = f32_to_bf16(m[s]);
        }
    }
}

__global__ void recurrent_mixer_forward_trace_kernel(
    const float*        __restrict__ i_gate,
    const float*        __restrict__ f_gate,
    const float*        __restrict__ v_val,
    const float*        __restrict__ r_gate,
    const __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ lambda_raw,
    const __nv_bfloat16* __restrict__ beta_raw,
    __nv_bfloat16*       __restrict__ g,
    __nv_bfloat16*       __restrict__ new_state,
    __nv_bfloat16*       __restrict__ m_trace,
    int T,
    int R,
    int d
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) {
        return;
    }

    float m[8];
    float lambda[8];
    float beta_exp[8];
    float beta_sum = 0.0f;

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            m[s] = bf16_to_f32(state[s * d + j]);
            lambda[s] = sigmoid_f32(bf16_to_f32(lambda_raw[s * d + j]));
            beta_exp[s] = expf(bf16_to_f32(beta_raw[s * d + j]));
            beta_sum += beta_exp[s];
        }
    }

    for (int t = 0; t < T; ++t) {
        float i_t = i_gate[t * d + j];
        float f_t = f_gate[t * d + j];
        float v_t = v_val[t * d + j];
        float r_t = r_gate[t * d + j];
        float write = i_t * v_t;
        float c = 0.0f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < R) {
                float mix = lambda[s] * f_t;
                m[s] = mix * m[s] + (1.0f - mix) * write;
                m_trace[((t * R + s) * d) + j] = f32_to_bf16(m[s]);
                c += (beta_exp[s] / beta_sum) * m[s];
            }
        }

        g[t * d + j] = f32_to_bf16(r_t * c);
    }

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            new_state[s * d + j] = f32_to_bf16(m[s]);
        }
    }
}

__global__ void recurrent_mixer_backward_kernel(
    const float*         __restrict__ i_gate,
    const float*         __restrict__ f_gate,
    const float*         __restrict__ v_val,
    const float*         __restrict__ r_gate,
    const __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ lambda_raw,
    const __nv_bfloat16* __restrict__ beta_raw,
    const float*         __restrict__ grad_g,
    float*               __restrict__ d_i,
    float*               __restrict__ d_f,
    float*               __restrict__ d_v,
    float*               __restrict__ d_r,
    __nv_bfloat16*       __restrict__ d_state,
    int T,
    int R,
    int d
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) {
        return;
    }

    float lambda[8];
    float beta_exp[8];
    float beta[8];
    float beta_sum = 0.0f;
    float prev[8];
    float m_trace[512][8];
    float c_trace[512];

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            lambda[s] = sigmoid_f32(bf16_to_f32(lambda_raw[s * d + j]));
            beta_exp[s] = expf(bf16_to_f32(beta_raw[s * d + j]));
            beta_sum += beta_exp[s];
            prev[s] = bf16_to_f32(state[s * d + j]);
        }
    }
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        beta[s] = s < R ? beta_exp[s] / beta_sum : 0.0f;
    }

    for (int t = 0; t < T; ++t) {
        float it = i_gate[t * d + j];
        float ft = f_gate[t * d + j];
        float vt = v_val[t * d + j];
        float write = it * vt;
        float c = 0.0f;
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < R) {
                float mix = lambda[s] * ft;
                prev[s] = mix * prev[s] + (1.0f - mix) * write;
                m_trace[t][s] = prev[s];
                c += beta[s] * prev[s];
            }
        }
        c_trace[t] = c;
    }

    float d_m_next[8];
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        d_m_next[s] = 0.0f;
    }

    for (int t = T - 1; t >= 0; --t) {
        float dg = grad_g[t * d + j];
        float rt = r_gate[t * d + j];
        float it = i_gate[t * d + j];
        float ft = f_gate[t * d + j];
        float vt = v_val[t * d + j];
        float write = it * vt;
        float dc = dg * rt;
        float d_write = 0.0f;
        float dft = 0.0f;
        d_r[t * d + j] = dg * c_trace[t];

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < R) {
                float dm = d_m_next[s] + beta[s] * dc;
                float m_prev = (t == 0) ? bf16_to_f32(state[s * d + j]) : m_trace[t - 1][s];
                float mix = lambda[s] * ft;
                dft += dm * (m_prev - write) * lambda[s];
                d_write += dm * (1.0f - mix);
                d_m_next[s] = dm * mix;
            }
        }

        d_f[t * d + j] = dft;
        d_i[t * d + j] = d_write * vt;
        d_v[t * d + j] = d_write * it;
    }

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            d_state[s * d + j] = f32_to_bf16(d_m_next[s]);
        }
    }
}

__global__ void recurrent_mixer_backward_trace_kernel(
    const float*         __restrict__ i_gate,
    const float*         __restrict__ f_gate,
    const float*         __restrict__ v_val,
    const float*         __restrict__ r_gate,
    const __nv_bfloat16* __restrict__ state,
    const __nv_bfloat16* __restrict__ lambda_raw,
    const __nv_bfloat16* __restrict__ beta_raw,
    const __nv_bfloat16* __restrict__ m_trace,
    const float*         __restrict__ grad_g,
    float*               __restrict__ d_i,
    float*               __restrict__ d_f,
    float*               __restrict__ d_v,
    float*               __restrict__ d_r,
    __nv_bfloat16*       __restrict__ d_state,
    int T,
    int R,
    int d
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) {
        return;
    }

    float lambda[8];
    float beta_exp[8];
    float beta[8];
    float beta_sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            lambda[s] = sigmoid_f32(bf16_to_f32(lambda_raw[s * d + j]));
            beta_exp[s] = expf(bf16_to_f32(beta_raw[s * d + j]));
            beta_sum += beta_exp[s];
        }
    }
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        beta[s] = s < R ? beta_exp[s] / beta_sum : 0.0f;
    }

    float d_m_next[8];
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        d_m_next[s] = 0.0f;
    }

    for (int t = T - 1; t >= 0; --t) {
        float dg = grad_g[t * d + j];
        float rt = r_gate[t * d + j];
        float it = i_gate[t * d + j];
        float ft = f_gate[t * d + j];
        float vt = v_val[t * d + j];
        float write = it * vt;
        float c = 0.0f;
        float dc = dg * rt;
        float d_write = 0.0f;
        float dft = 0.0f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < R) {
                float mt = bf16_to_f32(m_trace[((t * R + s) * d) + j]);
                c += beta[s] * mt;
            }
        }
        d_r[t * d + j] = dg * c;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s < R) {
                float dm = d_m_next[s] + beta[s] * dc;
                float m_prev = (t == 0) ? bf16_to_f32(state[s * d + j]) : bf16_to_f32(m_trace[(((t - 1) * R + s) * d) + j]);
                float mix = lambda[s] * ft;
                dft += dm * (m_prev - write) * lambda[s];
                d_write += dm * (1.0f - mix);
                d_m_next[s] = dm * mix;
            }
        }

        d_f[t * d + j] = dft;
        d_i[t * d + j] = d_write * vt;
        d_v[t * d + j] = d_write * it;
    }

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        if (s < R) {
            d_state[s * d + j] = f32_to_bf16(d_m_next[s]);
        }
    }
}

void launch_tlinear_forward(
    const at::BFloat16* X,
    const uint8_t* Wpack,
    const at::BFloat16* alpha,
    at::BFloat16* Y,
    int T,
    int d_in,
    int d_out
) {
    init_trit_lut_cuda();
    int stride = row_stride_bytes(d_in);

    dim3 block(FAST_THREADS_PER_CTA, 1, 1);
    dim3 grid(ceil_div_int(T, FAST_TILE_T), ceil_div_int(d_out, FAST_TILE_O), 1);

    tlinear_forward_tiled_t_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X),
        Wpack,
        reinterpret_cast<const __nv_bfloat16*>(alpha),
        reinterpret_cast<__nv_bfloat16*>(Y),
        T,
        d_in,
        d_out,
        stride
    );
}

void launch_tlinear_backward_input(
    const at::BFloat16* dY,
    const uint8_t* Wpack,
    const at::BFloat16* alpha,
    float* dX,
    int T,
    int d_in,
    int d_out
) {
    init_trit_lut_cuda();
    int stride = row_stride_bytes(d_in);

    dim3 block(FAST_DX_THREADS_PER_CTA, 1, 1);
    dim3 grid(ceil_div_int(T, FAST_DX_TILE_T), ceil_div_int(d_in, FAST_DX_TILE_I), 1);

    tlinear_backward_input_tiled_t_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(dY),
        Wpack,
        reinterpret_cast<const __nv_bfloat16*>(alpha),
        dX,
        T,
        d_in,
        d_out,
        stride
    );
}

void launch_deterministic_tlinear_forward(
    const at::BFloat16* X,
    const at::BFloat16* alpha,
    at::BFloat16* Y,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
) {
    size_t shmem = static_cast<size_t>(d_in) * sizeof(float);
    cudaFuncSetAttribute(
        deterministic_tlinear_forward_fwht_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem)
    );
    dim3 block(512, 1, 1);
    dim3 grid(T, 1, 1);
    deterministic_tlinear_forward_fwht_kernel<<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X),
        reinterpret_cast<const __nv_bfloat16*>(alpha),
        reinterpret_cast<__nv_bfloat16*>(Y),
        T,
        d_in,
        d_out,
        layer,
        matrix_id
    );
}

void launch_deterministic_tlinear_forward_multi(
    const at::BFloat16* X,
    const at::BFloat16* alpha0,
    const at::BFloat16* alpha1,
    const at::BFloat16* alpha2,
    const at::BFloat16* alpha3,
    at::BFloat16* Y0,
    at::BFloat16* Y1,
    at::BFloat16* Y2,
    at::BFloat16* Y3,
    int count,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id0
) {
    size_t shmem = static_cast<size_t>(d_in) * sizeof(float);
    cudaFuncSetAttribute(
        deterministic_tlinear_forward_multi_fwht_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem)
    );
    dim3 block(512, 1, 1);
    dim3 grid(T, 1, 1);
    deterministic_tlinear_forward_multi_fwht_kernel<<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X),
        reinterpret_cast<const __nv_bfloat16*>(alpha0),
        reinterpret_cast<const __nv_bfloat16*>(alpha1),
        reinterpret_cast<const __nv_bfloat16*>(alpha2),
        reinterpret_cast<const __nv_bfloat16*>(alpha3),
        reinterpret_cast<__nv_bfloat16*>(Y0),
        reinterpret_cast<__nv_bfloat16*>(Y1),
        reinterpret_cast<__nv_bfloat16*>(Y2),
        reinterpret_cast<__nv_bfloat16*>(Y3),
        count,
        T,
        d_in,
        d_out,
        layer,
        matrix_id0
    );
}

void launch_deterministic_tlinear_backward_input(
    const at::BFloat16* dY,
    const at::BFloat16* alpha,
    float* dX,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
) {
    size_t shmem = static_cast<size_t>(d_in) * sizeof(float);
    cudaFuncSetAttribute(
        deterministic_tlinear_backward_input_fwht_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem)
    );
    dim3 block(512, 1, 1);
    dim3 grid(T, 1, 1);
    deterministic_tlinear_backward_input_fwht_kernel<<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(dY),
        reinterpret_cast<const __nv_bfloat16*>(alpha),
        dX,
        T,
        d_in,
        d_out,
        layer,
        matrix_id
    );
}

void launch_embedding_forward(
    const uint16_t* token_ids,
    const uint8_t* Epack,
    const at::BFloat16* alpha,
    at::BFloat16* X,
    int T,
    int d
) {
    init_trit_lut_cuda();
    int stride = row_stride_bytes(d);
    int total = T * d;
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);

    embedding_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        token_ids,
        Epack,
        reinterpret_cast<const __nv_bfloat16*>(alpha),
        reinterpret_cast<__nv_bfloat16*>(X),
        T,
        d,
        stride
    );
}

void launch_recurrent_mixer_forward(
    const float* i_gate,
    const float* f_gate,
    const float* v_val,
    const float* r_gate,
    const at::BFloat16* state,
    const at::BFloat16* lambda_raw,
    const at::BFloat16* beta_raw,
    at::BFloat16* g,
    at::BFloat16* new_state,
    int T,
    int R,
    int d
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(d, 256), 1, 1);
    recurrent_mixer_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        i_gate,
        f_gate,
        v_val,
        r_gate,
        reinterpret_cast<const __nv_bfloat16*>(state),
        reinterpret_cast<const __nv_bfloat16*>(lambda_raw),
        reinterpret_cast<const __nv_bfloat16*>(beta_raw),
        reinterpret_cast<__nv_bfloat16*>(g),
        reinterpret_cast<__nv_bfloat16*>(new_state),
        T,
        R,
        d
    );
}

void launch_recurrent_mixer_forward_trace(
    const float* i_gate,
    const float* f_gate,
    const float* v_val,
    const float* r_gate,
    const at::BFloat16* state,
    const at::BFloat16* lambda_raw,
    const at::BFloat16* beta_raw,
    at::BFloat16* g,
    at::BFloat16* new_state,
    at::BFloat16* m_trace,
    int T,
    int R,
    int d
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(d, 256), 1, 1);
    recurrent_mixer_forward_trace_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        i_gate,
        f_gate,
        v_val,
        r_gate,
        reinterpret_cast<const __nv_bfloat16*>(state),
        reinterpret_cast<const __nv_bfloat16*>(lambda_raw),
        reinterpret_cast<const __nv_bfloat16*>(beta_raw),
        reinterpret_cast<__nv_bfloat16*>(g),
        reinterpret_cast<__nv_bfloat16*>(new_state),
        reinterpret_cast<__nv_bfloat16*>(m_trace),
        T,
        R,
        d
    );
}

void launch_recurrent_mixer_backward(
    const float* i_gate,
    const float* f_gate,
    const float* v_val,
    const float* r_gate,
    const at::BFloat16* state,
    const at::BFloat16* lambda_raw,
    const at::BFloat16* beta_raw,
    const float* grad_g,
    float* d_i,
    float* d_f,
    float* d_v,
    float* d_r,
    at::BFloat16* d_state,
    int T,
    int R,
    int d
) {
    dim3 block(128, 1, 1);
    dim3 grid(ceil_div_int(d, 128), 1, 1);
    recurrent_mixer_backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        i_gate,
        f_gate,
        v_val,
        r_gate,
        reinterpret_cast<const __nv_bfloat16*>(state),
        reinterpret_cast<const __nv_bfloat16*>(lambda_raw),
        reinterpret_cast<const __nv_bfloat16*>(beta_raw),
        grad_g,
        d_i,
        d_f,
        d_v,
        d_r,
        reinterpret_cast<__nv_bfloat16*>(d_state),
        T,
        R,
        d
    );
}

void launch_recurrent_mixer_backward_trace(
    const float* i_gate,
    const float* f_gate,
    const float* v_val,
    const float* r_gate,
    const at::BFloat16* state,
    const at::BFloat16* lambda_raw,
    const at::BFloat16* beta_raw,
    const at::BFloat16* m_trace,
    const float* grad_g,
    float* d_i,
    float* d_f,
    float* d_v,
    float* d_r,
    at::BFloat16* d_state,
    int T,
    int R,
    int d
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(d, 256), 1, 1);
    recurrent_mixer_backward_trace_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        i_gate,
        f_gate,
        v_val,
        r_gate,
        reinterpret_cast<const __nv_bfloat16*>(state),
        reinterpret_cast<const __nv_bfloat16*>(lambda_raw),
        reinterpret_cast<const __nv_bfloat16*>(beta_raw),
        reinterpret_cast<const __nv_bfloat16*>(m_trace),
        grad_g,
        d_i,
        d_f,
        d_v,
        d_r,
        reinterpret_cast<__nv_bfloat16*>(d_state),
        T,
        R,
        d
    );
}

void launch_rms_norm_forward(
    const at::BFloat16* x,
    const at::BFloat16* gamma,
    at::BFloat16* y,
    int T,
    int d,
    float eps
) {
    dim3 block(256, 1, 1);
    dim3 grid(T, 1, 1);
    rms_norm_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        reinterpret_cast<__nv_bfloat16*>(y),
        T,
        d,
        eps
    );
}

void launch_rms_norm_backward(
    const at::BFloat16* x,
    const at::BFloat16* gamma,
    const float* grad_y,
    float* grad_x,
    int T,
    int d,
    float eps
) {
    dim3 block(256, 1, 1);
    dim3 grid(T, 1, 1);
    rms_norm_backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gamma),
        grad_y,
        grad_x,
        T,
        d,
        eps
    );
}

void launch_activation_forward(
    const at::BFloat16* x,
    float* y,
    int total,
    int activation
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);
    activation_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        y,
        total,
        activation
    );
}

void launch_activation_backward(
    const at::BFloat16* x,
    const float* grad_out,
    float* grad_x,
    int total,
    int activation
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);
    activation_backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        grad_out,
        grad_x,
        total,
        activation
    );
}

void launch_activation_backward_bf16(
    const at::BFloat16* x,
    const float* grad_out,
    at::BFloat16* grad_x,
    int total,
    int activation
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);
    activation_backward_bf16_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        grad_out,
        reinterpret_cast<__nv_bfloat16*>(grad_x),
        total,
        activation
    );
}

void launch_gated_silu_product(
    const at::BFloat16* a,
    const at::BFloat16* b,
    at::BFloat16* h,
    int total
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);
    gated_silu_product_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(h),
        total
    );
}

void launch_swiglu_backward(
    const at::BFloat16* a,
    const at::BFloat16* b,
    const float* grad_h,
    float* grad_a,
    float* grad_b,
    int total
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);
    swiglu_backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        grad_h,
        grad_a,
        grad_b,
        total
    );
}

void launch_swiglu_backward_bf16(
    const at::BFloat16* a,
    const at::BFloat16* b,
    const float* grad_h,
    at::BFloat16* grad_a,
    at::BFloat16* grad_b,
    int total
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);
    swiglu_backward_bf16_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        grad_h,
        reinterpret_cast<__nv_bfloat16*>(grad_a),
        reinterpret_cast<__nv_bfloat16*>(grad_b),
        total
    );
}

void launch_residual_add_forward(
    const at::BFloat16* base,
    const at::BFloat16* ternary,
    const at::BFloat16* adapter,
    at::BFloat16* out,
    int total,
    float rho
) {
    dim3 block(256, 1, 1);
    dim3 grid(ceil_div_int(total, 256), 1, 1);
    residual_add_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(base),
        reinterpret_cast<const __nv_bfloat16*>(ternary),
        reinterpret_cast<const __nv_bfloat16*>(adapter),
        reinterpret_cast<__nv_bfloat16*>(out),
        total,
        rho
    );
}

void launch_adapter_forward(
    const at::BFloat16* x,
    const float* A,
    const float* B,
    float* hidden,
    at::BFloat16* out,
    int T,
    int d,
    int r
) {
    dim3 block_a(256, 1, 1);
    dim3 grid_a(T, r, 1);
    adapter_a_forward_kernel<<<grid_a, block_a, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        A,
        hidden,
        T,
        d,
        r
    );

    dim3 block_b(256, 1, 1);
    dim3 grid_b(ceil_div_int(T * d, 256), 1, 1);
    adapter_b_forward_kernel<<<grid_b, block_b, 0, at::cuda::getCurrentCUDAStream()>>>(
        hidden,
        B,
        reinterpret_cast<__nv_bfloat16*>(out),
        T,
        d,
        r
    );
}

void pack_ternary_matrix_cpu(
    const int8_t* T,
    uint8_t* Wpack,
    int d_out,
    int d_in
) {
    int stride = row_stride_bytes(d_in);

    for (int o = 0; o < d_out; ++o) {
        for (int kb = 0; kb < stride; ++kb) {
            uint32_t byte_value = 0;

            for (int r = 0; r < 5; ++r) {
                int i = kb * 5 + r;
                int c = 1;

                if (i < d_in) {
                    int8_t w = T[o * d_in + i];

                    if (w == -1) {
                        c = 0;
                    } else if (w == 0) {
                        c = 1;
                    } else if (w == 1) {
                        c = 2;
                    } else {
                        std::abort();
                    }
                }

                byte_value += static_cast<uint32_t>(c) * static_cast<uint32_t>(POW3[r]);
            }

            if (byte_value > 242u) {
                std::abort();
            }

            Wpack[o * stride + kb] = static_cast<uint8_t>(byte_value);
        }
    }
}

void launch_deterministic_pack_ternary_cuda(
    uint8_t* Wpack,
    at::BFloat16* alpha,
    int d_out,
    int d_in,
    int layer,
    int matrix_id
) {
    int stride = row_stride_bytes(d_in);
    int total = d_out * stride;
    dim3 block_pack(256, 1, 1);
    dim3 grid_pack(ceil_div_int(total, 256), 1, 1);
    deterministic_pack_kernel<<<grid_pack, block_pack, 0, at::cuda::getCurrentCUDAStream()>>>(
        Wpack,
        d_out,
        d_in,
        stride,
        layer,
        matrix_id
    );

    dim3 block_alpha(256, 1, 1);
    dim3 grid_alpha(d_out, 1, 1);
    deterministic_alpha_kernel<<<grid_alpha, block_alpha, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__nv_bfloat16*>(alpha),
        d_out,
        d_in,
        layer,
        matrix_id
    );
}
