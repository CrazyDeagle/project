#include "tlinear.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

constexpr int S_MAX = 8192;
constexpr int D_MODEL = 4096;
constexpr int D_FF = 16384;
constexpr int D_Z = 8192;
constexpr int VOCAB_SIZE = 258;

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void cuda_sync_checked(const char* phase) {
    cudaError_t err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, phase, ": ", cudaGetErrorString(err));
}

struct TrainProfiler {
    bool enabled;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point last;

    explicit TrainProfiler(bool enabled_) : enabled(enabled_) {
        start = std::chrono::steady_clock::now();
        last = start;
    }

    void mark(const char* phase) {
        if (!enabled) {
            return;
        }
        cuda_sync_checked(phase);
        auto now = std::chrono::steady_clock::now();
        double total = std::chrono::duration<double>(now - start).count();
        double delta = std::chrono::duration<double>(now - last).count();
        last = now;
        std::cout << "native_phase=" << phase
                  << " seconds_total=" << total
                  << " seconds_delta=" << delta
                  << std::endl;
    }
};

struct TrainWorkspaceLayout {
    int64_t x_offset;
    int64_t rec_trace_offset;
    int64_t z_trace_offset;
    int64_t ff_offset;
    int64_t mix_offset;
    int64_t logits_offset;
    int64_t total_bytes;
};

TrainWorkspaceLayout make_train_workspace_layout(int64_t sequence_len) {
    TORCH_CHECK(sequence_len == 512, "native training workspace is fixed to U_train=512");
    int64_t offset = 0;
    TrainWorkspaceLayout layout{};
    layout.x_offset = offset;
    offset += 65ll * sequence_len * D_MODEL * 2ll;
    layout.rec_trace_offset = offset;
    offset += 64ll * sequence_len * 8ll * D_MODEL * 2ll;
    layout.z_trace_offset = offset;
    offset += 5ll * sequence_len * D_MODEL * 2ll;
    layout.ff_offset = offset;
    offset += 3ll * sequence_len * D_FF * 2ll;
    layout.mix_offset = offset;
    offset += 6ll * sequence_len * D_MODEL * 2ll;
    layout.logits_offset = offset;
    offset += sequence_len * VOCAB_SIZE * 4ll;
    layout.total_bytes = offset;
    return layout;
}

bool valid_d_in(int d_in) {
    return d_in == D_MODEL || d_in == D_Z || d_in == D_FF;
}

bool valid_d_out(int d_out) {
    return d_out == VOCAB_SIZE || d_out == D_MODEL || d_out == D_Z || d_out == D_FF;
}

void validate_tlinear_shape(const torch::Tensor& X, const torch::Tensor& Wpack, const torch::Tensor& alpha) {
    TORCH_CHECK(X.is_cuda(), "X must be CUDA");
    TORCH_CHECK(Wpack.is_cuda(), "Wpack must be CUDA");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16, "X must be bfloat16");
    TORCH_CHECK(alpha.scalar_type() == torch::kBFloat16, "alpha must be bfloat16");
    TORCH_CHECK(Wpack.scalar_type() == torch::kUInt8, "Wpack must be uint8");
    TORCH_CHECK(X.dim() == 2, "X must have shape [T, d_in]");
    TORCH_CHECK(Wpack.dim() == 2, "Wpack must have shape [d_out, S5(d_in)]");
    TORCH_CHECK(alpha.dim() == 1, "alpha must have shape [d_out]");

    int64_t T = X.size(0);
    int64_t d_in = X.size(1);
    int64_t d_out = Wpack.size(0);
    int64_t stride = Wpack.size(1);

    TORCH_CHECK(T >= 1 && T <= S_MAX, "T must be in [1, 8192]");
    TORCH_CHECK(valid_d_in(static_cast<int>(d_in)), "d_in must be one of {4096, 8192, 16384}");
    TORCH_CHECK(valid_d_out(static_cast<int>(d_out)), "d_out must be one of {258, 4096, 8192, 16384}");
    TORCH_CHECK(stride == row_stride_bytes(static_cast<int>(d_in)), "Wpack row stride must equal S5(d_in)");
    TORCH_CHECK(alpha.size(0) == d_out, "alpha length must equal d_out");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(Wpack.is_contiguous(), "Wpack must be contiguous");
    TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
}

torch::Tensor tlinear_forward_native(torch::Tensor X, torch::Tensor Wpack, torch::Tensor alpha, int d_in, int d_out) {
    TORCH_CHECK(X.is_contiguous() && Wpack.is_contiguous() && alpha.is_contiguous(), "native TLinear tensors must be contiguous");
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && alpha.scalar_type() == torch::kBFloat16, "native TLinear bf16 dtype mismatch");
    TORCH_CHECK(Wpack.scalar_type() == torch::kUInt8, "native TLinear Wpack must be uint8");
    TORCH_CHECK(X.size(1) == d_in, "native TLinear input dim mismatch");
    TORCH_CHECK(Wpack.size(0) == d_out && Wpack.size(1) == row_stride_bytes(d_in), "native TLinear Wpack shape mismatch");
    TORCH_CHECK(alpha.size(0) == d_out, "native TLinear alpha shape mismatch");
    auto Y = torch::empty({X.size(0), d_out}, X.options());
    launch_tlinear_forward(
        X.data_ptr<at::BFloat16>(),
        Wpack.data_ptr<uint8_t>(),
        alpha.data_ptr<at::BFloat16>(),
        Y.data_ptr<at::BFloat16>(),
        static_cast<int>(X.size(0)),
        d_in,
        d_out
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor deterministic_tlinear_forward_native(
    torch::Tensor X,
    torch::Tensor alpha,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
) {
    TORCH_CHECK(X.is_contiguous() && alpha.is_contiguous(), "deterministic native TLinear tensors must be contiguous");
    TORCH_CHECK(X.scalar_type() == torch::kBFloat16 && alpha.scalar_type() == torch::kBFloat16, "deterministic native TLinear bf16 dtype mismatch");
    TORCH_CHECK(X.size(1) == d_in && alpha.size(0) == d_out, "deterministic native TLinear shape mismatch");
    auto Y = torch::empty({X.size(0), d_out}, X.options());
    launch_deterministic_tlinear_forward(
        X.data_ptr<at::BFloat16>(),
        alpha.data_ptr<at::BFloat16>(),
        Y.data_ptr<at::BFloat16>(),
        static_cast<int>(X.size(0)),
        d_in,
        d_out,
        layer,
        matrix_id
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

std::vector<torch::Tensor> deterministic_tlinear_forward_multi_native(
    torch::Tensor X,
    std::vector<torch::Tensor> alphas,
    int d_in,
    int d_out,
    int layer,
    int matrix_id0
) {
    TORCH_CHECK(!alphas.empty() && alphas.size() <= 4, "deterministic multi TLinear supports 1..4 outputs");
    TORCH_CHECK(X.is_contiguous() && X.scalar_type() == torch::kBFloat16, "deterministic multi TLinear input must be contiguous bf16");
    int count = static_cast<int>(alphas.size());
    std::vector<torch::Tensor> outputs;
    outputs.reserve(alphas.size());
    for (size_t i = 0; i < alphas.size(); ++i) {
        TORCH_CHECK(alphas[i].is_contiguous() && alphas[i].scalar_type() == torch::kBFloat16 && alphas[i].size(0) == d_out, "deterministic multi TLinear alpha mismatch");
        outputs.push_back(torch::empty({X.size(0), d_out}, X.options()));
    }
    while (outputs.size() < 4) {
        outputs.push_back(outputs[0]);
    }
    while (alphas.size() < 4) {
        alphas.push_back(alphas[0]);
    }
    launch_deterministic_tlinear_forward_multi(
        X.data_ptr<at::BFloat16>(),
        alphas[0].data_ptr<at::BFloat16>(),
        alphas[1].data_ptr<at::BFloat16>(),
        alphas[2].data_ptr<at::BFloat16>(),
        alphas[3].data_ptr<at::BFloat16>(),
        outputs[0].data_ptr<at::BFloat16>(),
        outputs[1].data_ptr<at::BFloat16>(),
        outputs[2].data_ptr<at::BFloat16>(),
        outputs[3].data_ptr<at::BFloat16>(),
        count,
        static_cast<int>(X.size(0)),
        d_in,
        d_out,
        layer,
        matrix_id0
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    outputs.resize(static_cast<size_t>(count));
    return outputs;
}

torch::Tensor rms_norm_native(torch::Tensor x, torch::Tensor gamma, float eps) {
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous(), "native RMS tensors must be contiguous");
    auto y = torch::empty_like(x);
    launch_rms_norm_forward(
        x.data_ptr<at::BFloat16>(),
        gamma.data_ptr<at::BFloat16>(),
        y.data_ptr<at::BFloat16>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1)),
        eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor activation_native(torch::Tensor x, int activation) {
    TORCH_CHECK(x.is_contiguous(), "native activation tensor must be contiguous");
    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
    launch_activation_forward(
        x.data_ptr<at::BFloat16>(),
        y.data_ptr<float>(),
        static_cast<int>(x.numel()),
        activation
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor gated_silu_native(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "native gated tensors must be contiguous");
    auto h = torch::empty_like(a);
    launch_gated_silu_product(
        a.data_ptr<at::BFloat16>(),
        b.data_ptr<at::BFloat16>(),
        h.data_ptr<at::BFloat16>(),
        static_cast<int>(a.numel())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return h;
}

torch::Tensor residual_native(torch::Tensor base, torch::Tensor ternary, torch::Tensor adapter, float rho) {
    TORCH_CHECK(base.is_contiguous() && ternary.is_contiguous() && adapter.is_contiguous(), "native residual tensors must be contiguous");
    auto out = torch::empty_like(base);
    launch_residual_add_forward(
        base.data_ptr<at::BFloat16>(),
        ternary.data_ptr<at::BFloat16>(),
        adapter.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        static_cast<int>(base.numel()),
        rho
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.to(torch::kBFloat16);
}

torch::Tensor adapter_native(torch::Tensor x, torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(x.is_contiguous() && A.is_contiguous() && B.is_contiguous(), "native adapter tensors must be contiguous");
    auto hidden = torch::empty({x.size(0), 64}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty_like(x);
    launch_adapter_forward(
        x.data_ptr<at::BFloat16>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        hidden.data_ptr<float>(),
        out.data_ptr<at::BFloat16>(),
        static_cast<int>(x.size(0)),
        D_MODEL,
        64
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.to(torch::kBFloat16);
}

std::tuple<torch::Tensor, torch::Tensor> recurrent_native(
    torch::Tensor i_gate,
    torch::Tensor f_gate,
    torch::Tensor v_val,
    torch::Tensor r_gate,
    torch::Tensor state,
    torch::Tensor lambda_raw,
    torch::Tensor beta_raw
) {
    auto g = torch::empty(i_gate.sizes(), state.options());
    auto new_state = torch::empty_like(state);
    launch_recurrent_mixer_forward(
        i_gate.data_ptr<float>(),
        f_gate.data_ptr<float>(),
        v_val.data_ptr<float>(),
        r_gate.data_ptr<float>(),
        state.data_ptr<at::BFloat16>(),
        lambda_raw.data_ptr<at::BFloat16>(),
        beta_raw.data_ptr<at::BFloat16>(),
        g.data_ptr<at::BFloat16>(),
        new_state.data_ptr<at::BFloat16>(),
        static_cast<int>(i_gate.size(0)),
        8,
        D_MODEL
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(g, new_state);
}

std::tuple<torch::Tensor, torch::Tensor> recurrent_native_trace(
    torch::Tensor i_gate,
    torch::Tensor f_gate,
    torch::Tensor v_val,
    torch::Tensor r_gate,
    torch::Tensor state,
    torch::Tensor lambda_raw,
    torch::Tensor beta_raw,
    torch::Tensor m_trace
) {
    auto g = torch::empty(i_gate.sizes(), state.options());
    auto new_state = torch::empty_like(state);
    launch_recurrent_mixer_forward_trace(
        i_gate.data_ptr<float>(),
        f_gate.data_ptr<float>(),
        v_val.data_ptr<float>(),
        r_gate.data_ptr<float>(),
        state.data_ptr<at::BFloat16>(),
        lambda_raw.data_ptr<at::BFloat16>(),
        beta_raw.data_ptr<at::BFloat16>(),
        g.data_ptr<at::BFloat16>(),
        new_state.data_ptr<at::BFloat16>(),
        m_trace.data_ptr<at::BFloat16>(),
        static_cast<int>(i_gate.size(0)),
        8,
        D_MODEL
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(g, new_state);
}

torch::Tensor embedding_native(torch::Tensor token_ids, torch::Tensor Epack, torch::Tensor alpha) {
    auto X = torch::empty({token_ids.size(0), D_MODEL}, alpha.options());
    launch_embedding_forward(
        token_ids.data_ptr<uint16_t>(),
        Epack.data_ptr<uint8_t>(),
        alpha.data_ptr<at::BFloat16>(),
        X.data_ptr<at::BFloat16>(),
        static_cast<int>(token_ids.size(0)),
        D_MODEL
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return X;
}

int8_t hadamard_value(int64_t a, int64_t b) {
    uint64_t x = static_cast<uint64_t>(a & b);
#if defined(_MSC_VER)
    int parity = static_cast<int>(__popcnt64(x) & 1);
#else
    int parity = static_cast<int>(__builtin_popcountll(x) & 1);
#endif
    return parity == 0 ? static_cast<int8_t>(1) : static_cast<int8_t>(-1);
}

int8_t deterministic_trit(int o, int i, int layer, int matrix_id, int d_in) {
    int64_t z = 1103515245ll * static_cast<int64_t>(o + 1)
        + 12345ll * static_cast<int64_t>(i + 1)
        + 97ll * static_cast<int64_t>(matrix_id)
        + 131ll * static_cast<int64_t>(layer);
    if ((z % 16) == 0) {
        return 0;
    }
    return hadamard_value(o % d_in, i);
}

} // namespace

torch::Tensor tlinear_forward(torch::Tensor X, torch::Tensor Wpack, torch::Tensor alpha) {
    validate_tlinear_shape(X, Wpack, alpha);
    const c10::cuda::CUDAGuard device_guard(X.device());

    auto Y = torch::empty({X.size(0), Wpack.size(0)}, X.options());
    launch_tlinear_forward(
        X.data_ptr<at::BFloat16>(),
        Wpack.data_ptr<uint8_t>(),
        alpha.data_ptr<at::BFloat16>(),
        Y.data_ptr<at::BFloat16>(),
        static_cast<int>(X.size(0)),
        static_cast<int>(X.size(1)),
        static_cast<int>(Wpack.size(0))
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor tlinear_backward_input(torch::Tensor dY, torch::Tensor Wpack, torch::Tensor alpha, int64_t d_in) {
    TORCH_CHECK(dY.is_cuda(), "dY must be CUDA");
    TORCH_CHECK(Wpack.is_cuda(), "Wpack must be CUDA");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
    TORCH_CHECK(dY.scalar_type() == torch::kBFloat16, "dY must be bfloat16");
    TORCH_CHECK(alpha.scalar_type() == torch::kBFloat16, "alpha must be bfloat16");
    TORCH_CHECK(Wpack.scalar_type() == torch::kUInt8, "Wpack must be uint8");
    TORCH_CHECK(dY.dim() == 2, "dY must have shape [T, d_out]");
    TORCH_CHECK(Wpack.dim() == 2, "Wpack must have shape [d_out, S5(d_in)]");
    TORCH_CHECK(dY.is_contiguous(), "dY must be contiguous");
    TORCH_CHECK(Wpack.is_contiguous(), "Wpack must be contiguous");
    TORCH_CHECK(alpha.is_contiguous(), "alpha must be contiguous");
    TORCH_CHECK(valid_d_in(static_cast<int>(d_in)), "d_in must be one of {4096, 8192, 16384}");
    TORCH_CHECK(dY.size(0) >= 1 && dY.size(0) <= S_MAX, "T must be in [1, 8192]");
    TORCH_CHECK(dY.size(1) == Wpack.size(0), "dY d_out must match Wpack rows");
    TORCH_CHECK(alpha.size(0) == Wpack.size(0), "alpha length must equal d_out");
    TORCH_CHECK(Wpack.size(1) == row_stride_bytes(static_cast<int>(d_in)), "Wpack row stride must equal S5(d_in)");

    const c10::cuda::CUDAGuard device_guard(dY.device());
    auto dX = torch::empty({dY.size(0), d_in}, dY.options().dtype(torch::kFloat32));
    launch_tlinear_backward_input(
        dY.data_ptr<at::BFloat16>(),
        Wpack.data_ptr<uint8_t>(),
        alpha.data_ptr<at::BFloat16>(),
        dX.data_ptr<float>(),
        static_cast<int>(dY.size(0)),
        static_cast<int>(d_in),
        static_cast<int>(Wpack.size(0))
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dX;
}

torch::Tensor embedding_forward(torch::Tensor token_ids, torch::Tensor Epack, torch::Tensor alpha) {
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA");
    TORCH_CHECK(Epack.is_cuda(), "Epack must be CUDA");
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
    TORCH_CHECK(token_ids.scalar_type() == torch::kUInt16, "token_ids must be uint16");
    TORCH_CHECK(Epack.scalar_type() == torch::kUInt8, "Epack must be uint8");
    TORCH_CHECK(alpha.scalar_type() == torch::kBFloat16, "alpha must be bfloat16");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must have shape [T]");
    TORCH_CHECK(Epack.dim() == 2 && Epack.size(0) == VOCAB_SIZE, "Epack must have shape [258, S5(4096)]");
    TORCH_CHECK(Epack.size(1) == row_stride_bytes(D_MODEL), "Epack row stride must equal S5(4096)");
    TORCH_CHECK(alpha.dim() == 1 && alpha.size(0) == VOCAB_SIZE, "alpha must have shape [258]");
    TORCH_CHECK(token_ids.size(0) >= 1 && token_ids.size(0) <= S_MAX, "T must be in [1, 8192]");

    const c10::cuda::CUDAGuard device_guard(token_ids.device());
    auto X = torch::empty({token_ids.size(0), D_MODEL}, alpha.options());
    launch_embedding_forward(
        token_ids.data_ptr<uint16_t>(),
        Epack.data_ptr<uint8_t>(),
        alpha.data_ptr<at::BFloat16>(),
        X.data_ptr<at::BFloat16>(),
        static_cast<int>(token_ids.size(0)),
        D_MODEL
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return X;
}

std::tuple<torch::Tensor, torch::Tensor> recurrent_mixer_forward(
    torch::Tensor i_gate,
    torch::Tensor f_gate,
    torch::Tensor v_val,
    torch::Tensor r_gate,
    torch::Tensor state,
    torch::Tensor lambda_raw,
    torch::Tensor beta_raw
) {
    TORCH_CHECK(i_gate.is_cuda() && f_gate.is_cuda() && v_val.is_cuda() && r_gate.is_cuda(), "gates must be CUDA");
    TORCH_CHECK(state.is_cuda() && lambda_raw.is_cuda() && beta_raw.is_cuda(), "state and recurrent params must be CUDA");
    TORCH_CHECK(i_gate.scalar_type() == torch::kFloat32, "i_gate must be fp32");
    TORCH_CHECK(f_gate.scalar_type() == torch::kFloat32, "f_gate must be fp32");
    TORCH_CHECK(v_val.scalar_type() == torch::kFloat32, "v_val must be fp32");
    TORCH_CHECK(r_gate.scalar_type() == torch::kFloat32, "r_gate must be fp32");
    TORCH_CHECK(state.scalar_type() == torch::kBFloat16, "state must be bf16");
    TORCH_CHECK(lambda_raw.scalar_type() == torch::kBFloat16, "lambda_raw must be bf16");
    TORCH_CHECK(beta_raw.scalar_type() == torch::kBFloat16, "beta_raw must be bf16");
    TORCH_CHECK(i_gate.dim() == 2, "gates must have shape [T, d]");
    TORCH_CHECK(f_gate.sizes() == i_gate.sizes(), "f_gate shape mismatch");
    TORCH_CHECK(v_val.sizes() == i_gate.sizes(), "v_val shape mismatch");
    TORCH_CHECK(r_gate.sizes() == i_gate.sizes(), "r_gate shape mismatch");
    TORCH_CHECK(state.dim() == 2 && lambda_raw.dim() == 2 && beta_raw.dim() == 2, "state/lambda/beta must have shape [R, d]");
    TORCH_CHECK(state.sizes() == lambda_raw.sizes() && state.sizes() == beta_raw.sizes(), "state/lambda/beta shape mismatch");
    TORCH_CHECK(i_gate.size(0) >= 1 && i_gate.size(0) <= S_MAX, "T must be in [1, 8192]");
    TORCH_CHECK(i_gate.size(1) == D_MODEL, "d must be 4096");
    TORCH_CHECK(state.size(0) == 8 && state.size(1) == D_MODEL, "state must have shape [8,4096]");
    TORCH_CHECK(i_gate.is_contiguous() && f_gate.is_contiguous() && v_val.is_contiguous() && r_gate.is_contiguous(), "gates must be contiguous");
    TORCH_CHECK(state.is_contiguous() && lambda_raw.is_contiguous() && beta_raw.is_contiguous(), "state/lambda/beta must be contiguous");

    const c10::cuda::CUDAGuard device_guard(i_gate.device());
    auto g = torch::empty(i_gate.sizes(), state.options());
    auto new_state = torch::empty_like(state);
    launch_recurrent_mixer_forward(
        i_gate.data_ptr<float>(),
        f_gate.data_ptr<float>(),
        v_val.data_ptr<float>(),
        r_gate.data_ptr<float>(),
        state.data_ptr<at::BFloat16>(),
        lambda_raw.data_ptr<at::BFloat16>(),
        beta_raw.data_ptr<at::BFloat16>(),
        g.data_ptr<at::BFloat16>(),
        new_state.data_ptr<at::BFloat16>(),
        static_cast<int>(i_gate.size(0)),
        static_cast<int>(state.size(0)),
        static_cast<int>(i_gate.size(1))
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(g, new_state);
}

torch::Tensor rms_norm_forward(torch::Tensor x, torch::Tensor gamma, double eps) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda(), "x and gamma must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16, "gamma must be bf16");
    TORCH_CHECK(x.dim() == 2, "x must have shape [T, d]");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= S_MAX, "T must be in [1, 8192]");
    TORCH_CHECK(x.size(1) == D_MODEL, "RMSNorm d must be 4096");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == D_MODEL, "gamma must have shape [4096]");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous(), "x and gamma must be contiguous");
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty_like(x);
    launch_rms_norm_forward(
        x.data_ptr<at::BFloat16>(),
        gamma.data_ptr<at::BFloat16>(),
        y.data_ptr<at::BFloat16>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1)),
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor activation_forward(torch::Tensor x, int64_t activation) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(activation == 0 || activation == 1, "activation must be 0 sigmoid or 1 silu");
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
    launch_activation_forward(
        x.data_ptr<at::BFloat16>(),
        y.data_ptr<float>(),
        static_cast<int>(x.numel()),
        static_cast<int>(activation)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor gated_silu_product(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && b.scalar_type() == torch::kBFloat16, "a and b must be bf16");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b shapes must match");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
    const c10::cuda::CUDAGuard device_guard(a.device());
    auto h = torch::empty_like(a);
    launch_gated_silu_product(
        a.data_ptr<at::BFloat16>(),
        b.data_ptr<at::BFloat16>(),
        h.data_ptr<at::BFloat16>(),
        static_cast<int>(a.numel())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return h;
}

torch::Tensor residual_add_forward(torch::Tensor base, torch::Tensor ternary, torch::Tensor adapter, double rho) {
    TORCH_CHECK(base.is_cuda() && ternary.is_cuda() && adapter.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(base.scalar_type() == torch::kBFloat16, "base must be bf16");
    TORCH_CHECK(ternary.scalar_type() == torch::kBFloat16, "ternary must be bf16");
    TORCH_CHECK(adapter.scalar_type() == torch::kBFloat16, "adapter must be bf16");
    TORCH_CHECK(base.sizes() == ternary.sizes() && base.sizes() == adapter.sizes(), "residual inputs must have identical shapes");
    TORCH_CHECK(base.is_contiguous() && ternary.is_contiguous() && adapter.is_contiguous(), "residual inputs must be contiguous");
    const c10::cuda::CUDAGuard device_guard(base.device());
    auto out = torch::empty_like(base);
    launch_residual_add_forward(
        base.data_ptr<at::BFloat16>(),
        ternary.data_ptr<at::BFloat16>(),
        adapter.data_ptr<at::BFloat16>(),
        out.data_ptr<at::BFloat16>(),
        static_cast<int>(base.numel()),
        static_cast<float>(rho)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.to(torch::kBFloat16);
}

std::tuple<torch::Tensor, torch::Tensor> adapter_forward(torch::Tensor x, torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(x.is_cuda() && A.is_cuda() && B.is_cuda(), "x, A and B must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "A and B must be fp32");
    TORCH_CHECK(x.dim() == 2 && A.dim() == 2 && B.dim() == 2, "adapter tensors must be matrices");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= S_MAX, "T must be in [1, 8192]");
    TORCH_CHECK(x.size(1) == D_MODEL, "adapter input d must be 4096");
    TORCH_CHECK(A.size(0) == 64 && A.size(1) == D_MODEL, "A must have shape [64,4096]");
    TORCH_CHECK(B.size(0) == D_MODEL && B.size(1) == 64, "B must have shape [4096,64]");
    TORCH_CHECK(x.is_contiguous() && A.is_contiguous() && B.is_contiguous(), "adapter tensors must be contiguous");
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto hidden = torch::empty({x.size(0), 64}, x.options().dtype(torch::kFloat32));
    auto out = torch::empty_like(x);
    launch_adapter_forward(
        x.data_ptr<at::BFloat16>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        hidden.data_ptr<float>(),
        out.data_ptr<at::BFloat16>(),
        static_cast<int>(x.size(0)),
        D_MODEL,
        64
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(out, hidden);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> adapter_backward_exact(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor grad_out
) {
    TORCH_CHECK(x.is_cuda() && A.is_cuda() && B.is_cuda() && grad_out.is_cuda(), "adapter backward tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "A and B must be fp32");
    TORCH_CHECK(x.dim() == 2 && A.dim() == 2 && B.dim() == 2 && grad_out.dim() == 2, "adapter backward tensors must be matrices");
    TORCH_CHECK(x.size(1) == D_MODEL && A.size(0) == 64 && A.size(1) == D_MODEL, "A/x shape mismatch");
    TORCH_CHECK(B.size(0) == D_MODEL && B.size(1) == 64, "B shape mismatch");
    TORCH_CHECK(grad_out.size(0) == x.size(0) && grad_out.size(1) == D_MODEL, "grad_out shape mismatch");

    const c10::cuda::CUDAGuard device_guard(x.device());
    auto xf = x.to(torch::kFloat32);
    auto go = grad_out.to(torch::kBFloat16).to(torch::kFloat32);
    auto hidden = torch::einsum("td,rd->tr", {xf, A});
    auto grad_hidden = torch::einsum("td,dr->tr", {go, B});
    auto grad_B = torch::einsum("td,tr->dr", {go, hidden});
    auto grad_A = torch::einsum("tr,td->rd", {grad_hidden, xf});
    auto grad_x = torch::einsum("tr,rd->td", {grad_hidden, A}).to(torch::kBFloat16);
    return std::make_tuple(grad_x, grad_A, grad_B);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> adapter_backward_with_factors(
    torch::Tensor x,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor grad_out
) {
    TORCH_CHECK(x.is_cuda() && A.is_cuda() && B.is_cuda() && grad_out.is_cuda(), "adapter backward tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "A and B must be fp32");
    TORCH_CHECK(x.dim() == 2 && A.dim() == 2 && B.dim() == 2 && grad_out.dim() == 2, "adapter backward tensors must be matrices");
    TORCH_CHECK(x.size(1) == D_MODEL && A.size(0) == 64 && A.size(1) == D_MODEL, "A/x shape mismatch");
    TORCH_CHECK(B.size(0) == D_MODEL && B.size(1) == 64, "B shape mismatch");
    TORCH_CHECK(grad_out.size(0) == x.size(0) && grad_out.size(1) == D_MODEL, "grad_out shape mismatch");

    const c10::cuda::CUDAGuard device_guard(x.device());
    auto xf = x.to(torch::kFloat32);
    auto go = grad_out.to(torch::kBFloat16).to(torch::kFloat32);
    auto hidden = torch::einsum("td,rd->tr", {xf, A});
    auto grad_hidden = torch::einsum("td,dr->tr", {go, B});
    auto grad_B = torch::einsum("td,tr->dr", {go, hidden});
    auto grad_A = torch::einsum("tr,td->rd", {grad_hidden, xf});
    auto grad_x = torch::einsum("tr,rd->td", {grad_hidden, A}).to(torch::kBFloat16);
    return std::make_tuple(grad_x, grad_A, grad_B, hidden, grad_hidden);
}

torch::Tensor activation_backward_exact(torch::Tensor preact, torch::Tensor grad_out, int64_t activation) {
    TORCH_CHECK(preact.is_cuda() && grad_out.is_cuda(), "activation backward tensors must be CUDA");
    TORCH_CHECK(preact.scalar_type() == torch::kBFloat16, "preact must be bf16");
    TORCH_CHECK(preact.sizes() == grad_out.sizes(), "activation backward shapes must match");
    TORCH_CHECK(activation == 0 || activation == 1, "activation must be 0 sigmoid or 1 silu");
    const c10::cuda::CUDAGuard device_guard(preact.device());
    auto grad = grad_out.to(torch::kFloat32).contiguous();
    auto x = preact.contiguous();
    auto out = torch::empty(preact.sizes(), preact.options().dtype(torch::kBFloat16));
    launch_activation_backward_bf16(
        x.data_ptr<at::BFloat16>(),
        grad.data_ptr<float>(),
        out.data_ptr<at::BFloat16>(),
        static_cast<int>(preact.numel()),
        static_cast<int>(activation)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor rms_norm_backward_exact(torch::Tensor x, torch::Tensor gamma, torch::Tensor grad_y, double eps) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda() && grad_y.is_cuda(), "RMSNorm backward tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && gamma.scalar_type() == torch::kBFloat16, "x and gamma must be bf16");
    TORCH_CHECK(x.dim() == 2 && gamma.dim() == 1 && grad_y.dim() == 2, "RMSNorm backward shape rank mismatch");
    TORCH_CHECK(x.size(1) == D_MODEL && gamma.size(0) == D_MODEL && grad_y.sizes() == x.sizes(), "RMSNorm backward shape mismatch");

    const c10::cuda::CUDAGuard device_guard(x.device());
    pybind11::gil_scoped_release no_gil;
    torch::AutoGradMode enable_grad(true);
    auto x_req = x.detach().clone();
    x_req.set_requires_grad(true);
    auto x_direct = x_req.to(torch::kFloat32);
    auto x_norm = x_req.to(torch::kFloat32);
    auto y = x_direct * torch::rsqrt(x_norm.square().mean(1, true) + static_cast<float>(eps)) * gamma.to(torch::kFloat32);
    auto grads = torch::autograd::grad({y}, {x_req}, {grad_y.to(torch::kFloat32)}, false, false);
    return grads[0];
}

torch::Tensor rms_norm_backward_native(torch::Tensor x, torch::Tensor gamma, torch::Tensor grad_y, double eps) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda() && grad_y.is_cuda(), "RMSNorm backward tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && gamma.scalar_type() == torch::kBFloat16, "x and gamma must be bf16");
    TORCH_CHECK(x.dim() == 2 && gamma.dim() == 1 && grad_y.dim() == 2, "RMSNorm backward shape rank mismatch");
    TORCH_CHECK(x.size(1) == D_MODEL && gamma.size(0) == D_MODEL && grad_y.sizes() == x.sizes(), "RMSNorm backward shape mismatch");
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto xc = x.contiguous();
    auto gc = gamma.contiguous();
    auto gy = grad_y.to(torch::kFloat32).contiguous();
    auto out = torch::empty(x.sizes(), x.options().dtype(torch::kFloat32));
    launch_rms_norm_backward(
        xc.data_ptr<at::BFloat16>(),
        gc.data_ptr<at::BFloat16>(),
        gy.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1)),
        static_cast<float>(eps)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> swiglu_backward_exact(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_h
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && grad_h.is_cuda(), "SwiGLU backward tensors must be CUDA");
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == grad_h.sizes(), "SwiGLU backward shapes must match");
    const c10::cuda::CUDAGuard device_guard(a.device());
    auto gh = grad_h.to(torch::kFloat32).contiguous();
    auto ac = a.contiguous();
    auto bc = b.contiguous();
    auto da = torch::empty(a.sizes(), a.options().dtype(torch::kBFloat16));
    auto db = torch::empty(b.sizes(), b.options().dtype(torch::kBFloat16));
    launch_swiglu_backward_bf16(
        ac.data_ptr<at::BFloat16>(),
        bc.data_ptr<at::BFloat16>(),
        gh.data_ptr<float>(),
        da.data_ptr<at::BFloat16>(),
        db.data_ptr<at::BFloat16>(),
        static_cast<int>(a.numel())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(da, db);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> recurrent_mixer_backward_exact(
    torch::Tensor i_gate,
    torch::Tensor f_gate,
    torch::Tensor v_val,
    torch::Tensor r_gate,
    torch::Tensor state,
    torch::Tensor lambda_raw,
    torch::Tensor beta_raw,
    torch::Tensor grad_g
) {
    TORCH_CHECK(i_gate.is_cuda() && f_gate.is_cuda() && v_val.is_cuda() && r_gate.is_cuda(), "gates must be CUDA");
    TORCH_CHECK(state.is_cuda() && lambda_raw.is_cuda() && beta_raw.is_cuda() && grad_g.is_cuda(), "state/recurrent/grad tensors must be CUDA");
    TORCH_CHECK(i_gate.scalar_type() == torch::kFloat32 && f_gate.scalar_type() == torch::kFloat32, "i_gate and f_gate must be fp32");
    TORCH_CHECK(v_val.scalar_type() == torch::kFloat32 && r_gate.scalar_type() == torch::kFloat32, "v_val and r_gate must be fp32");
    TORCH_CHECK(state.scalar_type() == torch::kBFloat16 && lambda_raw.scalar_type() == torch::kBFloat16 && beta_raw.scalar_type() == torch::kBFloat16, "state/lambda/beta must be bf16");
    TORCH_CHECK(i_gate.sizes() == f_gate.sizes() && i_gate.sizes() == v_val.sizes() && i_gate.sizes() == r_gate.sizes(), "gate shape mismatch");
    TORCH_CHECK(grad_g.sizes() == i_gate.sizes(), "grad_g shape mismatch");
    TORCH_CHECK(i_gate.dim() == 2 && i_gate.size(1) == D_MODEL && i_gate.size(0) >= 1 && i_gate.size(0) <= S_MAX, "gate shape must be [T,4096]");
    TORCH_CHECK(state.dim() == 2 && state.size(0) == 8 && state.size(1) == D_MODEL, "state must be [8,4096]");
    TORCH_CHECK(lambda_raw.sizes() == state.sizes() && beta_raw.sizes() == state.sizes(), "lambda/beta shape mismatch");

    const c10::cuda::CUDAGuard device_guard(i_gate.device());
    auto dg = grad_g.to(torch::kFloat32).contiguous();
    auto ic = i_gate.contiguous();
    auto fc = f_gate.contiguous();
    auto vc = v_val.contiguous();
    auto rc = r_gate.contiguous();
    auto sc = state.contiguous();
    auto lc = lambda_raw.contiguous();
    auto bc = beta_raw.contiguous();
    auto d_i = torch::empty_like(i_gate);
    auto d_f = torch::empty_like(f_gate);
    auto d_v = torch::empty_like(v_val);
    auto d_r = torch::empty_like(r_gate);
    auto d_state = torch::empty_like(state);
    launch_recurrent_mixer_backward(
        ic.data_ptr<float>(),
        fc.data_ptr<float>(),
        vc.data_ptr<float>(),
        rc.data_ptr<float>(),
        sc.data_ptr<at::BFloat16>(),
        lc.data_ptr<at::BFloat16>(),
        bc.data_ptr<at::BFloat16>(),
        dg.data_ptr<float>(),
        d_i.data_ptr<float>(),
        d_f.data_ptr<float>(),
        d_v.data_ptr<float>(),
        d_r.data_ptr<float>(),
        d_state.data_ptr<at::BFloat16>(),
        static_cast<int>(i_gate.size(0)),
        static_cast<int>(state.size(0)),
        static_cast<int>(i_gate.size(1))
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(d_i, d_f, d_v, d_r, d_state);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> recurrent_mixer_backward_trace(
    torch::Tensor i_gate,
    torch::Tensor f_gate,
    torch::Tensor v_val,
    torch::Tensor r_gate,
    torch::Tensor state,
    torch::Tensor lambda_raw,
    torch::Tensor beta_raw,
    torch::Tensor m_trace,
    torch::Tensor grad_g
) {
    TORCH_CHECK(i_gate.is_cuda() && f_gate.is_cuda() && v_val.is_cuda() && r_gate.is_cuda(), "gates must be CUDA");
    TORCH_CHECK(state.is_cuda() && lambda_raw.is_cuda() && beta_raw.is_cuda() && m_trace.is_cuda() && grad_g.is_cuda(), "state/recurrent/trace/grad tensors must be CUDA");
    TORCH_CHECK(i_gate.scalar_type() == torch::kFloat32 && f_gate.scalar_type() == torch::kFloat32, "i_gate and f_gate must be fp32");
    TORCH_CHECK(v_val.scalar_type() == torch::kFloat32 && r_gate.scalar_type() == torch::kFloat32, "v_val and r_gate must be fp32");
    TORCH_CHECK(state.scalar_type() == torch::kBFloat16 && lambda_raw.scalar_type() == torch::kBFloat16 && beta_raw.scalar_type() == torch::kBFloat16 && m_trace.scalar_type() == torch::kBFloat16, "state/lambda/beta/trace must be bf16");
    TORCH_CHECK(i_gate.sizes() == f_gate.sizes() && i_gate.sizes() == v_val.sizes() && i_gate.sizes() == r_gate.sizes(), "gate shape mismatch");
    TORCH_CHECK(grad_g.sizes() == i_gate.sizes(), "grad_g shape mismatch");
    TORCH_CHECK(m_trace.dim() == 3 && m_trace.size(0) == i_gate.size(0) && m_trace.size(1) == 8 && m_trace.size(2) == D_MODEL, "m_trace must be [T,8,4096]");

    const c10::cuda::CUDAGuard device_guard(i_gate.device());
    auto dg = grad_g.to(torch::kFloat32).contiguous();
    auto ic = i_gate.contiguous();
    auto fc = f_gate.contiguous();
    auto vc = v_val.contiguous();
    auto rc = r_gate.contiguous();
    auto sc = state.contiguous();
    auto lc = lambda_raw.contiguous();
    auto bc = beta_raw.contiguous();
    auto d_i = torch::empty_like(i_gate);
    auto d_f = torch::empty_like(f_gate);
    auto d_v = torch::empty_like(v_val);
    auto d_r = torch::empty_like(r_gate);
    auto d_state = torch::empty_like(state);
    launch_recurrent_mixer_backward_trace(
        ic.data_ptr<float>(),
        fc.data_ptr<float>(),
        vc.data_ptr<float>(),
        rc.data_ptr<float>(),
        sc.data_ptr<at::BFloat16>(),
        lc.data_ptr<at::BFloat16>(),
        bc.data_ptr<at::BFloat16>(),
        m_trace.data_ptr<at::BFloat16>(),
        dg.data_ptr<float>(),
        d_i.data_ptr<float>(),
        d_f.data_ptr<float>(),
        d_v.data_ptr<float>(),
        d_r.data_ptr<float>(),
        d_state.data_ptr<at::BFloat16>(),
        static_cast<int>(i_gate.size(0)),
        static_cast<int>(state.size(0)),
        static_cast<int>(i_gate.size(1))
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(d_i, d_f, d_v, d_r, d_state);
}

void block_kfac_update_curvature(
    torch::Tensor inputs,
    torch::Tensor grad_outputs,
    torch::Tensor a_cov,
    torch::Tensor g_cov,
    torch::Tensor a_inv,
    torch::Tensor g_inv,
    double damping,
    double ema
) {
    TORCH_CHECK(inputs.is_cuda() && grad_outputs.is_cuda(), "curvature inputs must be CUDA");
    TORCH_CHECK(a_cov.is_cuda() && g_cov.is_cuda() && a_inv.is_cuda() && g_inv.is_cuda(), "curvature state must be CUDA");
    TORCH_CHECK(a_cov.scalar_type() == torch::kFloat32 && g_cov.scalar_type() == torch::kFloat32, "covariances must be fp32");
    TORCH_CHECK(a_inv.scalar_type() == torch::kFloat32 && g_inv.scalar_type() == torch::kFloat32, "inverse covariances must be fp32");
    TORCH_CHECK(inputs.size(0) == grad_outputs.size(0), "curvature sample count mismatch");
    TORCH_CHECK(inputs.size(1) % 64 == 0 && grad_outputs.size(1) % 64 == 0, "K-FAC dimensions must be multiples of 64");
    TORCH_CHECK(a_cov.size(0) == inputs.size(1) / 64 && g_cov.size(0) == grad_outputs.size(1) / 64, "curvature block count mismatch");

    const c10::cuda::CUDAGuard device_guard(inputs.device());
    auto a = inputs.reshape({inputs.size(0), inputs.size(1) / 64, 64}).to(torch::kFloat32);
    auto g = grad_outputs.reshape({grad_outputs.size(0), grad_outputs.size(1) / 64, 64}).to(torch::kFloat32);
    auto a_hat = torch::einsum("nbq,nbp->bqp", {a, a}) / static_cast<double>(inputs.size(0));
    auto g_hat = torch::einsum("ncq,ncp->cqp", {g, g}) / static_cast<double>(grad_outputs.size(0));
    a_cov.mul_(1.0 - ema).add_(a_hat, ema);
    g_cov.mul_(1.0 - ema).add_(g_hat, ema);
    auto eye = torch::eye(64, torch::TensorOptions().dtype(torch::kFloat32).device(inputs.device()));
    a_inv.copy_(torch::linalg_inv(a_cov + damping * eye));
    g_inv.copy_(torch::linalg_inv(g_cov + damping * eye));
}

double block_kfac_step_param(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor a_inv,
    torch::Tensor g_inv,
    double eta,
    double weight_decay,
    double trust_region_delta,
    double eps_opt
) {
    TORCH_CHECK(param.is_cuda() && grad.is_cuda() && a_inv.is_cuda() && g_inv.is_cuda(), "K-FAC step tensors must be CUDA");
    TORCH_CHECK(param.scalar_type() == torch::kFloat32 && grad.scalar_type() == torch::kFloat32, "param and grad must be fp32");
    TORCH_CHECK(param.sizes() == grad.sizes(), "param/grad shape mismatch");
    TORCH_CHECK(param.size(0) % 64 == 0 && param.size(1) % 64 == 0, "param dimensions must be multiples of 64");
    TORCH_CHECK(g_inv.size(0) == param.size(0) / 64 && a_inv.size(0) == param.size(1) / 64, "inverse block count mismatch");

    torch::NoGradGuard no_grad;
    const c10::cuda::CUDAGuard device_guard(param.device());
    int64_t o_blocks = param.size(0) / 64;
    int64_t i_blocks = param.size(1) / 64;
    auto grad_bar = grad + weight_decay * param;
    auto grad_blocks = grad_bar.reshape({o_blocks, 64, i_blocks, 64}).permute({0, 2, 1, 3}).contiguous();
    auto d_blocks = torch::empty_like(grad_blocks);
    for (int64_t c = 0; c < o_blocks; ++c) {
        auto left = g_inv.select(0, c);
        for (int64_t b = 0; b < i_blocks; ++b) {
            auto tmp = torch::matmul(left, grad_blocks.select(0, c).select(0, b));
            d_blocks.select(0, c).select(0, b).copy_(torch::matmul(tmp, a_inv.select(0, b)));
        }
    }
    auto nat = d_blocks.permute({0, 2, 1, 3}).contiguous().reshape_as(param);
    auto nu_t = (grad_bar * nat).sum().clamp_min(0.0);
    double nu = nu_t.item<double>();
    double chi = std::min(1.0, std::sqrt(trust_region_delta / (nu + eps_opt)));
    param.add_(nat, -eta * chi);
    return nu;
}

torch::Tensor workspace_bf16_view(torch::Tensor workspace, int64_t offset_bytes, std::vector<int64_t> sizes) {
    TORCH_CHECK(offset_bytes >= 0, "workspace offset must be non-negative");
    int64_t numel = 1;
    for (int64_t s : sizes) {
        TORCH_CHECK(s >= 0, "workspace view size must be non-negative");
        numel *= s;
    }
    TORCH_CHECK(offset_bytes + numel * 2ll <= workspace.numel(), "workspace bf16 view exceeds allocation");
    auto* ptr = workspace.data_ptr<uint8_t>() + offset_bytes;
    return torch::from_blob(ptr, sizes, workspace.options().dtype(torch::kBFloat16));
}

torch::Tensor workspace_float_view(torch::Tensor workspace, int64_t offset_bytes, std::vector<int64_t> sizes) {
    TORCH_CHECK(offset_bytes >= 0, "workspace offset must be non-negative");
    TORCH_CHECK((offset_bytes & 3ll) == 0, "workspace fp32 view must be 4-byte aligned");
    int64_t numel = 1;
    for (int64_t s : sizes) {
        TORCH_CHECK(s >= 0, "workspace view size must be non-negative");
        numel *= s;
    }
    TORCH_CHECK(offset_bytes + numel * 4ll <= workspace.numel(), "workspace fp32 view exceeds allocation");
    auto* ptr = workspace.data_ptr<uint8_t>() + offset_bytes;
    return torch::from_blob(ptr, sizes, workspace.options().dtype(torch::kFloat32));
}

torch::Tensor tlinear_backward_input_native(
    torch::Tensor grad_y,
    torch::Tensor Wpack,
    torch::Tensor alpha,
    int d_in
) {
    TORCH_CHECK(grad_y.dim() == 2, "TLinear backward grad_y must be [T,d_out]");
    auto gy = grad_y.to(torch::kBFloat16).contiguous();
    auto dX = torch::empty({gy.size(0), d_in}, gy.options().dtype(torch::kFloat32));
    launch_tlinear_backward_input(
        gy.data_ptr<at::BFloat16>(),
        Wpack.data_ptr<uint8_t>(),
        alpha.data_ptr<at::BFloat16>(),
        dX.data_ptr<float>(),
        static_cast<int>(gy.size(0)),
        d_in,
        static_cast<int>(Wpack.size(0))
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dX;
}

torch::Tensor deterministic_tlinear_backward_input_native(
    torch::Tensor grad_y,
    torch::Tensor alpha,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
) {
    TORCH_CHECK(grad_y.dim() == 2, "deterministic TLinear backward grad_y must be [T,d_out]");
    TORCH_CHECK(alpha.size(0) == d_out, "deterministic TLinear backward alpha mismatch");
    auto gy = grad_y.to(torch::kBFloat16).contiguous();
    auto a = alpha.contiguous();
    auto dX = torch::empty({gy.size(0), d_in}, gy.options().dtype(torch::kFloat32));
    launch_deterministic_tlinear_backward_input(
        gy.data_ptr<at::BFloat16>(),
        a.data_ptr<at::BFloat16>(),
        dX.data_ptr<float>(),
        static_cast<int>(gy.size(0)),
        d_in,
        d_out,
        layer,
        matrix_id
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dX;
}

torch::Tensor silex_tlinear_forward_select(
    torch::Tensor X,
    torch::Tensor Wpack,
    torch::Tensor alpha,
    int d_in,
    int d_out,
    int layer,
    int matrix_id,
    bool deterministic_backbone
) {
    if (deterministic_backbone) {
        return deterministic_tlinear_forward_native(
            X,
            alpha.contiguous(),
            d_in,
            d_out,
            layer,
            matrix_id
        );
    }
    return tlinear_forward_native(
        X,
        Wpack.contiguous(),
        alpha.contiguous(),
        d_in,
        d_out
    );
}

std::vector<torch::Tensor> silex_tlinear_forward_multi_select(
    torch::Tensor X,
    std::vector<torch::Tensor> Wpacks,
    std::vector<torch::Tensor> alphas,
    int d_in,
    int d_out,
    int layer,
    int matrix_id0,
    bool deterministic_backbone
) {
    TORCH_CHECK(!alphas.empty() && alphas.size() <= 4, "multi TLinear supports 1..4 outputs");
    TORCH_CHECK(Wpacks.size() == alphas.size(), "multi TLinear Wpack/alpha count mismatch");
    if (deterministic_backbone) {
        return deterministic_tlinear_forward_multi_native(
            X,
            alphas,
            d_in,
            d_out,
            layer,
            matrix_id0
        );
    }
    std::vector<torch::Tensor> outputs;
    outputs.reserve(alphas.size());
    for (size_t i = 0; i < alphas.size(); ++i) {
        outputs.push_back(tlinear_forward_native(
            X,
            Wpacks[i].contiguous(),
            alphas[i].contiguous(),
            d_in,
            d_out
        ));
    }
    return outputs;
}

torch::Tensor silex_tlinear_backward_input_select(
    torch::Tensor grad_y,
    torch::Tensor Wpack,
    torch::Tensor alpha,
    int d_in,
    int d_out,
    int layer,
    int matrix_id,
    bool deterministic_backbone
) {
    if (deterministic_backbone) {
        return deterministic_tlinear_backward_input_native(
            grad_y,
            alpha.contiguous(),
            d_in,
            d_out,
            layer,
            matrix_id
        );
    }
    return tlinear_backward_input_native(
        grad_y,
        Wpack.contiguous(),
        alpha.contiguous(),
        d_in
    );
}

struct CurriculumGradResult {
    std::vector<torch::Tensor> dlogits;
    std::array<double, 5> nll_by_k;
    double mono;
    double latent_gain;
};

CurriculumGradResult curriculum_logits_grad(
    const std::vector<torch::Tensor>& logits_by_depth,
    torch::Tensor labels,
    torch::Tensor loss_mask,
    int64_t stage,
    torch::Tensor teacher_logits_final
) {
    TORCH_CHECK(logits_by_depth.size() == 5, "training requires logits for depths k=0..4");
    TORCH_CHECK(labels.is_cuda() && loss_mask.is_cuda(), "labels and loss_mask must be CUDA");
    TORCH_CHECK(labels.scalar_type() == torch::kLong, "labels must be int64/long");
    TORCH_CHECK(loss_mask.scalar_type() == torch::kFloat32, "loss_mask must be fp32");
    TORCH_CHECK(labels.dim() == 1 && labels.size(0) == 511, "labels must have shape [511]");
    TORCH_CHECK(loss_mask.dim() == 1 && loss_mask.size(0) == 511, "loss_mask must have shape [511]");
    TORCH_CHECK(stage == 1 || stage == 2 || stage == 3, "stage must be 1, 2, or 3");

    constexpr double omega[5] = {
        2.0 / 30.0,
        4.0 / 30.0,
        6.0 / 30.0,
        8.0 / 30.0,
        10.0 / 30.0,
    };

    auto mask = loss_mask.contiguous();
    double denom = std::max(1.0, mask.sum().item<double>());
    auto mask_col = (mask / denom).reshape({511, 1});
    auto label_col = labels.contiguous().reshape({511, 1});

    std::vector<torch::Tensor> ce;
    std::vector<torch::Tensor> dce;
    std::vector<torch::Tensor> probs;
    ce.reserve(5);
    dce.reserve(5);
    probs.reserve(5);

    CurriculumGradResult result{};
    result.dlogits.reserve(5);

    for (int k = 0; k < 5; ++k) {
        TORCH_CHECK(logits_by_depth[k].is_cuda(), "logits must be CUDA");
        TORCH_CHECK(logits_by_depth[k].dim() == 2 && logits_by_depth[k].size(0) == 511 && logits_by_depth[k].size(1) == VOCAB_SIZE, "logits depth tensor must be [511,258]");
        auto logits = logits_by_depth[k].to(torch::kFloat32).contiguous();
        auto logp = torch::log_softmax(logits, -1);
        auto p = torch::softmax(logits, -1);
        probs.push_back(p);
        ce.push_back(-logp.gather(1, label_col).reshape({511}));
        auto one_hot = torch::zeros_like(p);
        one_hot.scatter_(1, label_col, 1.0);
        auto base = p - one_hot;
        dce.push_back(base);
        result.dlogits.push_back(base * (mask_col * omega[k]));
        result.nll_by_k[static_cast<size_t>(k)] = ((ce.back() * mask).sum() / denom).item<double>();
    }

    auto mono_acc = torch::zeros({}, mask.options());
    constexpr double mono_loss_weight = 0.10;
    for (int k = 0; k < 4; ++k) {
        auto diff = ce[static_cast<size_t>(k + 1)] - ce[static_cast<size_t>(k)];
        mono_acc = mono_acc + (torch::relu(diff) * mask).sum();
        auto active = (diff > 0).to(torch::kFloat32) * mask * (mono_loss_weight / (4.0 * denom));
        auto active_col = active.reshape({511, 1});
        result.dlogits[static_cast<size_t>(k + 1)].add_(dce[static_cast<size_t>(k + 1)] * active_col);
        result.dlogits[static_cast<size_t>(k)].add_(dce[static_cast<size_t>(k)] * active_col, -1.0);
    }
    result.mono = (mono_acc / (4.0 * denom)).item<double>();
    result.latent_gain = result.nll_by_k[0] - result.nll_by_k[4];

    if (stage == 3 && teacher_logits_final.defined() && teacher_logits_final.numel() > 0) {
        TORCH_CHECK(teacher_logits_final.is_cuda(), "teacher logits must be CUDA when provided");
        TORCH_CHECK(teacher_logits_final.dim() == 2 && teacher_logits_final.size(0) == 511 && teacher_logits_final.size(1) == VOCAB_SIZE, "teacher logits must be [511,258]");
        auto teacher = teacher_logits_final.to(torch::kFloat32).contiguous();
        auto q = torch::softmax(teacher, -1);
        auto kd_scale = mask_col * (0.25 / 5.0);
        for (int k = 0; k < 5; ++k) {
            result.dlogits[static_cast<size_t>(k)].add_((probs[static_cast<size_t>(k)] - q) * kd_scale);
        }
    }

    for (int k = 0; k < 5; ++k) {
        result.dlogits[static_cast<size_t>(k)] = result.dlogits[static_cast<size_t>(k)].contiguous();
    }
    return result;
}

bool layer_is_active(int layer_zero_based, const std::unordered_set<int64_t>& active_layers) {
    return active_layers.empty() || active_layers.count(static_cast<int64_t>(layer_zero_based + 1)) != 0;
}

void add_mdl_grad_inplace(torch::Tensor grad, torch::Tensor param) {
    constexpr double inv_np = 1.0 / 67108864.0;
    constexpr double mdl_weight = 1.0e-6;
    constexpr double tau = 0.0009765625;
    constexpr double inv_ln2 = 1.4426950408889634;
    auto mdl = torch::sign(param) / (torch::abs(param) + tau);
    grad.add_(mdl, mdl_weight * inv_np * inv_ln2);
}

void update_curvature_and_collect(
    int kfac_index,
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor inputs,
    torch::Tensor grad_outputs,
    std::vector<torch::Tensor>& a_covs,
    std::vector<torch::Tensor>& g_covs,
    std::vector<torch::Tensor>& a_invs,
    std::vector<torch::Tensor>& g_invs,
    std::vector<torch::Tensor>& step_params,
    std::vector<torch::Tensor>& step_grads,
    std::vector<torch::Tensor>& step_a_invs,
    std::vector<torch::Tensor>& step_g_invs,
    double damping,
    double ema
) {
    TORCH_CHECK(kfac_index >= 0 && static_cast<size_t>(kfac_index) < a_covs.size(), "K-FAC index out of range");
    block_kfac_update_curvature(
        inputs.contiguous(),
        grad_outputs.contiguous(),
        a_covs[static_cast<size_t>(kfac_index)].contiguous(),
        g_covs[static_cast<size_t>(kfac_index)].contiguous(),
        a_invs[static_cast<size_t>(kfac_index)].contiguous(),
        g_invs[static_cast<size_t>(kfac_index)].contiguous(),
        damping,
        ema
    );
    add_mdl_grad_inplace(grad, param);
    step_params.push_back(param);
    step_grads.push_back(grad.contiguous());
    step_a_invs.push_back(a_invs[static_cast<size_t>(kfac_index)]);
    step_g_invs.push_back(g_invs[static_cast<size_t>(kfac_index)]);
}

std::pair<double, double> apply_global_kfac_step(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& grads,
    const std::vector<torch::Tensor>& a_invs,
    const std::vector<torch::Tensor>& g_invs,
    double eta,
    double weight_decay,
    double trust_region_delta,
    double eps_opt
) {
    TORCH_CHECK(params.size() == grads.size() && params.size() == a_invs.size() && params.size() == g_invs.size(), "K-FAC step vector size mismatch");
    if (params.empty()) {
        return {0.0, 1.0};
    }
    torch::NoGradGuard no_grad;
    const c10::cuda::CUDAGuard device_guard(params[0].device());

    auto nu_t = torch::zeros({}, params[0].options().dtype(torch::kFloat32));
    for (size_t idx = 0; idx < params.size(); ++idx) {
        auto param = params[idx];
        auto grad_bar = grads[idx] + weight_decay * param;
        int64_t o_blocks = param.size(0) / 64;
        int64_t i_blocks = param.size(1) / 64;
        auto grad_blocks = grad_bar.reshape({o_blocks, 64, i_blocks, 64}).permute({0, 2, 1, 3}).contiguous();
        auto d_blocks = torch::empty_like(grad_blocks);
        for (int64_t c = 0; c < o_blocks; ++c) {
            auto left = g_invs[idx].select(0, c);
            for (int64_t b = 0; b < i_blocks; ++b) {
                auto tmp = torch::matmul(left, grad_blocks.select(0, c).select(0, b));
                d_blocks.select(0, c).select(0, b).copy_(torch::matmul(tmp, a_invs[idx].select(0, b)));
            }
        }
        auto nat = d_blocks.permute({0, 2, 1, 3}).contiguous().reshape_as(param);
        nu_t = nu_t + (grad_bar * nat).sum();
    }

    double nu = std::max(0.0, nu_t.item<double>());
    double chi = std::min(1.0, std::sqrt(trust_region_delta / (nu + eps_opt)));
    for (size_t idx = 0; idx < params.size(); ++idx) {
        auto param = params[idx];
        auto grad_bar = grads[idx] + weight_decay * param;
        int64_t o_blocks = param.size(0) / 64;
        int64_t i_blocks = param.size(1) / 64;
        auto grad_blocks = grad_bar.reshape({o_blocks, 64, i_blocks, 64}).permute({0, 2, 1, 3}).contiguous();
        auto d_blocks = torch::empty_like(grad_blocks);
        for (int64_t c = 0; c < o_blocks; ++c) {
            auto left = g_invs[idx].select(0, c);
            for (int64_t b = 0; b < i_blocks; ++b) {
                auto tmp = torch::matmul(left, grad_blocks.select(0, c).select(0, b));
                d_blocks.select(0, c).select(0, b).copy_(torch::matmul(tmp, a_invs[idx].select(0, b)));
            }
        }
        auto nat = d_blocks.permute({0, 2, 1, 3}).contiguous().reshape_as(param);
        param.add_(nat, -eta * chi);
    }
    return {nu, chi};
}

torch::Tensor apply_output_adapter_logits(
    torch::Tensor normalized,
    torch::Tensor base_logits,
    torch::Tensor output_adapter_down,
    torch::Tensor output_adapter_up,
    bool use_output_adapter
) {
    if (!use_output_adapter) {
        return base_logits;
    }
    TORCH_CHECK(output_adapter_down.is_cuda() && output_adapter_up.is_cuda(), "output adapter tensors must be CUDA");
    TORCH_CHECK(output_adapter_down.scalar_type() == torch::kFloat32, "output_adapter_down must be fp32");
    TORCH_CHECK(output_adapter_up.scalar_type() == torch::kFloat32, "output_adapter_up must be fp32");
    TORCH_CHECK(output_adapter_down.dim() == 2 && output_adapter_down.size(1) == D_MODEL, "output_adapter_down must be [rank,4096]");
    TORCH_CHECK(output_adapter_up.dim() == 2 && output_adapter_up.size(0) == VOCAB_SIZE, "output_adapter_up must be [258,rank]");
    TORCH_CHECK(output_adapter_up.size(1) == output_adapter_down.size(0), "output adapter rank mismatch");
    auto hidden = torch::matmul(normalized.to(torch::kFloat32), output_adapter_down.contiguous().transpose(0, 1));
    auto delta = torch::matmul(hidden, output_adapter_up.contiguous().transpose(0, 1));
    return base_logits + delta;
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> silex_forward_cuda_impl(
    torch::Tensor token_ids,
    torch::Tensor state,
    torch::Tensor e_wpack,
    torch::Tensor e_alpha,
    std::vector<torch::Tensor> layer_wpacks,
    std::vector<torch::Tensor> layer_alphas,
    std::vector<torch::Tensor> gamma_m,
    std::vector<torch::Tensor> gamma_f,
    std::vector<torch::Tensor> lambda_raw,
    std::vector<torch::Tensor> beta_raw,
    std::vector<torch::Tensor> A_m,
    std::vector<torch::Tensor> B_m,
    std::vector<torch::Tensor> A_f,
    std::vector<torch::Tensor> B_f,
    std::vector<torch::Tensor> z_wpacks,
    std::vector<torch::Tensor> z_alphas,
    torch::Tensor gamma_z,
    torch::Tensor gamma_out,
    torch::Tensor output_adapter_down,
    torch::Tensor output_adapter_up,
    bool use_output_adapter,
    int64_t K,
    bool return_all_depths,
    bool deterministic_backbone
) {
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA");
    TORCH_CHECK(token_ids.scalar_type() == torch::kUInt16, "token_ids must be uint16");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must have shape [T]");
    TORCH_CHECK(token_ids.size(0) >= 1 && token_ids.size(0) <= S_MAX, "T must be in [1,8192]");
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kBFloat16, "state must be CUDA bf16");
    TORCH_CHECK(state.dim() == 3 && state.size(0) == 64 && state.size(1) == 8 && state.size(2) == D_MODEL, "state must have shape [64,8,4096]");
    TORCH_CHECK(K >= 0 && K <= 32, "K must be in [0,32]");
    TORCH_CHECK(layer_wpacks.size() == 64 * 8, "layer_wpacks must contain 512 tensors");
    TORCH_CHECK(layer_alphas.size() == 64 * 8, "layer_alphas must contain 512 tensors");
    TORCH_CHECK(gamma_m.size() == 64 && gamma_f.size() == 64, "gamma vectors must contain 64 tensors");
    TORCH_CHECK(lambda_raw.size() == 64 && beta_raw.size() == 64, "recurrent vectors must contain 64 tensors");
    TORCH_CHECK(A_m.size() == 64 && B_m.size() == 64 && A_f.size() == 64 && B_f.size() == 64, "adapter vectors must contain 64 tensors");
    TORCH_CHECK(z_wpacks.size() == 3 && z_alphas.size() == 3, "latent vectors must contain 3 tensors");

    const c10::cuda::CUDAGuard device_guard(token_ids.device());
    constexpr float EPS_NORM = 0.000244140625f;
    constexpr float RHO = 0.08838834764831845f;
    constexpr float RHO_Z = 0.125f;

    auto x = embedding_native(token_ids.contiguous(), e_wpack.contiguous(), e_alpha.contiguous());
    auto new_state = torch::empty_like(state);

    for (int l = 0; l < 64; ++l) {
        int base = l * 8;
        auto u = rms_norm_native(x.contiguous(), gamma_m[l].contiguous(), EPS_NORM);
        int layer_id = l + 1;
        auto gate_pre = silex_tlinear_forward_multi_select(
            u,
            {layer_wpacks[base + 0].contiguous(), layer_wpacks[base + 1].contiguous(), layer_wpacks[base + 2].contiguous(), layer_wpacks[base + 3].contiguous()},
            {layer_alphas[base + 0].contiguous(), layer_alphas[base + 1].contiguous(), layer_alphas[base + 2].contiguous(), layer_alphas[base + 3].contiguous()},
            D_MODEL,
            D_MODEL,
            layer_id,
            0,
            deterministic_backbone
        );
        auto i_gate = activation_native(gate_pre[0], 0);
        auto f_gate = activation_native(gate_pre[1], 0);
        auto v_val = activation_native(gate_pre[2], 1);
        auto r_gate = activation_native(gate_pre[3], 0);
        auto rec = recurrent_native(
            i_gate.contiguous(),
            f_gate.contiguous(),
            v_val.contiguous(),
            r_gate.contiguous(),
            state.select(0, l).contiguous(),
            lambda_raw[l].contiguous(),
            beta_raw[l].contiguous()
        );
        auto g = std::get<0>(rec);
        new_state.select(0, l).copy_(std::get<1>(rec));
        auto p_m = adapter_native(u, A_m[l].contiguous(), B_m[l].contiguous());
        auto o = silex_tlinear_forward_select(g, layer_wpacks[base + 4].contiguous(), layer_alphas[base + 4].contiguous(), D_MODEL, D_MODEL, layer_id, 4, deterministic_backbone);
        auto x_tilde = residual_native(x.contiguous(), o.contiguous(), p_m.contiguous(), RHO);

        auto u_f = rms_norm_native(x_tilde.contiguous(), gamma_f[l].contiguous(), EPS_NORM);
        auto ab = silex_tlinear_forward_multi_select(
            u_f,
            {layer_wpacks[base + 5].contiguous(), layer_wpacks[base + 6].contiguous()},
            {layer_alphas[base + 5].contiguous(), layer_alphas[base + 6].contiguous()},
            D_MODEL,
            D_FF,
            layer_id,
            5,
            deterministic_backbone
        );
        auto a = ab[0];
        auto b = ab[1];
        auto h = gated_silu_native(a.contiguous(), b.contiguous());
        auto p_f = adapter_native(u_f, A_f[l].contiguous(), B_f[l].contiguous());
        auto c = silex_tlinear_forward_select(h, layer_wpacks[base + 7].contiguous(), layer_alphas[base + 7].contiguous(), D_FF, D_MODEL, layer_id, 7, deterministic_backbone);
        x = residual_native(x_tilde.contiguous(), c.contiguous(), p_f.contiguous(), RHO);
    }

    std::vector<torch::Tensor> logits_by_depth;
    auto cur = x;
    if (return_all_depths) {
        auto out0 = rms_norm_native(cur.contiguous(), gamma_out.contiguous(), EPS_NORM);
        auto base0 = tlinear_forward_native(out0, e_wpack.contiguous(), e_alpha.contiguous(), D_MODEL, VOCAB_SIZE).to(torch::kFloat32);
        logits_by_depth.push_back(apply_output_adapter_logits(out0, base0, output_adapter_down, output_adapter_up, use_output_adapter));
    }

    for (int64_t k = 0; k < K; ++k) {
        auto n = rms_norm_native(cur.contiguous(), gamma_z.contiguous(), EPS_NORM);
        auto za = silex_tlinear_forward_select(n, z_wpacks[0].contiguous(), z_alphas[0].contiguous(), D_MODEL, D_Z, 0, 8, deterministic_backbone);
        auto zb = silex_tlinear_forward_select(n, z_wpacks[1].contiguous(), z_alphas[1].contiguous(), D_MODEL, D_Z, 0, 9, deterministic_backbone);
        auto q = gated_silu_native(za.contiguous(), zb.contiguous());
        auto z3 = silex_tlinear_forward_select(q, z_wpacks[2].contiguous(), z_alphas[2].contiguous(), D_Z, D_MODEL, 0, 10, deterministic_backbone);
        auto zero = torch::empty_like(cur);
        zero.zero_();
        cur = residual_native(cur.contiguous(), z3.contiguous(), zero.contiguous(), RHO_Z);
        if (return_all_depths) {
            auto outk = rms_norm_native(cur.contiguous(), gamma_out.contiguous(), EPS_NORM);
            auto basek = tlinear_forward_native(outk, e_wpack.contiguous(), e_alpha.contiguous(), D_MODEL, VOCAB_SIZE).to(torch::kFloat32);
            logits_by_depth.push_back(apply_output_adapter_logits(outk, basek, output_adapter_down, output_adapter_up, use_output_adapter));
        }
    }

    torch::Tensor logits;
    if (return_all_depths) {
        logits = logits_by_depth.back();
    } else {
        auto out = rms_norm_native(cur.contiguous(), gamma_out.contiguous(), EPS_NORM);
        auto base = tlinear_forward_native(out, e_wpack.contiguous(), e_alpha.contiguous(), D_MODEL, VOCAB_SIZE).to(torch::kFloat32);
        logits = apply_output_adapter_logits(out, base, output_adapter_down, output_adapter_up, use_output_adapter);
    }
    return std::make_tuple(logits, new_state, logits_by_depth);
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> silex_forward_cuda(
    torch::Tensor token_ids,
    torch::Tensor state,
    torch::Tensor e_wpack,
    torch::Tensor e_alpha,
    std::vector<torch::Tensor> layer_wpacks,
    std::vector<torch::Tensor> layer_alphas,
    std::vector<torch::Tensor> gamma_m,
    std::vector<torch::Tensor> gamma_f,
    std::vector<torch::Tensor> lambda_raw,
    std::vector<torch::Tensor> beta_raw,
    std::vector<torch::Tensor> A_m,
    std::vector<torch::Tensor> B_m,
    std::vector<torch::Tensor> A_f,
    std::vector<torch::Tensor> B_f,
    std::vector<torch::Tensor> z_wpacks,
    std::vector<torch::Tensor> z_alphas,
    torch::Tensor gamma_z,
    torch::Tensor gamma_out,
    int64_t K,
    bool return_all_depths,
    bool deterministic_backbone
) {
    return silex_forward_cuda_impl(
        token_ids,
        state,
        e_wpack,
        e_alpha,
        layer_wpacks,
        layer_alphas,
        gamma_m,
        gamma_f,
        lambda_raw,
        beta_raw,
        A_m,
        B_m,
        A_f,
        B_f,
        z_wpacks,
        z_alphas,
        gamma_z,
        gamma_out,
        torch::Tensor(),
        torch::Tensor(),
        false,
        K,
        return_all_depths,
        deterministic_backbone
    );
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> silex_forward_cuda_output_adapter(
    torch::Tensor token_ids,
    torch::Tensor state,
    torch::Tensor e_wpack,
    torch::Tensor e_alpha,
    std::vector<torch::Tensor> layer_wpacks,
    std::vector<torch::Tensor> layer_alphas,
    std::vector<torch::Tensor> gamma_m,
    std::vector<torch::Tensor> gamma_f,
    std::vector<torch::Tensor> lambda_raw,
    std::vector<torch::Tensor> beta_raw,
    std::vector<torch::Tensor> A_m,
    std::vector<torch::Tensor> B_m,
    std::vector<torch::Tensor> A_f,
    std::vector<torch::Tensor> B_f,
    std::vector<torch::Tensor> z_wpacks,
    std::vector<torch::Tensor> z_alphas,
    torch::Tensor gamma_z,
    torch::Tensor gamma_out,
    torch::Tensor output_adapter_down,
    torch::Tensor output_adapter_up,
    int64_t K,
    bool return_all_depths,
    bool deterministic_backbone
) {
    return silex_forward_cuda_impl(
        token_ids,
        state,
        e_wpack,
        e_alpha,
        layer_wpacks,
        layer_alphas,
        gamma_m,
        gamma_f,
        lambda_raw,
        beta_raw,
        A_m,
        B_m,
        A_f,
        B_f,
        z_wpacks,
        z_alphas,
        gamma_z,
        gamma_out,
        output_adapter_down,
        output_adapter_up,
        true,
        K,
        return_all_depths,
        deterministic_backbone
    );
}

int64_t silex_train_workspace_bytes(int64_t sequence_len) {
    return make_train_workspace_layout(sequence_len).total_bytes;
}

pybind11::dict silex_train_workspace_layout(int64_t sequence_len) {
    auto layout = make_train_workspace_layout(sequence_len);
    pybind11::dict out;
    out["x_offset"] = layout.x_offset;
    out["rec_trace_offset"] = layout.rec_trace_offset;
    out["z_trace_offset"] = layout.z_trace_offset;
    out["ff_offset"] = layout.ff_offset;
    out["mix_offset"] = layout.mix_offset;
    out["logits_offset"] = layout.logits_offset;
    out["total_bytes"] = layout.total_bytes;
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> silex_train_chunk_cuda(
    torch::Tensor token_ids_512,
    torch::Tensor state,
    torch::Tensor workspace,
    torch::Tensor e_wpack,
    torch::Tensor e_alpha,
    std::vector<torch::Tensor> layer_wpacks,
    std::vector<torch::Tensor> layer_alphas,
    std::vector<torch::Tensor> gamma_m,
    std::vector<torch::Tensor> gamma_f,
    std::vector<torch::Tensor> lambda_raw,
    std::vector<torch::Tensor> beta_raw,
    std::vector<torch::Tensor> A_m,
    std::vector<torch::Tensor> B_m,
    std::vector<torch::Tensor> A_f,
    std::vector<torch::Tensor> B_f,
    std::vector<torch::Tensor> z_wpacks,
    std::vector<torch::Tensor> z_alphas,
    torch::Tensor gamma_z,
    torch::Tensor gamma_out,
    bool deterministic_backbone
) {
    TORCH_CHECK(token_ids_512.is_cuda(), "token_ids_512 must be CUDA");
    TORCH_CHECK(token_ids_512.scalar_type() == torch::kUInt16, "token_ids_512 must be uint16");
    TORCH_CHECK(token_ids_512.numel() == 512, "local training chunk must contain exactly 512 tokens");
    TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA");
    TORCH_CHECK(workspace.scalar_type() == torch::kUInt8, "workspace must be uint8");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(workspace.numel() >= silex_train_workspace_bytes(512), "workspace is smaller than the static TDD training buffer");

    // Training predicts y[t+1] from y[<=t], so the model consumes the first 511
    // tokens and returns depth logits for those 511 effective positions.
    auto input_tokens = token_ids_512.narrow(0, 0, 511).contiguous();
    return silex_forward_cuda(
        input_tokens,
        state,
        e_wpack,
        e_alpha,
        layer_wpacks,
        layer_alphas,
        gamma_m,
        gamma_f,
        lambda_raw,
        beta_raw,
        A_m,
        B_m,
        A_f,
        B_f,
        z_wpacks,
        z_alphas,
        gamma_z,
        gamma_out,
        4,
        true,
        deterministic_backbone
    );
}

pybind11::dict silex_train_chunk_cuda_update(
    torch::Tensor token_ids_512,
    torch::Tensor state,
    torch::Tensor workspace,
    torch::Tensor labels,
    torch::Tensor loss_mask,
    torch::Tensor teacher_logits_final,
    torch::Tensor e_wpack,
    torch::Tensor e_alpha,
    std::vector<torch::Tensor> layer_wpacks,
    std::vector<torch::Tensor> layer_alphas,
    std::vector<torch::Tensor> gamma_m,
    std::vector<torch::Tensor> gamma_f,
    std::vector<torch::Tensor> lambda_raw,
    std::vector<torch::Tensor> beta_raw,
    std::vector<torch::Tensor> A_m,
    std::vector<torch::Tensor> B_m,
    std::vector<torch::Tensor> A_f,
    std::vector<torch::Tensor> B_f,
    std::vector<torch::Tensor> z_wpacks,
    std::vector<torch::Tensor> z_alphas,
    torch::Tensor gamma_z,
    torch::Tensor gamma_out,
    bool deterministic_backbone,
    std::vector<torch::Tensor> kfac_a_covs,
    std::vector<torch::Tensor> kfac_g_covs,
    std::vector<torch::Tensor> kfac_a_invs,
    std::vector<torch::Tensor> kfac_g_invs,
    std::vector<int64_t> active_layers,
    int64_t stage,
    double eta,
    double damping,
    double trust_region_delta,
    double ema,
    double weight_decay,
    double eps_opt
) {
    TORCH_CHECK(token_ids_512.is_cuda(), "token_ids_512 must be CUDA");
    TORCH_CHECK(token_ids_512.scalar_type() == torch::kUInt16, "token_ids_512 must be uint16");
    TORCH_CHECK(token_ids_512.numel() == 512, "local training chunk must contain exactly 512 tokens");
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kBFloat16, "state must be CUDA bf16");
    TORCH_CHECK(state.dim() == 3 && state.size(0) == 64 && state.size(1) == 8 && state.size(2) == D_MODEL, "state must be [64,8,4096]");
    TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kUInt8 && workspace.is_contiguous(), "workspace must be contiguous CUDA uint8");
    TORCH_CHECK(workspace.numel() >= silex_train_workspace_bytes(512), "workspace is smaller than the static TDD training buffer");
    TORCH_CHECK(layer_wpacks.size() == 64 * 8 && layer_alphas.size() == 64 * 8, "layer TLinear vectors must have 512 tensors");
    TORCH_CHECK(gamma_m.size() == 64 && gamma_f.size() == 64 && lambda_raw.size() == 64 && beta_raw.size() == 64, "per-layer metadata vectors must have 64 tensors");
    TORCH_CHECK(A_m.size() == 64 && B_m.size() == 64 && A_f.size() == 64 && B_f.size() == 64, "adapter vectors must have 64 tensors");
    TORCH_CHECK(z_wpacks.size() == 3 && z_alphas.size() == 3, "latent TLinear vectors must have 3 tensors");
    TORCH_CHECK(kfac_a_covs.size() == 256 && kfac_g_covs.size() == 256 && kfac_a_invs.size() == 256 && kfac_g_invs.size() == 256, "K-FAC state must contain 256 matrices");
    TrainProfiler profiler(env_enabled("SILEX_PROFILE_TRAIN"));
    profiler.mark("enter");

    torch::NoGradGuard no_grad;
    const c10::cuda::CUDAGuard device_guard(token_ids_512.device());
    constexpr float EPS_NORM = 0.000244140625f;
    constexpr float RHO = 0.08838834764831845f;
    constexpr float RHO_Z = 0.125f;
    constexpr int64_t T = 511;

    std::unordered_set<int64_t> active;
    for (int64_t x : active_layers) {
        TORCH_CHECK(x >= 1 && x <= 64, "active_layers are 1-based and must be in [1,64]");
        active.insert(x);
    }

    auto layout = make_train_workspace_layout(512);
    auto x_trace = workspace_bf16_view(workspace, layout.x_offset, {65, 512, D_MODEL});
    auto rec_trace = workspace_bf16_view(workspace, layout.rec_trace_offset, {64, 512, 8, D_MODEL});
    auto z_trace = workspace_bf16_view(workspace, layout.z_trace_offset, {5, 512, D_MODEL});
    profiler.mark("workspace_views");

    auto input_tokens = token_ids_512.narrow(0, 0, T).contiguous();
    auto x0 = embedding_native(input_tokens, e_wpack.contiguous(), e_alpha.contiguous());
    x_trace.select(0, 0).narrow(0, 0, T).copy_(x0);
    profiler.mark("embedding");

    auto new_state = torch::empty_like(state);
    auto x = x_trace.select(0, 0).narrow(0, 0, T);
    for (int l = 0; l < 64; ++l) {
        int base = l * 8;
        auto u = rms_norm_native(x.contiguous(), gamma_m[l].contiguous(), EPS_NORM);
        int layer_id = l + 1;
        auto gate_pre = silex_tlinear_forward_multi_select(
            u,
            {layer_wpacks[base + 0].contiguous(), layer_wpacks[base + 1].contiguous(), layer_wpacks[base + 2].contiguous(), layer_wpacks[base + 3].contiguous()},
            {layer_alphas[base + 0].contiguous(), layer_alphas[base + 1].contiguous(), layer_alphas[base + 2].contiguous(), layer_alphas[base + 3].contiguous()},
            D_MODEL,
            D_MODEL,
            layer_id,
            0,
            deterministic_backbone
        );
        auto i_gate = activation_native(gate_pre[0], 0);
        auto f_gate = activation_native(gate_pre[1], 0);
        auto v_val = activation_native(gate_pre[2], 1);
        auto r_gate = activation_native(gate_pre[3], 0);
        auto rec = recurrent_native_trace(
            i_gate.contiguous(),
            f_gate.contiguous(),
            v_val.contiguous(),
            r_gate.contiguous(),
            state.select(0, l).contiguous(),
            lambda_raw[l].contiguous(),
            beta_raw[l].contiguous(),
            rec_trace.select(0, l).narrow(0, 0, T)
        );
        auto g = std::get<0>(rec);
        new_state.select(0, l).copy_(std::get<1>(rec));
        auto p_m = adapter_native(u, A_m[l].contiguous(), B_m[l].contiguous());
        auto o = silex_tlinear_forward_select(g, layer_wpacks[base + 4].contiguous(), layer_alphas[base + 4].contiguous(), D_MODEL, D_MODEL, layer_id, 4, deterministic_backbone);
        auto x_tilde = residual_native(x.contiguous(), o.contiguous(), p_m.contiguous(), RHO);
        auto u_f = rms_norm_native(x_tilde.contiguous(), gamma_f[l].contiguous(), EPS_NORM);
        auto ab = silex_tlinear_forward_multi_select(
            u_f,
            {layer_wpacks[base + 5].contiguous(), layer_wpacks[base + 6].contiguous()},
            {layer_alphas[base + 5].contiguous(), layer_alphas[base + 6].contiguous()},
            D_MODEL,
            D_FF,
            layer_id,
            5,
            deterministic_backbone
        );
        auto a = ab[0];
        auto b = ab[1];
        auto h = gated_silu_native(a.contiguous(), b.contiguous());
        auto p_f = adapter_native(u_f, A_f[l].contiguous(), B_f[l].contiguous());
        auto c = silex_tlinear_forward_select(h, layer_wpacks[base + 7].contiguous(), layer_alphas[base + 7].contiguous(), D_FF, D_MODEL, layer_id, 7, deterministic_backbone);
        auto x_next = residual_native(x_tilde.contiguous(), c.contiguous(), p_f.contiguous(), RHO);
        x_trace.select(0, l + 1).narrow(0, 0, T).copy_(x_next);
        x = x_trace.select(0, l + 1).narrow(0, 0, T);
        if ((l + 1) % 8 == 0) {
            std::string phase = "forward_layer_" + std::to_string(l + 1);
            profiler.mark(phase.c_str());
        }
    }
    profiler.mark("forward_backbone_done");

    std::vector<torch::Tensor> logits_by_depth;
    logits_by_depth.reserve(5);
    z_trace.select(0, 0).narrow(0, 0, T).copy_(x);
    auto cur = z_trace.select(0, 0).narrow(0, 0, T);
    auto out0 = rms_norm_native(cur.contiguous(), gamma_out.contiguous(), EPS_NORM);
    logits_by_depth.push_back(tlinear_forward_native(out0, e_wpack.contiguous(), e_alpha.contiguous(), D_MODEL, VOCAB_SIZE).to(torch::kFloat32));
    for (int k = 1; k <= 4; ++k) {
        auto n = rms_norm_native(cur.contiguous(), gamma_z.contiguous(), EPS_NORM);
        auto zab = silex_tlinear_forward_multi_select(
            n,
            {z_wpacks[0].contiguous(), z_wpacks[1].contiguous()},
            {z_alphas[0].contiguous(), z_alphas[1].contiguous()},
            D_MODEL,
            D_Z,
            0,
            8,
            deterministic_backbone
        );
        auto za = zab[0];
        auto zb = zab[1];
        auto q = gated_silu_native(za.contiguous(), zb.contiguous());
        auto z3 = silex_tlinear_forward_select(q, z_wpacks[2].contiguous(), z_alphas[2].contiguous(), D_Z, D_MODEL, 0, 10, deterministic_backbone);
        auto zero = torch::zeros_like(cur);
        auto next = residual_native(cur.contiguous(), z3.contiguous(), zero.contiguous(), RHO_Z);
        z_trace.select(0, k).narrow(0, 0, T).copy_(next);
        cur = z_trace.select(0, k).narrow(0, 0, T);
        auto outk = rms_norm_native(cur.contiguous(), gamma_out.contiguous(), EPS_NORM);
        logits_by_depth.push_back(tlinear_forward_native(outk, e_wpack.contiguous(), e_alpha.contiguous(), D_MODEL, VOCAB_SIZE).to(torch::kFloat32));
    }
    profiler.mark("latent_forward_done");

    auto grad_result = curriculum_logits_grad(
        logits_by_depth,
        labels.contiguous(),
        loss_mask.contiguous(),
        stage,
        teacher_logits_final
    );
    logits_by_depth.clear();
    profiler.mark("loss_grad_done");

    std::vector<torch::Tensor> dz;
    dz.reserve(5);
    for (int k = 0; k < 5; ++k) {
        auto zk = z_trace.select(0, k).narrow(0, 0, T);
        auto d_out = tlinear_backward_input_native(grad_result.dlogits[static_cast<size_t>(k)], e_wpack.contiguous(), e_alpha.contiguous(), D_MODEL);
        dz.push_back(rms_norm_backward_native(zk.contiguous(), gamma_out.contiguous(), d_out.contiguous(), EPS_NORM).contiguous());
    }
    grad_result.dlogits.clear();
    profiler.mark("head_backward_done");

    for (int k = 4; k >= 1; --k) {
        auto d_cur = dz[static_cast<size_t>(k)].contiguous();
        auto z_prev = z_trace.select(0, k - 1).narrow(0, 0, T);
        auto n = rms_norm_native(z_prev.contiguous(), gamma_z.contiguous(), EPS_NORM);
        auto zab = silex_tlinear_forward_multi_select(
            n,
            {z_wpacks[0].contiguous(), z_wpacks[1].contiguous()},
            {z_alphas[0].contiguous(), z_alphas[1].contiguous()},
            D_MODEL,
            D_Z,
            0,
            8,
            deterministic_backbone
        );
        auto za = zab[0];
        auto zb = zab[1];
        auto d_q = silex_tlinear_backward_input_select(d_cur * RHO_Z, z_wpacks[2].contiguous(), z_alphas[2].contiguous(), D_Z, D_MODEL, 0, 10, deterministic_backbone);
        auto sw = swiglu_backward_exact(za.contiguous(), zb.contiguous(), d_q.contiguous());
        auto d_n = silex_tlinear_backward_input_select(std::get<0>(sw), z_wpacks[0].contiguous(), z_alphas[0].contiguous(), D_MODEL, D_Z, 0, 8, deterministic_backbone);
        d_n.add_(silex_tlinear_backward_input_select(std::get<1>(sw), z_wpacks[1].contiguous(), z_alphas[1].contiguous(), D_MODEL, D_Z, 0, 9, deterministic_backbone));
        dz[static_cast<size_t>(k - 1)].add_(d_cur);
        dz[static_cast<size_t>(k - 1)].add_(rms_norm_backward_native(z_prev.contiguous(), gamma_z.contiguous(), d_n.contiguous(), EPS_NORM));
    }
    profiler.mark("latent_backward_done");

    std::vector<torch::Tensor> step_params;
    std::vector<torch::Tensor> step_grads;
    std::vector<torch::Tensor> step_a_invs;
    std::vector<torch::Tensor> step_g_invs;
    step_params.reserve(256);
    step_grads.reserve(256);
    step_a_invs.reserve(256);
    step_g_invs.reserve(256);
    const int64_t rec_layer_bytes = 512ll * 8ll * D_MODEL * 2ll;
    const int64_t grad_slot_bytes = 64ll * D_MODEL * 4ll;

    auto d_x = dz[0].contiguous();
    dz.clear();
    for (int l = 63; l >= 0; --l) {
        std::vector<torch::Tensor> layer_params;
        std::vector<torch::Tensor> layer_grads;
        std::vector<torch::Tensor> layer_a_invs;
        std::vector<torch::Tensor> layer_g_invs;
        layer_params.reserve(4);
        layer_grads.reserve(4);
        layer_a_invs.reserve(4);
        layer_g_invs.reserve(4);
        int base = l * 8;
        int layer_id = l + 1;
        auto x_prev = x_trace.select(0, l).narrow(0, 0, T);
        auto u = rms_norm_native(x_prev.contiguous(), gamma_m[l].contiguous(), EPS_NORM);
        auto gate_pre = silex_tlinear_forward_multi_select(
            u,
            {layer_wpacks[base + 0].contiguous(), layer_wpacks[base + 1].contiguous(), layer_wpacks[base + 2].contiguous(), layer_wpacks[base + 3].contiguous()},
            {layer_alphas[base + 0].contiguous(), layer_alphas[base + 1].contiguous(), layer_alphas[base + 2].contiguous(), layer_alphas[base + 3].contiguous()},
            D_MODEL,
            D_MODEL,
            layer_id,
            0,
            deterministic_backbone
        );
        auto pre_i = gate_pre[0];
        auto pre_f = gate_pre[1];
        auto pre_v = gate_pre[2];
        auto pre_r = gate_pre[3];
        auto i_gate = activation_native(pre_i.contiguous(), 0);
        auto f_gate = activation_native(pre_f.contiguous(), 0);
        auto v_val = activation_native(pre_v.contiguous(), 1);
        auto r_gate = activation_native(pre_r.contiguous(), 0);
        auto rec = recurrent_native(
            i_gate.contiguous(),
            f_gate.contiguous(),
            v_val.contiguous(),
            r_gate.contiguous(),
            state.select(0, l).contiguous(),
            lambda_raw[l].contiguous(),
            beta_raw[l].contiguous()
        );
        auto g = std::get<0>(rec);
        auto p_m = adapter_native(u, A_m[l].contiguous(), B_m[l].contiguous());
        auto o = silex_tlinear_forward_select(g, layer_wpacks[base + 4].contiguous(), layer_alphas[base + 4].contiguous(), D_MODEL, D_MODEL, layer_id, 4, deterministic_backbone);
        auto x_tilde = residual_native(x_prev.contiguous(), o.contiguous(), p_m.contiguous(), RHO);
        auto u_f = rms_norm_native(x_tilde.contiguous(), gamma_f[l].contiguous(), EPS_NORM);
        auto ab = silex_tlinear_forward_multi_select(
            u_f,
            {layer_wpacks[base + 5].contiguous(), layer_wpacks[base + 6].contiguous()},
            {layer_alphas[base + 5].contiguous(), layer_alphas[base + 6].contiguous()},
            D_MODEL,
            D_FF,
            layer_id,
            5,
            deterministic_backbone
        );
        auto a = ab[0];
        auto b = ab[1];
        auto h = gated_silu_native(a.contiguous(), b.contiguous());

        auto d_y = d_x.contiguous();
        auto d_x_tilde = d_y.clone();
        auto d_c = d_y * RHO;
        auto d_p_f = d_y * RHO;

        auto f_back = adapter_backward_with_factors(u_f.contiguous(), A_f[l].contiguous(), B_f[l].contiguous(), d_p_f.contiguous());
        auto d_u_f = std::get<0>(f_back).to(torch::kFloat32);
        if (layer_is_active(l, active)) {
            int idx_a_f = 4 * l + 2;
            int idx_b_f = 4 * l + 3;
            update_curvature_and_collect(
                idx_a_f,
                A_f[l],
                std::get<1>(f_back),
                u_f.to(torch::kFloat32),
                std::get<4>(f_back).to(torch::kFloat32),
                kfac_a_covs,
                kfac_g_covs,
                kfac_a_invs,
                kfac_g_invs,
                layer_params,
                layer_grads,
                layer_a_invs,
                layer_g_invs,
                damping,
                ema
            );
            update_curvature_and_collect(
                idx_b_f,
                B_f[l],
                std::get<2>(f_back),
                std::get<3>(f_back).to(torch::kFloat32),
                d_p_f.to(torch::kBFloat16).to(torch::kFloat32),
                kfac_a_covs,
                kfac_g_covs,
                kfac_a_invs,
                kfac_g_invs,
                layer_params,
                layer_grads,
                layer_a_invs,
                layer_g_invs,
                damping,
                ema
            );
        }

        auto d_h = silex_tlinear_backward_input_select(d_c, layer_wpacks[base + 7].contiguous(), layer_alphas[base + 7].contiguous(), D_FF, D_MODEL, layer_id, 7, deterministic_backbone);
        auto sw = swiglu_backward_exact(a.contiguous(), b.contiguous(), d_h.contiguous());
        d_u_f.add_(silex_tlinear_backward_input_select(std::get<0>(sw), layer_wpacks[base + 5].contiguous(), layer_alphas[base + 5].contiguous(), D_MODEL, D_FF, layer_id, 5, deterministic_backbone));
        d_u_f.add_(silex_tlinear_backward_input_select(std::get<1>(sw), layer_wpacks[base + 6].contiguous(), layer_alphas[base + 6].contiguous(), D_MODEL, D_FF, layer_id, 6, deterministic_backbone));
        d_x_tilde.add_(rms_norm_backward_native(x_tilde.contiguous(), gamma_f[l].contiguous(), d_u_f.contiguous(), EPS_NORM));

        auto d_x_prev = d_x_tilde.clone();
        auto d_o = d_x_tilde * RHO;
        auto d_p_m = d_x_tilde * RHO;
        auto m_back = adapter_backward_with_factors(u.contiguous(), A_m[l].contiguous(), B_m[l].contiguous(), d_p_m.contiguous());
        auto d_u = std::get<0>(m_back).to(torch::kFloat32);
        if (layer_is_active(l, active)) {
            int idx_a_m = 4 * l + 0;
            int idx_b_m = 4 * l + 1;
            update_curvature_and_collect(
                idx_a_m,
                A_m[l],
                std::get<1>(m_back),
                u.to(torch::kFloat32),
                std::get<4>(m_back).to(torch::kFloat32),
                kfac_a_covs,
                kfac_g_covs,
                kfac_a_invs,
                kfac_g_invs,
                layer_params,
                layer_grads,
                layer_a_invs,
                layer_g_invs,
                damping,
                ema
            );
            update_curvature_and_collect(
                idx_b_m,
                B_m[l],
                std::get<2>(m_back),
                std::get<3>(m_back).to(torch::kFloat32),
                d_p_m.to(torch::kBFloat16).to(torch::kFloat32),
                kfac_a_covs,
                kfac_g_covs,
                kfac_a_invs,
                kfac_g_invs,
                layer_params,
                layer_grads,
                layer_a_invs,
                layer_g_invs,
                damping,
                ema
            );
        }

        auto d_g = silex_tlinear_backward_input_select(d_o, layer_wpacks[base + 4].contiguous(), layer_alphas[base + 4].contiguous(), D_MODEL, D_MODEL, layer_id, 4, deterministic_backbone);
        auto rec_back = recurrent_mixer_backward_trace(
            i_gate.contiguous(),
            f_gate.contiguous(),
            v_val.contiguous(),
            r_gate.contiguous(),
            state.select(0, l).contiguous(),
            lambda_raw[l].contiguous(),
            beta_raw[l].contiguous(),
            rec_trace.select(0, l).narrow(0, 0, T),
            d_g.contiguous()
        );
        d_u.add_(silex_tlinear_backward_input_select(activation_backward_exact(pre_i.contiguous(), std::get<0>(rec_back), 0), layer_wpacks[base + 0].contiguous(), layer_alphas[base + 0].contiguous(), D_MODEL, D_MODEL, layer_id, 0, deterministic_backbone));
        d_u.add_(silex_tlinear_backward_input_select(activation_backward_exact(pre_f.contiguous(), std::get<1>(rec_back), 0), layer_wpacks[base + 1].contiguous(), layer_alphas[base + 1].contiguous(), D_MODEL, D_MODEL, layer_id, 1, deterministic_backbone));
        d_u.add_(silex_tlinear_backward_input_select(activation_backward_exact(pre_v.contiguous(), std::get<2>(rec_back), 1), layer_wpacks[base + 2].contiguous(), layer_alphas[base + 2].contiguous(), D_MODEL, D_MODEL, layer_id, 2, deterministic_backbone));
        d_u.add_(silex_tlinear_backward_input_select(activation_backward_exact(pre_r.contiguous(), std::get<3>(rec_back), 0), layer_wpacks[base + 3].contiguous(), layer_alphas[base + 3].contiguous(), D_MODEL, D_MODEL, layer_id, 3, deterministic_backbone));
        d_x_prev.add_(rms_norm_backward_native(x_prev.contiguous(), gamma_m[l].contiguous(), d_u.contiguous(), EPS_NORM));
        d_x = d_x_prev.contiguous();
        for (size_t gi = 0; gi < layer_grads.size(); ++gi) {
            auto param = layer_params[gi];
            auto slot = workspace_float_view(
                workspace,
                layout.rec_trace_offset + static_cast<int64_t>(l) * rec_layer_bytes + static_cast<int64_t>(gi) * grad_slot_bytes,
                {param.size(0), param.size(1)}
            );
            slot.copy_(layer_grads[gi]);
            step_params.push_back(param);
            step_grads.push_back(slot);
            step_a_invs.push_back(layer_a_invs[gi]);
            step_g_invs.push_back(layer_g_invs[gi]);
        }
        if (l % 8 == 0) {
            std::string phase = "backward_reached_layer_" + std::to_string(l + 1);
            profiler.mark(phase.c_str());
        }
    }
    profiler.mark("backward_layers_done");

    auto kfac_step = apply_global_kfac_step(
        step_params,
        step_grads,
        step_a_invs,
        step_g_invs,
        eta,
        weight_decay,
        trust_region_delta,
        eps_opt
    );
    double nu = kfac_step.first;
    double chi = kfac_step.second;
    profiler.mark("kfac_step_done");

    constexpr double omega[5] = {2.0 / 30.0, 4.0 / 30.0, 6.0 / 30.0, 8.0 / 30.0, 10.0 / 30.0};
    double weighted_nll = 0.0;
    for (int k = 0; k < 5; ++k) {
        weighted_nll += omega[k] * grad_result.nll_by_k[static_cast<size_t>(k)];
    }

    pybind11::dict out;
    out["new_state"] = new_state;
    out["nll"] = weighted_nll;
    out["nll0"] = grad_result.nll_by_k[0];
    out["nll1"] = grad_result.nll_by_k[1];
    out["nll2"] = grad_result.nll_by_k[2];
    out["nll3"] = grad_result.nll_by_k[3];
    out["nll4"] = grad_result.nll_by_k[4];
    out["mono"] = grad_result.mono;
    out["latent_gain"] = grad_result.latent_gain;
    out["natural_norm"] = nu;
    out["trust_chi"] = chi;
    out["updated_matrices"] = static_cast<int64_t>(step_params.size());
    return out;
}

torch::Tensor pack_ternary(torch::Tensor T) {
    TORCH_CHECK(!T.is_cuda(), "T must be a CPU int8 tensor");
    TORCH_CHECK(T.scalar_type() == torch::kInt8, "T must be int8");
    TORCH_CHECK(T.dim() == 2, "T must have shape [d_out, d_in]");
    TORCH_CHECK(T.is_contiguous(), "T must be contiguous");

    int d_out = static_cast<int>(T.size(0));
    int d_in = static_cast<int>(T.size(1));
    TORCH_CHECK(valid_d_in(d_in), "d_in must be one of {4096, 8192, 16384}");
    TORCH_CHECK(valid_d_out(d_out), "d_out must be one of {258, 4096, 8192, 16384}");

    auto Wpack = torch::empty({d_out, row_stride_bytes(d_in)}, torch::TensorOptions().dtype(torch::kUInt8));
    pack_ternary_matrix_cpu(T.data_ptr<int8_t>(), Wpack.data_ptr<uint8_t>(), d_out, d_in);
    return Wpack;
}

std::tuple<torch::Tensor, torch::Tensor> deterministic_pack_ternary(int64_t d_out64, int64_t d_in64, int64_t layer64, int64_t matrix_id64) {
    int d_out = static_cast<int>(d_out64);
    int d_in = static_cast<int>(d_in64);
    int layer = static_cast<int>(layer64);
    int matrix_id = static_cast<int>(matrix_id64);
    TORCH_CHECK(valid_d_in(d_in), "d_in must be one of {4096, 8192, 16384}");
    TORCH_CHECK(valid_d_out(d_out), "d_out must be one of {258, 4096, 8192, 16384}");
    TORCH_CHECK(matrix_id >= 0 && matrix_id <= 11, "matrix_id must be in [0, 11]");

    int stride = row_stride_bytes(d_in);
    auto Wpack = torch::empty({d_out, stride}, torch::TensorOptions().dtype(torch::kUInt8));
    auto alpha = torch::empty({d_out}, torch::TensorOptions().dtype(torch::kBFloat16));
    auto* wptr = Wpack.data_ptr<uint8_t>();
    auto* aptr = alpha.data_ptr<at::BFloat16>();

    static constexpr uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int o = 0; o < d_out; ++o) {
        int nonzero = 0;
        for (int kb = 0; kb < stride; ++kb) {
            uint32_t byte_value = 0;
            for (int r = 0; r < 5; ++r) {
                int i = kb * 5 + r;
                int c = 1;
                if (i < d_in) {
                    int8_t w = deterministic_trit(o, i, layer, matrix_id, d_in);
                    nonzero += (w != 0);
                    c = static_cast<int>(w) + 1;
                }
                byte_value += static_cast<uint32_t>(c) * static_cast<uint32_t>(pow3[r]);
            }
            TORCH_CHECK(byte_value <= 242u, "invalid ternary packed byte");
            wptr[o * stride + kb] = static_cast<uint8_t>(byte_value);
        }
        float a = 1.0f / std::sqrt(static_cast<float>(std::max(1, nonzero)));
        aptr[o] = static_cast<at::BFloat16>(a);
    }

    return std::make_tuple(Wpack, alpha);
}

std::tuple<torch::Tensor, torch::Tensor> deterministic_pack_ternary_cuda(int64_t d_out64, int64_t d_in64, int64_t layer64, int64_t matrix_id64) {
    int d_out = static_cast<int>(d_out64);
    int d_in = static_cast<int>(d_in64);
    int layer = static_cast<int>(layer64);
    int matrix_id = static_cast<int>(matrix_id64);
    TORCH_CHECK(valid_d_in(d_in), "d_in must be one of {4096, 8192, 16384}");
    TORCH_CHECK(valid_d_out(d_out), "d_out must be one of {258, 4096, 8192, 16384}");
    TORCH_CHECK(matrix_id >= 0 && matrix_id <= 11, "matrix_id must be in [0, 11]");

    const c10::cuda::CUDAGuard device_guard(0);
    int stride = row_stride_bytes(d_in);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto Wpack = torch::empty({d_out, stride}, opts_u8);
    auto alpha = torch::empty({d_out}, opts_bf16);
    launch_deterministic_pack_ternary_cuda(
        Wpack.data_ptr<uint8_t>(),
        alpha.data_ptr<at::BFloat16>(),
        d_out,
        d_in,
        layer,
        matrix_id
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(Wpack, alpha);
}

int64_t row_stride_bytes_binding(int64_t d_in) {
    return row_stride_bytes(static_cast<int>(d_in));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_stride_bytes", &row_stride_bytes_binding, "S5(d_in) row stride");
    m.def("init_trit_lut", &init_trit_lut_cuda, "Initialize CUDA trit LUT");
    m.def("pack_ternary", &pack_ternary, "Pack CPU int8 ternary matrix");
    m.def("deterministic_pack_ternary", &deterministic_pack_ternary, "Deterministic TDD ternary initialization");
    m.def("deterministic_pack_ternary_cuda", &deterministic_pack_ternary_cuda, "Deterministic TDD ternary initialization directly on CUDA");
    m.def("tlinear_forward", &tlinear_forward, "TLinear forward CUDA");
    m.def("tlinear_backward_input", &tlinear_backward_input, "TLinear backward input CUDA");
    m.def("deterministic_tlinear_forward", &deterministic_tlinear_forward_native, "Fast deterministic Hadamard TLinear forward CUDA");
    m.def("deterministic_tlinear_forward_multi", &deterministic_tlinear_forward_multi_native, "Fast deterministic Hadamard multi-output TLinear forward CUDA");
    m.def("deterministic_tlinear_backward_input", &deterministic_tlinear_backward_input_native, "Fast deterministic Hadamard TLinear backward-input CUDA");
    m.def("embedding_forward", &embedding_forward, "Byte-level ternary embedding CUDA");
    m.def("recurrent_mixer_forward", &recurrent_mixer_forward, "Recurrent mixer forward CUDA");
    m.def("rms_norm_forward", &rms_norm_forward, "RMSNorm forward CUDA");
    m.def("activation_forward", &activation_forward, "Activation forward CUDA");
    m.def("gated_silu_product", &gated_silu_product, "SwiGLU product CUDA");
    m.def("residual_add_forward", &residual_add_forward, "Residual add CUDA");
    m.def("adapter_forward", &adapter_forward, "Low-rank plastic adapter forward CUDA");
    m.def("adapter_backward_exact", &adapter_backward_exact, "Low-rank plastic adapter backward CUDA");
    m.def("adapter_backward_with_factors", &adapter_backward_with_factors, "Low-rank adapter backward plus K-FAC factors CUDA");
    m.def("activation_backward_exact", &activation_backward_exact, "Sigmoid/Silu backward CUDA");
    m.def("rms_norm_backward_exact", &rms_norm_backward_exact, "RMSNorm backward CUDA");
    m.def("swiglu_backward_exact", &swiglu_backward_exact, "SwiGLU backward CUDA");
    m.def("recurrent_mixer_backward_exact", &recurrent_mixer_backward_exact, "Recurrent mixer BPTT backward CUDA");
    m.def("block_kfac_update_curvature", &block_kfac_update_curvature, "Block-KFAC curvature update CUDA");
    m.def("block_kfac_step_param", &block_kfac_step_param, "Block-KFAC single-parameter natural update CUDA");
    m.def("silex_forward_cuda", &silex_forward_cuda, "SilexCode native full forward CUDA");
    m.def("silex_forward_cuda_output_adapter", &silex_forward_cuda_output_adapter, "SilexCode native full forward CUDA with experimental output adapter");
    m.def("silex_train_workspace_bytes", &silex_train_workspace_bytes, "Static training workspace bytes");
    m.def("silex_train_workspace_layout", &silex_train_workspace_layout, "Static training workspace layout");
    m.def("silex_train_chunk_cuda", &silex_train_chunk_cuda, "SilexCode native training chunk CUDA");
    m.def("silex_train_chunk_cuda", &silex_train_chunk_cuda_update, "SilexCode native training chunk CUDA with backward and K-FAC update");
}
