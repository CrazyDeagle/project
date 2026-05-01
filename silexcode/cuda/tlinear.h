#pragma once

#include <torch/extension.h>

int ceil_div_int(int a, int b);
int align_up_int(int x, int a);
int row_stride_bytes(int d_in);

void init_trit_lut_cuda();

void launch_tlinear_forward(
    const at::BFloat16* X,
    const uint8_t* Wpack,
    const at::BFloat16* alpha,
    at::BFloat16* Y,
    int T,
    int d_in,
    int d_out
);

void launch_tlinear_backward_input(
    const at::BFloat16* dY,
    const uint8_t* Wpack,
    const at::BFloat16* alpha,
    float* dX,
    int T,
    int d_in,
    int d_out
);

void launch_deterministic_tlinear_forward(
    const at::BFloat16* X,
    const at::BFloat16* alpha,
    at::BFloat16* Y,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
);

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
);

void launch_deterministic_tlinear_backward_input(
    const at::BFloat16* dY,
    const at::BFloat16* alpha,
    float* dX,
    int T,
    int d_in,
    int d_out,
    int layer,
    int matrix_id
);

void launch_embedding_forward(
    const uint16_t* token_ids,
    const uint8_t* Epack,
    const at::BFloat16* alpha,
    at::BFloat16* X,
    int T,
    int d
);

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
);

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
);

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
);

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
);

void launch_rms_norm_forward(
    const at::BFloat16* x,
    const at::BFloat16* gamma,
    at::BFloat16* y,
    int T,
    int d,
    float eps
);

void launch_rms_norm_backward(
    const at::BFloat16* x,
    const at::BFloat16* gamma,
    const float* grad_y,
    float* grad_x,
    int T,
    int d,
    float eps
);

void launch_activation_forward(
    const at::BFloat16* x,
    float* y,
    int total,
    int activation
);

void launch_activation_backward(
    const at::BFloat16* x,
    const float* grad_out,
    float* grad_x,
    int total,
    int activation
);

void launch_activation_backward_bf16(
    const at::BFloat16* x,
    const float* grad_out,
    at::BFloat16* grad_x,
    int total,
    int activation
);

void launch_gated_silu_product(
    const at::BFloat16* a,
    const at::BFloat16* b,
    at::BFloat16* h,
    int total
);

void launch_swiglu_backward(
    const at::BFloat16* a,
    const at::BFloat16* b,
    const float* grad_h,
    float* grad_a,
    float* grad_b,
    int total
);

void launch_swiglu_backward_bf16(
    const at::BFloat16* a,
    const at::BFloat16* b,
    const float* grad_h,
    at::BFloat16* grad_a,
    at::BFloat16* grad_b,
    int total
);

void launch_residual_add_forward(
    const at::BFloat16* base,
    const at::BFloat16* ternary,
    const at::BFloat16* adapter,
    at::BFloat16* out,
    int total,
    float rho
);

void launch_adapter_forward(
    const at::BFloat16* x,
    const float* A,
    const float* B,
    float* hidden,
    at::BFloat16* out,
    int T,
    int d,
    int r
);

void pack_ternary_matrix_cpu(
    const int8_t* T,
    uint8_t* Wpack,
    int d_out,
    int d_in
);

void launch_deterministic_pack_ternary_cuda(
    uint8_t* Wpack,
    at::BFloat16* alpha,
    int d_out,
    int d_in,
    int layer,
    int matrix_id
);
