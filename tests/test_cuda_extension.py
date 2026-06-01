import pytest
import torch

from silexcode.model import _load_extension


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tlinear_forward_and_backward_input_match_reference() -> None:
    ext = _load_extension()
    torch.manual_seed(20)
    t = 9
    d_in = 4096
    d_out = 258
    w = torch.randint(-1, 2, (d_out, d_in), dtype=torch.int8)
    wpack = ext.pack_ternary(w).cuda()
    alpha = (torch.rand(d_out, device="cuda") * 0.25 + 0.75).to(torch.bfloat16)
    x = torch.randn(t, d_in, device="cuda", dtype=torch.bfloat16)

    y = ext.tlinear_forward(x, wpack, alpha)
    ref_y = (x.float() @ w.cuda().float().t() * alpha.float()).to(torch.bfloat16)
    assert torch.allclose(y.float(), ref_y.float(), atol=0.02, rtol=0.02)

    dy = torch.randn(t, d_out, device="cuda", dtype=torch.bfloat16)
    dx = ext.tlinear_backward_input(dy, wpack, alpha, d_in)
    ref_dx = (dy.float() * alpha.float()) @ w.cuda().float()
    assert torch.allclose(dx, ref_dx, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_deterministic_tlinear_matches_packed_kernel() -> None:
    ext = _load_extension()
    torch.manual_seed(21)
    cases = [
        (4096, 4096, 1, 0),
        (4096, 16384, 1, 5),
        (16384, 4096, 1, 7),
        (8192, 4096, 0, 10),
    ]
    for d_in, d_out, layer, matrix_id in cases:
        wpack, alpha = ext.deterministic_pack_ternary_cuda(d_out, d_in, layer, matrix_id)
        x = torch.randn(2, d_in, device="cuda", dtype=torch.bfloat16)
        packed_y = ext.tlinear_forward(x, wpack, alpha)
        fast_y = ext.deterministic_tlinear_forward(
            x.contiguous(), alpha.contiguous(), d_in, d_out, layer, matrix_id
        )
        assert torch.allclose(fast_y.float(), packed_y.float(), atol=0.02, rtol=0.02)

        dy = torch.randn(2, d_out, device="cuda", dtype=torch.bfloat16)
        packed_dx = ext.tlinear_backward_input(dy, wpack, alpha, d_in)
        fast_dx = ext.deterministic_tlinear_backward_input(
            dy.contiguous(), alpha.contiguous(), d_in, d_out, layer, matrix_id
        )
        assert torch.allclose(fast_dx, packed_dx, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_recurrent_mixer_forward_matches_reference() -> None:
    ext = _load_extension()
    torch.manual_seed(0)
    t = 3
    d = 4096
    r = 8
    i_gate = torch.rand(t, d, device="cuda", dtype=torch.float32)
    f_gate = torch.rand(t, d, device="cuda", dtype=torch.float32)
    v_val = torch.randn(t, d, device="cuda", dtype=torch.float32)
    r_gate = torch.rand(t, d, device="cuda", dtype=torch.float32)
    state = torch.randn(r, d, device="cuda", dtype=torch.bfloat16)
    lambda_raw = torch.randn(r, d, device="cuda", dtype=torch.bfloat16)
    beta_raw = torch.randn(r, d, device="cuda", dtype=torch.bfloat16)

    g, new_state = ext.recurrent_mixer_forward(
        i_gate, f_gate, v_val, r_gate, state, lambda_raw, beta_raw
    )

    lam = torch.sigmoid(lambda_raw.float())
    beta = torch.softmax(beta_raw.float(), dim=0)
    prev = state.float()
    outs = []
    for idx in range(t):
        mix = lam * f_gate[idx].unsqueeze(0)
        write = (i_gate[idx] * v_val[idx]).unsqueeze(0)
        prev = mix * prev + (1.0 - mix) * write
        outs.append((r_gate[idx] * (beta * prev).sum(dim=0)).to(torch.bfloat16))
    ref_g = torch.stack(outs, dim=0)
    ref_state = prev.to(torch.bfloat16)

    assert torch.allclose(g.float(), ref_g.float(), atol=0.02, rtol=0.02)
    assert torch.allclose(new_state.float(), ref_state.float(), atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_elementwise_cuda_kernels_match_reference() -> None:
    ext = _load_extension()
    torch.manual_seed(1)
    x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    y = ext.rms_norm_forward(x, gamma, 2.0**-12)
    ref_y = (
        x.float()
        * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 2.0**-12)
        * gamma.float()
    ).to(torch.bfloat16)
    assert torch.allclose(y.float(), ref_y.float(), atol=0.02, rtol=0.02)

    sig = ext.activation_forward(x, 0)
    silu = ext.activation_forward(x, 1)
    assert torch.allclose(sig, torch.sigmoid(x.float()), atol=1e-4, rtol=1e-4)
    assert torch.allclose(silu, torch.nn.functional.silu(x.float()), atol=1e-3, rtol=1e-3)

    a = torch.randn(2, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 8192, device="cuda", dtype=torch.bfloat16)
    h = ext.gated_silu_product(a, b)
    ref_h = (torch.nn.functional.silu(a.float()) * b.float()).to(torch.bfloat16)
    assert torch.allclose(h.float(), ref_h.float(), atol=0.02, rtol=0.02)

    adapter = torch.randn_like(x)
    out = ext.residual_add_forward(x, y, adapter, 0.125)
    ref_out = (x.float() + 0.125 * (y.float() + adapter.float())).to(torch.bfloat16)
    assert torch.allclose(out.float(), ref_out.float(), atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_adapter_forward_matches_reference_and_backward_shapes() -> None:
    from silexcode.model import low_rank_adapter

    torch.manual_seed(2)
    x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    A = torch.randn(64, 4096, device="cuda", dtype=torch.float32, requires_grad=True) * 0.01
    B = torch.randn(4096, 64, device="cuda", dtype=torch.float32, requires_grad=True) * 0.01
    A.retain_grad()
    B.retain_grad()
    out = low_rank_adapter(x, A, B)
    ref = torch.einsum("tr,dr->td", torch.einsum("td,rd->tr", x.float(), A), B).to(torch.bfloat16)
    assert torch.allclose(out.float(), ref.float(), atol=0.03, rtol=0.03)
    out.float().sum().backward()
    assert x.grad is not None and tuple(x.grad.shape) == (2, 4096)
    assert A.grad is not None and tuple(A.grad.shape) == (64, 4096)
    assert B.grad is not None and tuple(B.grad.shape) == (4096, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_silex_forward_cuda_zero_model_smoke() -> None:
    ext = _load_extension()
    device = "cuda"
    zero_byte_4096 = 121
    zero_byte_16384 = 121
    e_wpack = torch.full((258, 832), zero_byte_4096, dtype=torch.uint8, device=device)
    e_alpha = torch.ones(258, dtype=torch.bfloat16, device=device)
    w_d_d = torch.full((4096, 832), zero_byte_4096, dtype=torch.uint8, device=device)
    w_ff_d = torch.full((16384, 832), zero_byte_4096, dtype=torch.uint8, device=device)
    w_d_ff = torch.full((4096, 3296), zero_byte_16384, dtype=torch.uint8, device=device)
    a_d = torch.ones(4096, dtype=torch.bfloat16, device=device)
    a_ff = torch.ones(16384, dtype=torch.bfloat16, device=device)
    layer_wpacks = []
    layer_alphas = []
    for _ in range(64):
        layer_wpacks.extend([w_d_d, w_d_d, w_d_d, w_d_d, w_d_d, w_ff_d, w_ff_d, w_d_ff])
        layer_alphas.extend([a_d, a_d, a_d, a_d, a_d, a_ff, a_ff, a_d])
    gamma_m = [torch.ones(4096, dtype=torch.bfloat16, device=device) for _ in range(64)]
    gamma_f = [torch.ones(4096, dtype=torch.bfloat16, device=device) for _ in range(64)]
    lambda_raw = [torch.zeros(8, 4096, dtype=torch.bfloat16, device=device) for _ in range(64)]
    beta_raw = [torch.zeros(8, 4096, dtype=torch.bfloat16, device=device) for _ in range(64)]
    A = torch.zeros(64, 4096, dtype=torch.float32, device=device)
    B = torch.zeros(4096, 64, dtype=torch.float32, device=device)
    A_m = [A for _ in range(64)]
    B_m = [B for _ in range(64)]
    A_f = [A for _ in range(64)]
    B_f = [B for _ in range(64)]
    z_wpacks = [
        torch.full((8192, 832), zero_byte_4096, dtype=torch.uint8, device=device),
        torch.full((8192, 832), zero_byte_4096, dtype=torch.uint8, device=device),
        torch.full((4096, 1664), zero_byte_4096, dtype=torch.uint8, device=device),
    ]
    z_alphas = [
        torch.ones(8192, dtype=torch.bfloat16, device=device),
        torch.ones(8192, dtype=torch.bfloat16, device=device),
        torch.ones(4096, dtype=torch.bfloat16, device=device),
    ]
    tokens = torch.tensor([256, 65], dtype=torch.uint16, device=device)
    state = torch.zeros(64, 8, 4096, dtype=torch.bfloat16, device=device)
    gamma_z = torch.ones(4096, dtype=torch.bfloat16, device=device)
    gamma_out = torch.ones(4096, dtype=torch.bfloat16, device=device)

    logits, new_state, depths = ext.silex_forward_cuda(
        tokens,
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
        1,
        True,
        False,
    )
    assert tuple(logits.shape) == (2, 258)
    assert tuple(new_state.shape) == (64, 8, 4096)
    assert len(depths) == 2
    assert torch.all(logits == 0)
    assert torch.all(new_state == 0)

    if not hasattr(ext, "silex_forward_cuda_output_adapter"):
        pytest.skip("local extension was not rebuilt with output adapter binding")

    output_down = torch.randn(64, 4096, dtype=torch.float32, device=device) * 0.01
    output_up = torch.zeros(258, 64, dtype=torch.float32, device=device)
    adapter_logits, adapter_state, adapter_depths = ext.silex_forward_cuda_output_adapter(
        tokens,
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
        output_down,
        output_up,
        1,
        True,
        False,
    )
    assert torch.equal(adapter_logits, logits)
    assert torch.equal(adapter_state, new_state)
    assert len(adapter_depths) == len(depths)
