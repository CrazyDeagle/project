import torch
import pytest

from silexcode.constants import SILEX_T18_6B_R64_CONFIG
from silexcode.kfac import BlockKFACOptimizer
from silexcode.losses import latent_depth_weights, silex_latent_loss
from silexcode.model import _load_extension, low_rank_adapter


def test_latent_depth_weights_sum_to_one() -> None:
    w = latent_depth_weights(4, device=torch.device("cpu"))
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    assert torch.all(w[1:] > w[:-1])


def test_latent_loss_shapes_and_terms() -> None:
    cfg = SILEX_T18_6B_R64_CONFIG
    logits = [torch.zeros(cfg.u_train, cfg.vocab_size) for _ in range(cfg.k_train + 1)]
    tokens = torch.arange(cfg.u_train) % cfg.vocab_size
    plastic = [torch.nn.Parameter(torch.zeros(64, 4096))]
    parts = silex_latent_loss(logits, tokens, plastic, cfg)
    assert set(parts) == {"loss", "nll", "mono", "kl", "mdl"}
    assert torch.isfinite(parts["loss"])
    assert parts["mono"].item() == 0.0
    assert parts["kl"].item() == 0.0


def test_block_kfac_single_matrix_step() -> None:
    p = torch.nn.Parameter(torch.zeros(64, 4096, dtype=torch.float32))
    opt = BlockKFACOptimizer([("A", p)])
    inputs = torch.randn(8, 4096)
    grad_outputs = torch.randn(8, 64)
    opt.update_curvature("A", inputs, grad_outputs)
    p.grad = torch.ones_like(p) * 0.01
    nu = opt.step()
    assert nu >= 0.0
    assert torch.isfinite(p).all()


def test_block_kfac_stage_hyperparams_and_active_layers() -> None:
    p1 = torch.nn.Parameter(torch.zeros(64, 4096, dtype=torch.float32))
    p2 = torch.nn.Parameter(torch.zeros(64, 4096, dtype=torch.float32))
    opt = BlockKFACOptimizer([("layers.0.A_m", p1), ("layers.20.A_m", p2)])
    opt.reset_curvature(active_layers=list(range(1, 17)), damping=3e-4)
    opt.set_hyperparams(eta=0.04, damping=3e-4, trust_region_delta=5e-4)
    p1.grad = torch.ones_like(p1) * 0.01
    p2.grad = torch.ones_like(p2) * 0.01
    opt.step()
    assert p1.abs().sum() > 0
    assert p2.abs().sum() == 0


def test_native_workspace_layout_matches_tdd_budget() -> None:
    ext = _load_extension()
    layout = ext.silex_train_workspace_layout(512)
    assert layout["x_offset"] == 0
    assert layout["rec_trace_offset"] == 65 * 512 * 4096 * 2
    assert layout["z_trace_offset"] == layout["rec_trace_offset"] + 64 * 512 * 8 * 4096 * 2
    assert layout["ff_offset"] == layout["z_trace_offset"] + 5 * 512 * 4096 * 2
    assert layout["mix_offset"] == layout["ff_offset"] + 3 * 512 * 16384 * 2
    assert layout["logits_offset"] == layout["mix_offset"] + 6 * 512 * 4096 * 2
    assert layout["total_bytes"] == 2517110784
    assert ext.silex_train_workspace_bytes(512) == layout["total_bytes"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_adapter_backward_matches_pytorch_autograd() -> None:
    ext = _load_extension()
    torch.manual_seed(10)
    x = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    A = (torch.randn(64, 4096, device="cuda", dtype=torch.float32) * 0.01).requires_grad_()
    B = (torch.randn(4096, 64, device="cuda", dtype=torch.float32) * 0.01).requires_grad_()
    grad_out = torch.randn(3, 4096, device="cuda", dtype=torch.float32)

    out = low_rank_adapter(x, A, B)
    out.float().backward(grad_out)

    grad_x, grad_A, grad_B = ext.adapter_backward_exact(
        x.detach(),
        A.detach(),
        B.detach(),
        grad_out,
    )
    assert torch.allclose(grad_x.float(), x.grad.float(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(grad_A, A.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(grad_B, B.grad, atol=1e-4, rtol=1e-4)

    grad_x2, grad_A2, grad_B2, hidden, grad_hidden = ext.adapter_backward_with_factors(
        x.detach(),
        A.detach(),
        B.detach(),
        grad_out,
    )
    assert torch.allclose(grad_x2.float(), x.grad.float(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(grad_A2, A.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(grad_B2, B.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(hidden, torch.einsum("td,rd->tr", x.detach().float(), A.detach()), atol=1e-4, rtol=1e-4)
    assert torch.allclose(grad_hidden, torch.einsum("td,dr->tr", grad_out.to(torch.bfloat16).float(), B.detach()), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_rmsnorm_and_swiglu_backward_match_autograd() -> None:
    ext = _load_extension()
    torch.manual_seed(11)
    x = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gamma = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    grad_y = torch.randn(2, 4096, device="cuda", dtype=torch.float32)
    y = (x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 2.0**-12) * gamma.float())
    y.backward(grad_y)
    dx = ext.rms_norm_backward_exact(x.detach(), gamma, grad_y, 2.0**-12)
    assert torch.allclose(dx.float(), x.grad.float(), atol=1e-4, rtol=1e-4)

    a = torch.randn(2, 8192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2, 8192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_h = torch.randn(2, 8192, device="cuda", dtype=torch.float32)
    h = torch.nn.functional.silu(a.float()) * b.float()
    h.backward(grad_h)
    da, db = ext.swiglu_backward_exact(a.detach(), b.detach(), grad_h)
    assert torch.allclose(da.float(), a.grad.float(), atol=1e-4, rtol=1e-4)
    assert torch.allclose(db.float(), b.grad.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_activation_backward_matches_autograd() -> None:
    ext = _load_extension()
    torch.manual_seed(14)
    x = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(3, 4096, device="cuda", dtype=torch.float32)
    y = torch.sigmoid(x.float())
    y.backward(grad)
    dx = ext.activation_backward_exact(x.detach(), grad, 0)
    assert torch.allclose(dx.float(), x.grad.float(), atol=1e-4, rtol=1e-4)

    x2 = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad2 = torch.randn(3, 4096, device="cuda", dtype=torch.float32)
    y2 = torch.nn.functional.silu(x2.float())
    y2.backward(grad2)
    dx2 = ext.activation_backward_exact(x2.detach(), grad2, 1)
    assert torch.allclose(dx2.float(), x2.grad.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_recurrent_mixer_backward_matches_autograd_bptt() -> None:
    ext = _load_extension()
    torch.manual_seed(13)
    t = 5
    d = 4096
    r = 8
    i_gate = torch.rand(t, d, device="cuda", dtype=torch.float32, requires_grad=True)
    f_gate = torch.rand(t, d, device="cuda", dtype=torch.float32, requires_grad=True)
    v_val = torch.randn(t, d, device="cuda", dtype=torch.float32, requires_grad=True)
    r_gate = torch.rand(t, d, device="cuda", dtype=torch.float32, requires_grad=True)
    state = torch.randn(r, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lambda_raw = torch.randn(r, d, device="cuda", dtype=torch.bfloat16)
    beta_raw = torch.randn(r, d, device="cuda", dtype=torch.bfloat16)
    grad_g = torch.randn(t, d, device="cuda", dtype=torch.float32)

    lam = torch.sigmoid(lambda_raw.float())
    beta = torch.softmax(beta_raw.float(), dim=0)
    prev = state.float()
    outs = []
    for idx in range(t):
        mix = lam * f_gate[idx].unsqueeze(0)
        write = (i_gate[idx] * v_val[idx]).unsqueeze(0)
        prev = mix * prev + (1.0 - mix) * write
        outs.append(r_gate[idx] * (beta * prev).sum(dim=0))
    torch.stack(outs, dim=0).backward(grad_g)

    d_i, d_f, d_v, d_r, d_state = ext.recurrent_mixer_backward_exact(
        i_gate.detach(),
        f_gate.detach(),
        v_val.detach(),
        r_gate.detach(),
        state.detach(),
        lambda_raw,
        beta_raw,
        grad_g,
    )
    assert torch.allclose(d_i, i_gate.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(d_f, f_gate.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(d_v, v_val.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(d_r, r_gate.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(d_state.float(), state.grad.float(), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_kfac_curvature_and_step_match_python_reference() -> None:
    ext = _load_extension()
    torch.manual_seed(12)
    p_py = torch.nn.Parameter(torch.randn(64, 4096, device="cuda", dtype=torch.float32) * 0.001)
    p_native = torch.nn.Parameter(p_py.detach().clone())
    opt = BlockKFACOptimizer([("layers.0.A_m", p_py)])
    opt.reset_curvature(active_layers=[1], damping=1e-3)
    inputs = torch.randn(7, 4096, device="cuda", dtype=torch.float32)
    grad_outputs = torch.randn(7, 64, device="cuda", dtype=torch.float32)
    opt.update_curvature("layers.0.A_m", inputs, grad_outputs)

    st = opt.state["layers.0.A_m"]
    a_cov = torch.eye(64, device="cuda", dtype=torch.float32).expand(64, 64, 64).clone()
    g_cov = torch.eye(64, device="cuda", dtype=torch.float32).expand(1, 64, 64).clone()
    a_inv = a_cov.clone()
    g_inv = g_cov.clone()
    ext.block_kfac_update_curvature(inputs, grad_outputs, a_cov, g_cov, a_inv, g_inv, 1e-3, 0.05)
    assert torch.allclose(a_cov, st.a_cov, atol=1e-4, rtol=1e-4)
    assert torch.allclose(g_cov, st.g_cov, atol=1e-4, rtol=1e-4)
    assert torch.allclose(a_inv, st.a_inv, atol=1e-4, rtol=1e-4)
    assert torch.allclose(g_inv, st.g_inv, atol=1e-4, rtol=1e-4)

    grad = torch.randn_like(p_py) * 0.01
    p_py.grad = grad.clone()
    p_native.grad = grad.clone()
    nu_py = opt.step(active_layers=[1], eta=0.08, damping=1e-3, trust_region_delta=1e-3)
    nu_native = ext.block_kfac_step_param(
        p_native,
        p_native.grad,
        a_inv,
        g_inv,
        0.08,
        1e-5,
        1e-3,
        SILEX_T18_6B_R64_CONFIG.eps_opt,
    )
    assert abs(nu_py - nu_native) <= 1e-3
    assert torch.allclose(p_native, p_py, atol=1e-4, rtol=1e-4)
