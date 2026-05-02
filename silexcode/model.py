from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.cpp_extension import load

from .constants import (
    SILEX_T18_6B_R64_CONFIG,
    TOKEN_EOS,
    TOKEN_BOS,
    VALID_D_IN,
    VALID_D_OUT,
    SilexConfig,
    s5,
)


_EXT = None


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]

    try:
        _EXT = importlib.import_module("silexcode._C")
        return _EXT
    except ImportError:
        root = Path(__file__).resolve().parent
        _EXT = load(
            name="silexcode_ext",
            sources=[
                str(root / "cuda" / "bindings.cpp"),
                str(root / "cuda" / "tlinear_kernels.cu"),
            ],
            extra_cflags=["/O2", "/Zc:preprocessor"] if os.name == "nt" else ["-O2"],
            extra_cuda_cflags=(
                ["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"]
                if os.name == "nt"
                else ["-O3", "--use_fast_math"]
            ),
            with_cuda=True,
            verbose=False,
        )
        return _EXT


def _require_cuda_bf16_tensor(name: str, x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype is not torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _hadamard(a: int, b: int) -> float:
    return 1.0 if ((a & b).bit_count() & 1) == 0 else -1.0


def _plastic_a(layer_zero_based: int, kind: str, device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    offset = 0 if kind == "m" else 2048
    base = 64 * layer_zero_based + offset
    rows = ((base + torch.arange(64, device=device, dtype=torch.int64)) % 4096).view(64, 1)
    cols = torch.arange(4096, device=device, dtype=torch.int64).view(1, 4096)
    x = torch.bitwise_and(rows, cols)
    parity = x
    parity = torch.bitwise_xor(parity, parity >> 32)
    parity = torch.bitwise_xor(parity, parity >> 16)
    parity = torch.bitwise_xor(parity, parity >> 8)
    parity = torch.bitwise_xor(parity, parity >> 4)
    parity = torch.bitwise_xor(parity, parity >> 2)
    parity = torch.bitwise_xor(parity, parity >> 1)
    return torch.where((parity & 1).eq(0), 1.0, -1.0).to(torch.float32).mul_(2.0**-6)


def _output_adapter_down(rank: int, d_model: int, device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    rows = torch.arange(rank, device=device, dtype=torch.int64).view(rank, 1)
    cols = torch.arange(d_model, device=device, dtype=torch.int64).view(1, d_model)
    x = torch.bitwise_and(rows, cols)
    parity = x
    parity = torch.bitwise_xor(parity, parity >> 32)
    parity = torch.bitwise_xor(parity, parity >> 16)
    parity = torch.bitwise_xor(parity, parity >> 8)
    parity = torch.bitwise_xor(parity, parity >> 4)
    parity = torch.bitwise_xor(parity, parity >> 2)
    parity = torch.bitwise_xor(parity, parity >> 1)
    return torch.where((parity & 1).eq(0), 1.0, -1.0).to(torch.float32).mul_(2.0**-6)


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return _load_extension().rms_norm_forward(x.contiguous(), gamma.contiguous(), float(eps))


def sigmoid_bf16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    return _load_extension().activation_forward(x.contiguous(), 0)


def silu_bf16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    return _load_extension().activation_forward(x.contiguous(), 1)


def gated_silu_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _load_extension().gated_silu_product(a.contiguous(), b.contiguous())


def residual_add(base: torch.Tensor, ternary: torch.Tensor, adapter: torch.Tensor, rho: float) -> torch.Tensor:
    return _load_extension().residual_add_forward(
        base.contiguous(),
        ternary.contiguous(),
        adapter.contiguous(),
        float(rho),
    )


def sample_token_top_k_top_p(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    generator: torch.Generator,
) -> int:
    if logits.dim() != 1 or logits.numel() != 258:
        raise ValueError("logits must have shape [258]")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    scores = logits.float()
    if temperature == 0.0:
        return int(torch.argmax(scores).item())
    scores = scores / float(temperature)

    if 0 < top_k < scores.numel():
        values, indices = torch.topk(scores, top_k)
        filtered = torch.full_like(scores, -torch.inf)
        filtered.scatter_(0, indices, values)
        scores = filtered

    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores = scores.clone()
        scores[sorted_indices[remove]] = -torch.inf

    probs = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits.float()).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


def low_rank_adapter(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return _LowRankAdapterFn.apply(x.contiguous(), A.contiguous(), B.contiguous())


class _LowRankAdapterFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        out, hidden = _load_extension().adapter_forward(x, A, B)
        ctx.save_for_backward(x, A, B, hidden)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, A, B, hidden = ctx.saved_tensors
        grad = grad_out.float()
        grad_hidden = torch.einsum("td,dr->tr", grad, B)
        grad_B = torch.einsum("td,tr->dr", grad, hidden)
        grad_A = torch.einsum("tr,td->rd", grad_hidden, x.float())
        grad_x = torch.einsum("tr,rd->td", grad_hidden, A).to(torch.bfloat16)
        return grad_x, grad_A, grad_B


class _TLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, wpack: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        _require_cuda_bf16_tensor("x", x)
        if wpack.dtype is not torch.uint8 or not wpack.is_cuda or not wpack.is_contiguous():
            raise ValueError("wpack must be a contiguous CUDA uint8 tensor")
        _require_cuda_bf16_tensor("alpha", alpha)

        ext = _load_extension()
        y = ext.tlinear_forward(x, wpack, alpha)
        ctx.save_for_backward(wpack, alpha)
        ctx.d_in = x.shape[-1]
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        wpack, alpha = ctx.saved_tensors
        ext = _load_extension()
        grad_y_bf16 = grad_y.contiguous().to(torch.bfloat16)
        grad_x = ext.tlinear_backward_input(grad_y_bf16, wpack, alpha, int(ctx.d_in))
        return grad_x, None, None


class TLinear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        layer: int,
        matrix_id: int,
        device: torch.device | str = "cuda",
        wpack: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        if d_in not in VALID_D_IN:
            raise ValueError("d_in must be one of {4096, 8192, 16384}")
        if d_out not in VALID_D_OUT:
            raise ValueError("d_out must be one of {258, 4096, 8192, 16384}")
        self.d_in = d_in
        self.d_out = d_out
        self.layer = layer
        self.matrix_id = matrix_id
        self.deterministic = deterministic and wpack is None and alpha is None

        if wpack is None or alpha is None:
            if not deterministic:
                raise ValueError("wpack and alpha are required when deterministic=False")
            ext = _load_extension()
            if torch.device(device).type == "cuda":
                wpack, alpha = ext.deterministic_pack_ternary_cuda(d_out, d_in, layer, matrix_id)
                wpack = wpack.to(device=device, non_blocking=True)
                alpha = alpha.to(device=device, non_blocking=True)
            else:
                wpack_cpu, alpha_cpu = ext.deterministic_pack_ternary(d_out, d_in, layer, matrix_id)
                wpack = wpack_cpu.to(device=device, non_blocking=True)
                alpha = alpha_cpu.to(device=device, non_blocking=True)
        else:
            wpack = wpack.to(device=device, dtype=torch.uint8, non_blocking=True).contiguous()
            alpha = alpha.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()

        if tuple(wpack.shape) != (d_out, s5(d_in)):
            raise ValueError(f"wpack must have shape {(d_out, s5(d_in))}")
        if tuple(alpha.shape) != (d_out,):
            raise ValueError(f"alpha must have shape {(d_out,)}")

        self.register_buffer("wpack", wpack, persistent=True)
        self.register_buffer("alpha", alpha, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_in:
            raise ValueError(f"expected last dim {self.d_in}, got {x.shape[-1]}")
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.d_in).contiguous()
        y2 = _TLinearFn.apply(x2, self.wpack, self.alpha)
        return y2.reshape(*leading, self.d_out)


class TernaryEmbedding(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device | str = "cuda",
        wpack: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        if wpack is None or alpha is None:
            if not deterministic:
                raise ValueError("wpack and alpha are required when deterministic=False")
            ext = _load_extension()
            if torch.device(device).type == "cuda":
                wpack, alpha = ext.deterministic_pack_ternary_cuda(258, 4096, 0, 11)
                wpack = wpack.to(device=device, non_blocking=True)
                alpha = alpha.to(device=device, non_blocking=True)
            else:
                wpack_cpu, alpha_cpu = ext.deterministic_pack_ternary(258, 4096, 0, 11)
                wpack = wpack_cpu.to(device=device, non_blocking=True)
                alpha = alpha_cpu.to(device=device, non_blocking=True)
        else:
            wpack = wpack.to(device=device, dtype=torch.uint8, non_blocking=True).contiguous()
            alpha = alpha.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()

        self.register_buffer("wpack", wpack, persistent=True)
        self.register_buffer("alpha", alpha, persistent=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() == 2:
            if token_ids.shape[0] != 1:
                raise ValueError("batch size must be exactly 1")
            token_ids = token_ids[0]
        if token_ids.dim() != 1:
            raise ValueError("token_ids must have shape [T] or [1, T]")
        if token_ids.numel() < 1 or token_ids.numel() > 8192:
            raise ValueError("T must be in [1, 8192]")
        if token_ids.max().item() > 257 or token_ids.min().item() < 0:
            raise ValueError("token ids must be in [0, 257]")
        ids = token_ids.to(device=self.wpack.device, dtype=torch.uint16, non_blocking=True).contiguous()
        return _load_extension().embedding_forward(ids, self.wpack, self.alpha)


class SilexLayer(nn.Module):
    def __init__(self, layer_idx: int, config: SilexConfig, device: torch.device | str, deterministic: bool) -> None:
        super().__init__()
        d = config.d_model
        d_ff = config.d_ff
        self.config = config
        self.w_i = TLinear(d, d, layer=layer_idx + 1, matrix_id=0, device=device, deterministic=deterministic)
        self.w_f = TLinear(d, d, layer=layer_idx + 1, matrix_id=1, device=device, deterministic=deterministic)
        self.w_v = TLinear(d, d, layer=layer_idx + 1, matrix_id=2, device=device, deterministic=deterministic)
        self.w_r = TLinear(d, d, layer=layer_idx + 1, matrix_id=3, device=device, deterministic=deterministic)
        self.w_o = TLinear(d, d, layer=layer_idx + 1, matrix_id=4, device=device, deterministic=deterministic)
        self.w_a = TLinear(d, d_ff, layer=layer_idx + 1, matrix_id=5, device=device, deterministic=deterministic)
        self.w_b = TLinear(d, d_ff, layer=layer_idx + 1, matrix_id=6, device=device, deterministic=deterministic)
        self.w_c = TLinear(d_ff, d, layer=layer_idx + 1, matrix_id=7, device=device, deterministic=deterministic)

        self.register_buffer("gamma_m", torch.ones(d, dtype=torch.bfloat16, device=device), persistent=True)
        self.register_buffer("gamma_f", torch.ones(d, dtype=torch.bfloat16, device=device), persistent=True)
        self.register_buffer("lambda_raw", torch.zeros(config.recurrent_slots, d, dtype=torch.bfloat16, device=device), persistent=True)
        self.register_buffer("beta_raw", torch.zeros(config.recurrent_slots, d, dtype=torch.bfloat16, device=device), persistent=True)

        self.A_m = nn.Parameter(_plastic_a(layer_idx, "m", torch.device(device)))
        self.B_m = nn.Parameter(torch.zeros(d, config.plastic_rank, dtype=torch.float32, device=device))
        self.A_f = nn.Parameter(_plastic_a(layer_idx, "f", torch.device(device)))
        self.B_f = nn.Parameter(torch.zeros(d, config.plastic_rank, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        u = rms_norm(x, self.gamma_m, cfg.eps_norm)
        i_gate = sigmoid_bf16_to_fp32(self.w_i(u))
        f_gate = sigmoid_bf16_to_fp32(self.w_f(u))
        v_val = silu_bf16_to_fp32(self.w_v(u))
        r_gate = sigmoid_bf16_to_fp32(self.w_r(u))

        g, new_state = _load_extension().recurrent_mixer_forward(
            i_gate.contiguous(),
            f_gate.contiguous(),
            v_val.contiguous(),
            r_gate.contiguous(),
            state.contiguous(),
            self.lambda_raw,
            self.beta_raw,
        )
        p_m = low_rank_adapter(u, self.A_m, self.B_m)
        x_tilde = residual_add(x, self.w_o(g), p_m, cfg.rho)

        u_f = rms_norm(x_tilde, self.gamma_f, cfg.eps_norm)
        h = gated_silu_product(self.w_a(u_f), self.w_b(u_f))
        p_f = low_rank_adapter(u_f, self.A_f, self.B_f)
        y = residual_add(x_tilde, self.w_c(h), p_f, cfg.rho)
        return y, new_state


class LatentReasoner(nn.Module):
    def __init__(self, config: SilexConfig, device: torch.device | str, deterministic: bool) -> None:
        super().__init__()
        self.config = config
        self.w_z1 = TLinear(config.d_model, config.d_z, layer=0, matrix_id=8, device=device, deterministic=deterministic)
        self.w_z2 = TLinear(config.d_model, config.d_z, layer=0, matrix_id=9, device=device, deterministic=deterministic)
        self.w_z3 = TLinear(config.d_z, config.d_model, layer=0, matrix_id=10, device=device, deterministic=deterministic)
        self.register_buffer("gamma_z", torch.ones(config.d_model, dtype=torch.bfloat16, device=device), persistent=True)

    def forward(self, z: torch.Tensor, k: int) -> list[torch.Tensor]:
        if k < 0 or k > self.config.k_max:
            raise ValueError("latent depth k must be in [0, K_max]")
        zs = [z]
        cur = z
        for _ in range(k):
            n = rms_norm(cur, self.gamma_z, self.config.eps_norm)
            q = gated_silu_product(self.w_z1(n), self.w_z2(n))
            zero_adapter = torch.empty_like(cur)
            zero_adapter.zero_()
            cur = residual_add(cur, self.w_z3(q), zero_adapter, self.config.rho_z)
            zs.append(cur)
        return zs


class SilexCodeT18_6B_R64(nn.Module):
    name = "SilexCode-T18.6B-R64"

    def __init__(
        self,
        config: SilexConfig = SILEX_T18_6B_R64_CONFIG,
        *,
        device: torch.device | str = "cuda",
        deterministic: bool = True,
        enable_output_adapter: bool = False,
        output_adapter_rank: int = 64,
    ) -> None:
        super().__init__()
        if config.vocab_size != 258:
            raise ValueError("V must be exactly 258")
        if output_adapter_rank < 1:
            raise ValueError("output_adapter_rank must be positive")
        self.config = config
        self.embedding = TernaryEmbedding(device=device, deterministic=deterministic)
        self.layers = nn.ModuleList(
            [SilexLayer(i, config, device, deterministic) for i in range(config.layers)]
        )
        self.reasoner = LatentReasoner(config, device, deterministic)
        self.register_buffer("gamma_out", torch.ones(config.d_model, dtype=torch.bfloat16, device=device), persistent=True)
        self.output_adapter_enabled = bool(enable_output_adapter)
        self.output_adapter_rank = int(output_adapter_rank)
        if self.output_adapter_enabled:
            self.output_adapter_down = nn.Parameter(
                _output_adapter_down(self.output_adapter_rank, config.d_model, torch.device(device))
            )
            self.output_adapter_up = nn.Parameter(
                torch.zeros(config.vocab_size, self.output_adapter_rank, dtype=torch.float32, device=device)
            )

        for p in self.embedding.parameters():
            p.requires_grad_(False)
        self._freeze_non_plastic()
        self.use_native_runtime = True
        self.deterministic_backbone = bool(deterministic)

    def _freeze_non_plastic(self) -> None:
        for name, param in self.named_parameters():
            is_plastic = name.endswith(("A_m", "B_m", "A_f", "B_f"))
            is_output_adapter = self.output_adapter_enabled and name in {
                "output_adapter_down",
                "output_adapter_up",
            }
            param.requires_grad_(is_plastic or is_output_adapter)

    def output_adapter_parameters(self) -> list[nn.Parameter]:
        if not self.output_adapter_enabled:
            return []
        return [self.output_adapter_down, self.output_adapter_up]

    def freeze_internal_plastic_adapters(self) -> None:
        for name, param in self.named_parameters():
            if name.endswith(("A_m", "B_m", "A_f", "B_f")):
                param.requires_grad_(False)

    def initial_state(self) -> torch.Tensor:
        cfg = self.config
        return torch.zeros(
            cfg.layers,
            cfg.recurrent_slots,
            cfg.d_model,
            dtype=torch.bfloat16,
            device=self.gamma_out.device,
        )

    def backbone(self, token_ids: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.initial_state()
        x = self.embedding(token_ids)
        states = []
        for layer_idx, layer in enumerate(self.layers):
            x, s = layer(x, state[layer_idx])
            states.append(s)
        return x, torch.stack(states, dim=0)

    def logits_from_latents(self, zs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        logits = []
        for z in zs:
            o = rms_norm(z, self.gamma_out, self.config.eps_norm)
            base_logits = _TLinearFn.apply(o.contiguous(), self.embedding.wpack, self.embedding.alpha).float()
            if self.output_adapter_enabled:
                hidden = torch.nn.functional.linear(o.float(), self.output_adapter_down)
                base_logits = base_logits + torch.nn.functional.linear(hidden, self.output_adapter_up)
            logits.append(base_logits)
        return logits

    def _native_args(self):
        layer_wpacks = []
        layer_alphas = []
        gamma_m = []
        gamma_f = []
        lambda_raw = []
        beta_raw = []
        A_m = []
        B_m = []
        A_f = []
        B_f = []
        for layer in self.layers:
            for proj in (layer.w_i, layer.w_f, layer.w_v, layer.w_r, layer.w_o, layer.w_a, layer.w_b, layer.w_c):
                layer_wpacks.append(proj.wpack)
                layer_alphas.append(proj.alpha)
            gamma_m.append(layer.gamma_m)
            gamma_f.append(layer.gamma_f)
            lambda_raw.append(layer.lambda_raw)
            beta_raw.append(layer.beta_raw)
            A_m.append(layer.A_m)
            B_m.append(layer.B_m)
            A_f.append(layer.A_f)
            B_f.append(layer.B_f)
        z_wpacks = [self.reasoner.w_z1.wpack, self.reasoner.w_z2.wpack, self.reasoner.w_z3.wpack]
        z_alphas = [self.reasoner.w_z1.alpha, self.reasoner.w_z2.alpha, self.reasoner.w_z3.alpha]
        return (
            self.embedding.wpack,
            self.embedding.alpha,
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
            self.reasoner.gamma_z,
            self.gamma_out,
        )

    def mark_checkpoint_backbone_loaded(self) -> None:
        self.deterministic_backbone = False
        self.use_native_runtime = True
        for layer in self.layers:
            for proj in (layer.w_i, layer.w_f, layer.w_v, layer.w_r, layer.w_o, layer.w_a, layer.w_b, layer.w_c):
                proj.deterministic = False
        self.reasoner.w_z1.deterministic = False
        self.reasoner.w_z2.deterministic = False
        self.reasoner.w_z3.deterministic = False

    def _require_deterministic_native_runtime(self) -> None:
        if not self.deterministic_backbone:
            return

    def _native_kfac_args(self, kfac_optimizer):
        a_covs = []
        g_covs = []
        a_invs = []
        g_invs = []
        for layer_idx in range(self.config.layers):
            for suffix in ("A_m", "B_m", "A_f", "B_f"):
                name = f"layers.{layer_idx}.{suffix}"
                if name not in kfac_optimizer.state:
                    raise ValueError(f"missing K-FAC state for {name}")
                st = kfac_optimizer.state[name]
                a_covs.append(st.a_cov)
                g_covs.append(st.g_cov)
                a_invs.append(st.a_inv)
                g_invs.append(st.g_inv)
        return a_covs, g_covs, a_invs, g_invs

    def forward_python_reference(
        self,
        token_ids: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        k: int | None = None,
        return_all_depths: bool = False,
    ):
        if k is None:
            k = self.config.k_train if self.training else self.config.k_infer
        h, new_state = self.backbone(token_ids, state)
        zs = self.reasoner(h, k)
        logits = self.logits_from_latents(zs if return_all_depths else [zs[-1]])
        return logits if return_all_depths else logits[-1], new_state

    def forward_native(
        self,
        token_ids: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        k: int | None = None,
        return_all_depths: bool = False,
    ):
        if k is None:
            k = self.config.k_train if self.training else self.config.k_infer
        if state is None:
            state = self.initial_state()
        if token_ids.dim() == 2:
            if token_ids.shape[0] != 1:
                raise ValueError("batch size must be exactly 1")
            token_ids = token_ids[0]
        if token_ids.dim() != 1:
            raise ValueError("token_ids must have shape [T] or [1,T]")
        if token_ids.numel() < 1 or token_ids.numel() > self.config.s_max:
            raise ValueError("T must be in [1,8192]")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("token ids must be in [0,257]")
        ids = token_ids.to(device=self.gamma_out.device, dtype=torch.uint16, non_blocking=True).contiguous()
        args = self._native_args()
        logits, new_state, logits_by_depth = _load_extension().silex_forward_cuda(
            ids,
            state.contiguous(),
            *args,
            int(k),
            bool(return_all_depths),
            bool(self.deterministic_backbone),
        )
        return (logits_by_depth if return_all_depths else logits), new_state

    def train_workspace_bytes(self) -> int:
        return int(_load_extension().silex_train_workspace_bytes(self.config.u_train))

    def train_workspace_layout(self) -> dict[str, int]:
        return {
            str(k): int(v)
            for k, v in _load_extension().silex_train_workspace_layout(self.config.u_train).items()
        }

    def allocate_train_workspace(self) -> torch.Tensor:
        return torch.empty(
            self.train_workspace_bytes(),
            dtype=torch.uint8,
            device=self.gamma_out.device,
        )

    def train_chunk_cuda(
        self,
        token_ids_512: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        stage: int = 3,
        kfac_optimizer=None,
        active_layers: list[int] | None = None,
        eta: float | None = None,
        damping: float | None = None,
        trust_region_delta: float | None = None,
        teacher_logits_final: torch.Tensor | None = None,
    ):
        if state is None:
            state = self.initial_state()
        if workspace is None:
            workspace = self.allocate_train_workspace()
        if token_ids_512.dim() == 2:
            if token_ids_512.shape != (1, self.config.u_train):
                raise ValueError("token_ids_512 must have shape [512] or [1,512]")
            token_ids_512 = token_ids_512[0]
        if token_ids_512.dim() != 1 or token_ids_512.numel() != self.config.u_train:
            raise ValueError("token_ids_512 must contain exactly 512 tokens")
        token_check = token_ids_512.to(dtype=torch.int64)
        if token_check.min().item() < 0 or token_check.max().item() >= self.config.vocab_size:
            raise ValueError("token ids must be in [0,257]")
        ids = token_ids_512.to(device=self.gamma_out.device, dtype=torch.uint16, non_blocking=True).contiguous()

        if kfac_optimizer is not None:
            if labels is None:
                labels = token_ids_512[1:].to(device=self.gamma_out.device, dtype=torch.long)
            else:
                labels = labels.to(device=self.gamma_out.device, dtype=torch.long, non_blocking=True).contiguous()
            if labels.dim() != 1 or labels.numel() != self.config.u_train - 1:
                raise ValueError("labels must have shape [511]")

            if loss_mask is None:
                loss_mask = torch.ones(self.config.u_train - 1, device=self.gamma_out.device, dtype=torch.float32)
            else:
                loss_mask = loss_mask.to(device=self.gamma_out.device, dtype=torch.float32, non_blocking=True).contiguous()
            if loss_mask.dim() != 1 or loss_mask.numel() != self.config.u_train - 1:
                raise ValueError("loss_mask must have shape [511]")

            if teacher_logits_final is None:
                teacher_logits_final = torch.empty(0, device=self.gamma_out.device, dtype=torch.float32)
            else:
                teacher_logits_final = teacher_logits_final.to(device=self.gamma_out.device, dtype=torch.float32, non_blocking=True).contiguous()

            kfac_a_covs, kfac_g_covs, kfac_a_invs, kfac_g_invs = self._native_kfac_args(kfac_optimizer)
            active = list(active_layers if active_layers is not None else sorted(kfac_optimizer.active_layers or range(1, self.config.layers + 1)))
            result = _load_extension().silex_train_chunk_cuda(
                ids,
                state.contiguous(),
                workspace.contiguous(),
                labels.contiguous(),
                loss_mask.contiguous(),
                teacher_logits_final,
                *self._native_args(),
                bool(self.deterministic_backbone),
                kfac_a_covs,
                kfac_g_covs,
                kfac_a_invs,
                kfac_g_invs,
                active,
                int(stage),
                float(kfac_optimizer.lr if eta is None else eta),
                float(kfac_optimizer.damping if damping is None else damping),
                float(kfac_optimizer.trust_region if trust_region_delta is None else trust_region_delta),
                float(kfac_optimizer.ema),
                float(kfac_optimizer.weight_decay),
                float(self.config.eps_opt),
            )
            new_state = result["new_state"]
            metrics = {str(k): v for k, v in result.items() if str(k) != "new_state"}
            return metrics, new_state

        logits, new_state, logits_by_depth = _load_extension().silex_train_chunk_cuda(
            ids,
            state.contiguous(),
            workspace.contiguous(),
            *self._native_args(),
            bool(self.deterministic_backbone),
        )
        return torch.stack([x.float() for x in logits_by_depth], dim=0), new_state

    def forward_train(self, input_ids: torch.Tensor, k_train: int = 4, return_logits_by_depth: bool = True):
        if k_train != self.config.k_train:
            raise ValueError("k_train must be exactly 4")
        logits, _ = self.forward_python_reference(input_ids, k=k_train, return_all_depths=return_logits_by_depth)
        if return_logits_by_depth:
            return torch.stack([(x[:-1] if x.shape[0] == self.config.u_train else x).float() for x in logits], dim=0)
        return (logits[:-1] if logits.shape[0] == self.config.u_train else logits).float()

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        k: int | None = None,
        return_all_depths: bool = False,
    ):
        if self.use_native_runtime and not torch.is_grad_enabled() and not self.output_adapter_enabled:
            return self.forward_native(token_ids, state=state, k=k, return_all_depths=return_all_depths)
        return self.forward_python_reference(token_ids, state=state, k=k, return_all_depths=return_all_depths)

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        if len(prompt_ids) < 1 or len(prompt_ids) > self.config.s_max:
            raise ValueError("prompt length must be in [1, 8192]")
        out = list(prompt_ids)
        state = self.initial_state()
        ids = torch.tensor(prompt_ids, dtype=torch.int64, device=self.gamma_out.device).unsqueeze(0)
        logits, state = self.forward(ids, state=state, k=self.config.k_infer)
        next_id = int(torch.argmax(logits[-1], dim=-1).item())
        for _ in range(max_new_tokens):
            out.append(next_id)
            if next_id == TOKEN_EOS:
                break
            ids = torch.tensor([[next_id]], dtype=torch.int64, device=self.gamma_out.device)
            logits, state = self.forward(ids, state=state, k=self.config.k_infer)
            next_id = int(torch.argmax(logits[-1], dim=-1).item())
        return out

    @torch.no_grad()
    def generate_bytes(
        self,
        prompt_ids: list[int],
        max_new_bytes: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        stop_bytes: bytes,
    ) -> bytes:
        if len(prompt_ids) < 1 or len(prompt_ids) > self.config.s_max:
            raise ValueError("prompt length must be in [1, 8192]")
        if max_new_bytes < 0:
            raise ValueError("max_new_bytes must be non-negative")
        if any((int(x) < 0 or int(x) >= self.config.vocab_size) for x in prompt_ids):
            raise ValueError("prompt token ids must be in [0,257]")
        if not isinstance(stop_bytes, bytes):
            raise ValueError("stop_bytes must be bytes")

        device = self.gamma_out.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) & ((1 << 64) - 1))

        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        logits, state = self.forward_native(prompt, state=self.initial_state(), k=self.config.k_infer)
        generated = bytearray()
        token = sample_token_top_k_top_p(
            logits[-1],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )

        sampled_steps = 0
        max_sampled_steps = max(1, max_new_bytes * 2 + 16)
        while len(generated) < max_new_bytes and sampled_steps < max_sampled_steps:
            sampled_steps += 1
            if token == TOKEN_EOS:
                break
            if token <= 255:
                generated.append(token)
                if stop_bytes and bytes(generated).endswith(stop_bytes):
                    break
            elif token != TOKEN_BOS:
                raise ValueError(f"generated invalid token id: {token}")

            step_ids = torch.tensor([token], dtype=torch.long, device=device)
            logits, state = self.forward_native(step_ids, state=state, k=self.config.k_infer)
            token = sample_token_top_k_top_p(
                logits[-1],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )

        return bytes(generated)
