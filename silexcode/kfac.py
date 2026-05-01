from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import SILEX_T18_6B_R64_CONFIG, SilexConfig


@dataclass
class KFACBlockState:
    a_cov: torch.Tensor
    g_cov: torch.Tensor
    a_inv: torch.Tensor
    g_inv: torch.Tensor


class BlockKFACOptimizer:
    def __init__(
        self,
        named_parameters,
        *,
        config: SilexConfig = SILEX_T18_6B_R64_CONFIG,
        lr: float = 0.1,
        damping: float = 1.0e-3,
        ema: float = 0.05,
        weight_decay: float = 1.0e-5,
        trust_region: float = 1.0e-3,
    ) -> None:
        self.config = config
        self.lr = lr
        self.damping = damping
        self.ema = ema
        self.weight_decay = weight_decay
        self.trust_region = trust_region
        self.active_layers: set[int] | None = None
        self.params: list[tuple[str, torch.nn.Parameter]] = [
            (name, p) for name, p in named_parameters if p.requires_grad
        ]
        if not self.params:
            raise ValueError("BlockKFACOptimizer requires trainable plastic parameters")
        self.state: dict[str, KFACBlockState] = {}
        for name, p in self.params:
            self._validate_param(name, p)
            q = config.q_kfac
            o, i = p.shape
            device = p.device
            eye_a = torch.eye(q, dtype=torch.float32, device=device).expand(i // q, q, q).clone()
            eye_g = torch.eye(q, dtype=torch.float32, device=device).expand(o // q, q, q).clone()
            self.state[name] = KFACBlockState(
                a_cov=eye_a.clone(),
                g_cov=eye_g.clone(),
                a_inv=eye_a.clone(),
                g_inv=eye_g.clone(),
            )

    def set_hyperparams(
        self,
        *,
        eta: float | None = None,
        damping: float | None = None,
        trust_region_delta: float | None = None,
    ) -> None:
        if eta is not None:
            self.lr = float(eta)
        if damping is not None:
            self.damping = float(damping)
        if trust_region_delta is not None:
            self.trust_region = float(trust_region_delta)

    def _layer_number(self, name: str) -> int | None:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
            return int(parts[1]) + 1
        return None

    def _is_active(self, name: str) -> bool:
        if self.active_layers is None:
            return True
        layer = self._layer_number(name)
        return layer in self.active_layers

    @torch.no_grad()
    def reset_curvature(self, *, active_layers: list[int] | None = None, damping: float | None = None) -> None:
        if damping is not None:
            self.damping = float(damping)
        self.active_layers = set(active_layers) if active_layers is not None else None
        q = self.config.q_kfac
        for name, p in self.params:
            st = self.state[name]
            eye = torch.eye(q, dtype=torch.float32, device=p.device)
            st.a_cov.copy_(eye.expand_as(st.a_cov))
            st.g_cov.copy_(eye.expand_as(st.g_cov))
            inv = torch.linalg.inv(eye + self.damping * eye)
            st.a_inv.copy_(inv.expand_as(st.a_inv))
            st.g_inv.copy_(inv.expand_as(st.g_inv))

    def _validate_param(self, name: str, p: torch.Tensor) -> None:
        if p.dtype is not torch.float32:
            raise ValueError(f"{name} must be fp32")
        if p.dim() != 2:
            raise ValueError(f"{name} must be a matrix")
        if tuple(p.shape) not in {(64, 4096), (4096, 64)}:
            raise ValueError(f"{name} must have shape (64,4096) or (4096,64)")

    @torch.no_grad()
    def update_curvature(self, name: str, inputs: torch.Tensor, grad_outputs: torch.Tensor) -> None:
        if name not in self.state:
            raise KeyError(name)
        p = dict(self.params)[name]
        o, i = p.shape
        q = self.config.q_kfac
        if inputs.shape[-1] != i or grad_outputs.shape[-1] != o:
            raise ValueError(f"curvature tensors do not match {name} shape")
        a = inputs.reshape(-1, i).float()
        g = grad_outputs.reshape(-1, o).float()
        if a.shape[0] == 0:
            raise ValueError("at least one effective token is required")
        a_blocks = a.reshape(a.shape[0], i // q, q)
        g_blocks = g.reshape(g.shape[0], o // q, q)
        a_hat = torch.einsum("nbq,nbp->bqp", a_blocks, a_blocks) / float(a.shape[0])
        g_hat = torch.einsum("ncq,ncp->cqp", g_blocks, g_blocks) / float(g.shape[0])
        st = self.state[name]
        st.a_cov.mul_(1.0 - self.ema).add_(a_hat, alpha=self.ema)
        st.g_cov.mul_(1.0 - self.ema).add_(g_hat, alpha=self.ema)
        eye = torch.eye(q, dtype=torch.float32, device=p.device)
        st.a_inv.copy_(torch.linalg.inv(st.a_cov + self.damping * eye))
        st.g_inv.copy_(torch.linalg.inv(st.g_cov + self.damping * eye))

    @torch.no_grad()
    def step(
        self,
        *,
        active_layers: list[int] | None = None,
        eta: float | None = None,
        damping: float | None = None,
        trust_region_delta: float | None = None,
    ) -> float:
        if active_layers is not None:
            self.active_layers = set(active_layers)
        self.set_hyperparams(eta=eta, damping=damping, trust_region_delta=trust_region_delta)
        q = self.config.q_kfac
        nu = torch.zeros((), dtype=torch.float32, device=self.params[0][1].device)
        updates: list[tuple[torch.nn.Parameter, torch.Tensor]] = []

        for name, p in self.params:
            if not self._is_active(name):
                continue
            if p.grad is None:
                continue
            grad = p.grad.float() + self.weight_decay * p.float()
            st = self.state[name]
            o, i = p.shape
            grad_blocks = grad.reshape(o // q, q, i // q, q).permute(0, 2, 1, 3).contiguous()
            d_blocks = torch.empty_like(grad_blocks)
            for c in range(o // q):
                left = st.g_inv[c]
                for b in range(i // q):
                    d_blocks[c, b] = torch.einsum("xy,yz,zw->xw", left, grad_blocks[c, b], st.a_inv[b])
            nat = d_blocks.permute(0, 2, 1, 3).contiguous().reshape_as(p)
            nu = nu + (grad * nat).sum()
            updates.append((p, nat))

        nu = torch.clamp(nu, min=0.0)
        chi = torch.minimum(
            torch.ones_like(nu),
            torch.sqrt(torch.tensor(self.trust_region, dtype=torch.float32, device=nu.device) / (nu + self.config.eps_opt)),
        )
        for p, nat in updates:
            p.add_(nat, alpha=-self.lr * float(chi.item()))
        return float(nu.item())

    def zero_grad(self) -> None:
        for _, p in self.params:
            p.grad = None
