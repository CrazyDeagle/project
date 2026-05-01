from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from .constants import SILEX_T18_6B_R64_CONFIG, SilexConfig


def latent_depth_weights(k_train: int, *, device: torch.device) -> torch.Tensor:
    k = torch.arange(k_train + 1, device=device, dtype=torch.float32)
    return 2.0 * (k + 1.0) / float((k_train + 1) * (k_train + 2))


def plastic_mdl(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    total = None
    count = 0
    denom = 2.0**-10
    inv_log2 = 1.0 / math.log(2.0)
    for p in parameters:
        if not p.requires_grad:
            continue
        value = torch.log1p(p.float().abs() / denom).sum() * inv_log2
        total = value if total is None else total + value
        count += p.numel()
    if total is None or count == 0:
        raise ValueError("no trainable plastic parameters were provided")
    return total / float(count)


def silex_latent_loss(
    logits_by_depth: list[torch.Tensor],
    token_ids: torch.Tensor,
    plastic_parameters: Iterable[torch.nn.Parameter],
    config: SilexConfig = SILEX_T18_6B_R64_CONFIG,
) -> dict[str, torch.Tensor]:
    if len(logits_by_depth) != config.k_train + 1:
        raise ValueError(f"expected {config.k_train + 1} latent depths")
    if token_ids.dim() == 2:
        if token_ids.shape[0] != 1:
            raise ValueError("batch size must be exactly 1")
        token_ids = token_ids[0]
    if token_ids.dim() != 1 or token_ids.numel() != config.u_train:
        raise ValueError(f"training chunk must have exactly {config.u_train} tokens")
    if token_ids.min().item() < 0 or token_ids.max().item() >= config.vocab_size:
        raise ValueError("token ids must be in [0, 257]")

    targets = token_ids[1:].to(device=logits_by_depth[0].device, dtype=torch.long)
    ces = []
    log_probs = []
    probs = []

    for logits in logits_by_depth:
        if tuple(logits.shape) != (config.u_train, config.vocab_size):
            raise ValueError(f"logits must have shape {(config.u_train, config.vocab_size)}")
        effective = logits[:-1].float()
        ces.append(F.cross_entropy(effective, targets, reduction="none"))
        lp = F.log_softmax(effective, dim=-1)
        log_probs.append(lp)
        probs.append(lp.exp())

    ce_stack = torch.stack(ces, dim=0)
    weights = latent_depth_weights(config.k_train, device=ce_stack.device)
    nll = (weights[:, None] * ce_stack).sum(dim=0).mean()

    mono_terms = torch.clamp(ce_stack[1:] - ce_stack[:-1], min=0.0).square()
    mono = mono_terms.mean()

    final_p = probs[-1].detach()
    final_logp = log_probs[-1].detach()
    kl_terms = []
    for k in range(config.k_train):
        kl_terms.append((final_p * (final_logp - log_probs[k])).sum(dim=-1))
    kl = torch.stack(kl_terms, dim=0).mean()

    mdl = plastic_mdl(plastic_parameters)
    total = nll + 0.10 * mono + 0.05 * kl + 1.0e-6 * mdl
    return {
        "loss": total,
        "nll": nll,
        "mono": mono,
        "kl": kl,
        "mdl": mdl,
    }
