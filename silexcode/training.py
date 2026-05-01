from __future__ import annotations

import torch

from .constants import SILEX_T18_6B_R64_CONFIG, SilexConfig
from .kfac import BlockKFACOptimizer
from .losses import silex_latent_loss
from .model import SilexCodeT18_6B_R64


def plastic_named_parameters(model: SilexCodeT18_6B_R64):
    for name, p in model.named_parameters():
        if p.requires_grad:
            yield name, p


def count_trainable_parameters(model: SilexCodeT18_6B_R64) -> int:
    return sum(p.numel() for _, p in plastic_named_parameters(model))


def train_chunk(
    model: SilexCodeT18_6B_R64,
    optimizer: BlockKFACOptimizer,
    token_ids_512: torch.Tensor,
    *,
    config: SilexConfig = SILEX_T18_6B_R64_CONFIG,
) -> dict[str, float]:
    if token_ids_512.dim() == 2:
        if token_ids_512.shape != (1, config.u_train):
            raise ValueError(f"token chunk must have shape (1, {config.u_train})")
    elif token_ids_512.dim() == 1:
        if token_ids_512.numel() != config.u_train:
            raise ValueError(f"token chunk must have exactly {config.u_train} tokens")
    else:
        raise ValueError("token chunk must have shape [512] or [1,512]")

    model.train()
    optimizer.zero_grad()
    logits_by_depth, _ = model(token_ids_512, state=model.initial_state(), k=config.k_train, return_all_depths=True)
    parts = silex_latent_loss(logits_by_depth, token_ids_512, (p for _, p in plastic_named_parameters(model)), config)
    parts["loss"].backward()
    nu = optimizer.step()
    return {name: float(value.detach().item()) for name, value in parts.items()} | {"natural_norm": nu}
