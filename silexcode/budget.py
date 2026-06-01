from __future__ import annotations

from dataclasses import dataclass

from .constants import SILEX_T18_6B_R64_CONFIG, SilexConfig, s5

MIB = 2**20


@dataclass(frozen=True)
class MemoryBudget:
    ternary_pack_mb: float
    meta_mb: float
    weights_mb: float
    recurrent_state_mb: float
    plastic_mb: float
    activations_mb: float
    total_mb: float
    free_mb: float


def plastic_parameter_count(config: SilexConfig = SILEX_T18_6B_R64_CONFIG) -> int:
    return 2 * config.layers * (2 * config.d_model * config.plastic_rank)


def logical_ternary_weight_count(config: SilexConfig = SILEX_T18_6B_R64_CONFIG) -> int:
    per_layer = 5 * config.d_model**2 + 3 * config.d_model * config.d_ff
    backbone = config.layers * per_layer
    embedding = config.vocab_size * config.d_model
    latent = 3 * config.d_model * config.d_z
    return backbone + embedding + latent


def physical_ternary_pack_bytes(config: SilexConfig = SILEX_T18_6B_R64_CONFIG) -> int:
    d = config.d_model
    d_ff = config.d_ff
    d_z = config.d_z
    backbone = config.layers * (5 * d * s5(d) + 2 * d_ff * s5(d) + d * s5(d_ff))
    latent = 2 * d_z * s5(d) + d * s5(d_z)
    embedding = config.vocab_size * s5(d)
    return backbone + latent + embedding


def memory_budget(config: SilexConfig = SILEX_T18_6B_R64_CONFIG) -> MemoryBudget:
    ternary_pack_mb = physical_ternary_pack_bytes(config) / MIB
    n_scale = config.layers * (5 * config.d_model + 2 * config.d_ff + config.d_model)
    n_scale += config.vocab_size + (2 * config.d_z + config.d_model)
    n_rms = (2 * config.layers + 1) * config.d_model
    n_rec = 2 * config.layers * config.recurrent_slots * config.d_model
    meta_mb = 2 * (n_scale + n_rms + n_rec) / MIB
    weights_mb = ternary_pack_mb + meta_mb
    recurrent_state_mb = config.layers * config.recurrent_slots * config.d_model * 2 / MIB
    n_p = plastic_parameter_count(config)
    plastic_mb = 3 * n_p * 4 / MIB + (4 * config.layers) * 2.03125
    logits_mb = config.u_train * config.vocab_size * 4 / MIB
    activations_mb = 260.0 + 2048.0 + 20.0 + 48.0 + 24.0 + logits_mb
    total_mb = weights_mb + recurrent_state_mb + plastic_mb + activations_mb
    return MemoryBudget(
        ternary_pack_mb=ternary_pack_mb,
        meta_mb=meta_mb,
        weights_mb=weights_mb,
        recurrent_state_mb=recurrent_state_mb,
        plastic_mb=plastic_mb,
        activations_mb=activations_mb,
        total_mb=total_mb,
        free_mb=8192.0 - total_mb,
    )


def assert_tdd_invariants(config: SilexConfig = SILEX_T18_6B_R64_CONFIG) -> None:
    if config.vocab_size != 258:
        raise AssertionError("V must be exactly 258")
    if (s5(4096), s5(8192), s5(16384)) != (832, 1664, 3296):
        raise AssertionError("S5 row strides do not match the TDD")
    if plastic_parameter_count(config) != 67_108_864:
        raise AssertionError("plastic trainable parameter count must be 67,108,864")
    if logical_ternary_weight_count(config) != 18_355_331_072:
        raise AssertionError("logical ternary count for V=258 must be 18,355,331,072")
    budget = memory_budget(config)
    if abs(budget.total_mb - 7256.25598526001) > 1.0e-6:
        raise AssertionError(f"unexpected memory budget: {budget.total_mb}")
    if budget.total_mb >= 8192.0:
        raise AssertionError("memory budget exceeds 8192 MiB")
