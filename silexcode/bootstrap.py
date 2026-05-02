from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import export_plastic_checkpoint
from .dataset import GLOBAL_SEED, RNG, assert_ascii, encode_ascii_record, encode_ascii_record_without_eos
from .train import SEQ_LEN, compute_depth_losses, compute_token_diagnostics, model_forward_train


BOOTSTRAP_STAGE = 0
BOOTSTRAP_LEVELS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class BootstrapChunk:
    token_ids: list[int]
    labels: list[int]
    loss_mask: list[int]
    indices: list[int]
    levels: list[int]
    target_tokens: int


def _bootstrap_code(level: int, rng: RNG) -> str:
    if level == 0:
        return "def f(a,b,t):\n    return a\n"
    if level == 1:
        exprs = ["a+b", "a-b", "b-a", "a+t", "b+t", "a*b"]
        return "def f(a,b,t):\n    return " + rng.choice(exprs) + "\n"
    if level == 2:
        true_expr = rng.choice(["a+b", "a-b", "a+t"])
        false_expr = rng.choice(["b-a", "b+t", "a*b"])
        cond = rng.choice(["a<b", "a<=t", "a+b<t"])
        return (
            "def f(a,b,t):\n"
            f"    if {cond}:\n"
            f"        return {true_expr}\n"
            f"    return {false_expr}\n"
        )
    if level == 3:
        expr = rng.choice(["x[i]", "x[i]+1", "x[i]-1", "x[i]+i"])
        return (
            "def f(x,n,t):\n"
            "    y=[0]*n\n"
            "    for i in range(n):\n"
            f"        y[i]={expr}\n"
            "    return y\n"
        )
    if level == 4:
        pred = rng.choice(["1", "x[i]<t", "x[i]>=0", "i%2==0"])
        expr = rng.choice(["x[i]", "x[i]+1", "abs(x[i])"])
        return (
            "def f(x,n,t):\n"
            "    r=0\n"
            "    for i in range(n):\n"
            f"        if {pred}:\n"
            f"            r=r+{expr}\n"
            "    return r\n"
        )
    raise ValueError("INVALID_BOOTSTRAP_LEVEL")


def generate_bootstrap_record(level: int, index: int) -> dict:
    if level not in BOOTSTRAP_LEVELS:
        raise ValueError("INVALID_BOOTSTRAP_LEVEL")
    rng = RNG(GLOBAL_SEED ^ 0xB00757A9E0000000 ^ (level << 48) ^ index)
    code = _bootstrap_code(level, rng)
    assert_ascii(code)
    C = "<C>\n" + code + "</C>\n"
    prefix = f"<B{level}>\n<C>\n"
    target = code + "</C>\n"
    text = prefix + target
    ids = encode_ascii_record(text)
    if len(ids) > SEQ_LEN:
        raise ValueError("BOOTSTRAP_RECORD_TOO_LONG")
    return {
        "stage": BOOTSTRAP_STAGE,
        "level": level,
        "index": index,
        "family_id": 100 + level,
        "C": C,
        "prefix": prefix,
        "target": target,
        "token_ids": ids,
    }


def _record_segment(record: dict) -> tuple[list[int], int]:
    segment = encode_ascii_record(record["prefix"] + record["target"])
    prefix_len = len(encode_ascii_record_without_eos(record["prefix"]))
    return segment, prefix_len


def build_bootstrap_chunk(
    records: list[dict],
    *,
    seq_len: int = SEQ_LEN,
    include_padding_loss: bool = False,
) -> BootstrapChunk:
    if seq_len != SEQ_LEN:
        raise ValueError("bootstrap native training requires SEQ_LEN=512")
    if not records:
        raise ValueError("at least one record is required")
    prepared = []
    for order, record in enumerate(records):
        segment, prefix_len = _record_segment(record)
        if len(segment) <= seq_len:
            prepared.append((len(segment), order, record, segment, prefix_len))
    ids: list[int] = []
    loss_mask = [0] * (seq_len - 1)
    used_indices: list[int] = []
    levels: list[int] = []
    target_tokens = 0

    for _length, _order, record, segment, prefix_len in sorted(prepared, key=lambda x: (x[0], x[1])):
        if len(ids) + len(segment) > seq_len:
            continue
        base = len(ids)
        target_start = base + prefix_len - 1
        target_end_exclusive = base + len(segment) - 1
        ids.extend(segment)
        used_indices.append(int(record["index"]))
        levels.append(int(record["level"]))
        for pos in range(target_start, min(target_end_exclusive, seq_len - 1)):
            loss_mask[pos] = 1
            target_tokens += 1

    if not used_indices:
        raise ValueError("NO_BOOTSTRAP_RECORD_FITS")
    real_len = len(ids)
    if include_padding_loss and real_len < seq_len:
        for pos in range(max(0, real_len - 1), seq_len - 1):
            loss_mask[pos] = 1
            target_tokens += 1
    ids = ids + [257] * (seq_len - real_len)
    return BootstrapChunk(
        token_ids=ids,
        labels=ids[1:],
        loss_mask=loss_mask,
        indices=used_indices,
        levels=levels,
        target_tokens=target_tokens,
    )


def generate_bootstrap_chunk(
    level: int,
    start_index: int,
    *,
    max_records: int = 16,
    candidate_multiplier: int = 4,
    include_padding_loss: bool = False,
) -> BootstrapChunk:
    records = [
        generate_bootstrap_record(level, start_index + i)
        for i in range(max_records * max(1, candidate_multiplier))
    ]
    return build_bootstrap_chunk(records, include_padding_loss=include_padding_loss)


def _save_jsonl(output_dir: str, row: dict) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(output_dir) / "bootstrap_metrics.jsonl").open("a", encoding="ascii") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate_bootstrap(model, level: int, validation_indices: list[int]) -> dict[str, float]:
    nll4_sum = mono_sum = gain_sum = 0.0
    diag_sums: dict[str, float] = {}
    diag_counts: dict[str, int] = {}
    count = 0
    for idx in validation_indices:
        chunk = build_bootstrap_chunk([generate_bootstrap_record(level, idx)])
        input_ids = torch.tensor(chunk.token_ids[:-1], device="cuda", dtype=torch.long)
        labels = torch.tensor(chunk.labels, device="cuda", dtype=torch.long)
        mask = torch.tensor(chunk.loss_mask, device="cuda", dtype=torch.float32)
        logits_by_k = model_forward_train(model, input_ids)
        depth = compute_depth_losses(logits_by_k, labels, mask)
        diag = compute_token_diagnostics(logits_by_k, labels, mask)
        nll4_sum += float(depth["nll_by_k"][4].detach().cpu())
        mono_sum += float(depth["mono"].detach().cpu())
        gain_sum += float(depth["latent_gain"].detach().cpu())
        for key, value in diag.items():
            diag_sums[key] = diag_sums.get(key, 0.0) + float(value)
            diag_counts[key] = diag_counts.get(key, 0) + 1
        count += 1
    metrics = {
        "nll4": nll4_sum / count,
        "mono": mono_sum / count,
        "latent_gain": gain_sum / count,
    }
    for key, value in diag_sums.items():
        metrics[key] = value / max(1, diag_counts[key])
    return metrics


def train_bootstrap(
    model,
    kfac_optimizer,
    output_dir: str,
    *,
    levels: tuple[int, ...] = BOOTSTRAP_LEVELS,
    updates_per_level: int = 1000,
    eval_every: int = 100,
    val_size: int = 16,
    max_records_per_chunk: int = 16,
    candidate_multiplier: int = 4,
    eta: float = 0.01,
    damping: float = 1.0e-2,
    trust_region_delta: float = 3.0e-5,
    kfac_warmup_updates: int = 100,
    checkpoint_every_evals: int = 0,
    include_kfac_in_checkpoints: bool = False,
):
    torch.manual_seed(123456789)
    torch.cuda.manual_seed_all(123456789)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    workspace = model.allocate_train_workspace()
    global_update = 0
    active_layers = list(range(1, model.config.layers + 1))
    if hasattr(kfac_optimizer, "reset_curvature"):
        kfac_optimizer.reset_curvature(active_layers=active_layers, damping=damping)
    if hasattr(kfac_optimizer, "set_hyperparams"):
        kfac_optimizer.set_hyperparams(eta=eta, damping=damping, trust_region_delta=trust_region_delta)

    for level in levels:
        record_cursor = 100_000_000 + level * 10_000_000
        eval_count = 0
        validation_indices = [900_000_000 + level * 100_000 + i for i in range(val_size)]
        for local_update in range(updates_per_level):
            chunk = generate_bootstrap_chunk(
                level,
                record_cursor,
                max_records=max_records_per_chunk,
                candidate_multiplier=candidate_multiplier,
            )
            record_cursor += max_records_per_chunk * max(1, candidate_multiplier)
            token_ids = torch.tensor(chunk.token_ids, device="cuda", dtype=torch.long)
            labels = torch.tensor(chunk.labels, device="cuda", dtype=torch.long)
            mask = torch.tensor(chunk.loss_mask, device="cuda", dtype=torch.float32)
            effective_eta = 0.0 if local_update < kfac_warmup_updates else eta
            step_start = time.perf_counter()
            train_metrics, _state = model.train_chunk_cuda(
                token_ids,
                workspace=workspace,
                labels=labels,
                loss_mask=mask,
                stage=1,
                kfac_optimizer=kfac_optimizer,
                active_layers=active_layers,
                eta=effective_eta,
                damping=damping,
                trust_region_delta=trust_region_delta,
            )
            step_seconds = time.perf_counter() - step_start
            global_update += 1

            if local_update % eval_every == 0:
                val_metrics = evaluate_bootstrap(model, level, validation_indices)
                row = {
                    "bootstrap_level": level,
                    "global_update": global_update,
                    "local_update": local_update,
                    "indices": chunk.indices,
                    "levels": chunk.levels,
                    "target_tokens": chunk.target_tokens,
                    "target_fraction": chunk.target_tokens / float(SEQ_LEN - 1),
                    "train": {
                        **{k: float(v) for k, v in train_metrics.items() if k != "new_state"},
                        "step_seconds": float(step_seconds),
                        "updates_per_minute": float(60.0 / max(step_seconds, 1.0e-9)),
                        "eta": float(effective_eta),
                        "damping": float(damping),
                        "trust_region_delta": float(trust_region_delta),
                        "kfac_warmup_active": float(local_update < kfac_warmup_updates),
                        "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
                    },
                    "validation": val_metrics,
                }
                _save_jsonl(output_dir, row)
                eval_count += 1
                if checkpoint_every_evals > 0 and eval_count % checkpoint_every_evals == 0:
                    export_plastic_checkpoint(
                        model,
                        Path(output_dir) / f"bootstrap_level_{level}_update_{global_update}.plastic.silex",
                        kfac_optimizer=kfac_optimizer,
                        include_kfac=include_kfac_in_checkpoints,
                        metadata={
                            "bootstrap_level": level,
                            "global_update": global_update,
                            "local_update": local_update,
                        },
                    )

    export_plastic_checkpoint(
        model,
        Path(output_dir) / "bootstrap_latest.plastic.silex",
        kfac_optimizer=kfac_optimizer,
        include_kfac=include_kfac_in_checkpoints,
        metadata={"global_update": global_update, "levels": list(levels)},
    )
    return model


def train_bootstrap_output_adapter(
    model,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    *,
    levels: tuple[int, ...] = BOOTSTRAP_LEVELS,
    updates_per_level: int = 1000,
    eval_every: int = 100,
    val_size: int = 16,
    max_records_per_chunk: int = 16,
    candidate_multiplier: int = 4,
    checkpoint_every_evals: int = 0,
):
    if not getattr(model, "output_adapter_enabled", False):
        raise ValueError("output adapter bootstrap requires enable_output_adapter=True")

    torch.manual_seed(123456789)
    torch.cuda.manual_seed_all(123456789)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    global_update = 0
    for level in levels:
        record_cursor = 100_000_000 + level * 10_000_000
        eval_count = 0
        validation_indices = [900_000_000 + level * 100_000 + i for i in range(val_size)]
        for local_update in range(updates_per_level):
            chunk = generate_bootstrap_chunk(
                level,
                record_cursor,
                max_records=max_records_per_chunk,
                candidate_multiplier=candidate_multiplier,
            )
            record_cursor += max_records_per_chunk * max(1, candidate_multiplier)
            input_ids = torch.tensor(chunk.token_ids[:-1], device="cuda", dtype=torch.long)
            labels = torch.tensor(chunk.labels, device="cuda", dtype=torch.long)
            mask = torch.tensor(chunk.loss_mask, device="cuda", dtype=torch.float32)

            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits_by_k = model_forward_train(model, input_ids)
            depth = compute_depth_losses(logits_by_k, labels, mask)
            loss = depth["nll"] + 0.10 * depth["mono"]
            loss.backward()
            optimizer.step()
            step_seconds = time.perf_counter() - step_start
            global_update += 1

            if local_update % eval_every == 0:
                val_metrics = evaluate_bootstrap(model, level, validation_indices)
                row = {
                    "bootstrap_level": level,
                    "global_update": global_update,
                    "local_update": local_update,
                    "indices": chunk.indices,
                    "levels": chunk.levels,
                    "target_tokens": chunk.target_tokens,
                    "target_fraction": chunk.target_tokens / float(SEQ_LEN - 1),
                    "train": {
                        "loss": float(loss.detach().cpu()),
                        "nll": float(depth["nll"].detach().cpu()),
                        "mono": float(depth["mono"].detach().cpu()),
                        "nll4": float(depth["nll_by_k"][4].detach().cpu()),
                        "latent_gain": float(depth["latent_gain"].detach().cpu()),
                        "step_seconds": float(step_seconds),
                        "updates_per_minute": float(60.0 / max(step_seconds, 1.0e-9)),
                        "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
                    },
                    "validation": val_metrics,
                }
                _save_jsonl(output_dir, row)
                eval_count += 1
                if checkpoint_every_evals > 0 and eval_count % checkpoint_every_evals == 0:
                    export_plastic_checkpoint(
                        model,
                        Path(output_dir) / f"bootstrap_level_{level}_update_{global_update}.plastic.silex",
                        kfac_optimizer=None,
                        include_kfac=False,
                        metadata={
                            "bootstrap_level": level,
                            "global_update": global_update,
                            "local_update": local_update,
                            "output_adapter": True,
                        },
                    )

    export_plastic_checkpoint(
        model,
        Path(output_dir) / "bootstrap_latest.plastic.silex",
        kfac_optimizer=None,
        include_kfac=False,
        metadata={"global_update": global_update, "levels": list(levels), "output_adapter": True},
    )
    return model
