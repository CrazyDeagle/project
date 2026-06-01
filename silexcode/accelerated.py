from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import export_plastic_checkpoint
from .dataset import (
    GLOBAL_SEED,
    RNG,
    encode_ascii_record,
    encode_ascii_record_without_eos,
    generate_record,
)
from .train import (
    ADVANCE_CONSECUTIVE_EVALS,
    EVAL_EVERY_UPDATES,
    K_TRAIN,
    MAX_UPDATES,
    SEQ_LEN,
    STAGE_CONFIG,
    VAL_SIZE_PER_STAGE,
    build_ssd_pool,
    compute_depth_losses,
    evaluate_stage,
    extract_code_body_with_closing_C,
    extract_trace_input_line,
    extract_trace_lines_with_closing_R,
    open_teacher_cache_reader,
    precompute_stage3_teacher_cache,
    stage_ready,
)


@dataclass(frozen=True)
class PackedChunk:
    token_ids: list[int]
    labels: list[int]
    loss_mask: list[int]
    record_indices: list[int]
    family_ids: list[int]
    target_tokens: int


def _prefix_and_target(record: dict, stage: int) -> tuple[str, str]:
    if stage == 1:
        return "<S1>\n" + record["R"] + "<C>\n", extract_code_body_with_closing_C(record["C"])
    if stage == 2:
        return (
            "<S2>\n" + record["P"] + record["C"] + "<R>\n" + extract_trace_input_line(record["R"]),
            extract_trace_lines_with_closing_R(record["R"]),
        )
    if stage == 3:
        return "<S3>\n" + record["P"] + "<C>\n", extract_code_body_with_closing_C(record["C"])
    raise ValueError("INVALID_STAGE")


def build_packed_sequence_and_mask(
    records: list[dict],
    stage: int,
    *,
    seq_len: int = SEQ_LEN,
    include_padding_loss: bool = False,
    packing: str = "shortest",
) -> PackedChunk:
    if seq_len != SEQ_LEN:
        raise ValueError("accelerated native training requires SEQ_LEN=512")
    if not records:
        raise ValueError("at least one record is required")

    ids: list[int] = []
    loss_mask = [0] * (seq_len - 1)
    used_indices: list[int] = []
    family_ids: list[int] = []
    target_tokens = 0

    prepared: list[tuple[int, dict, list[int], int, int]] = []
    for order, record in enumerate(records):
        prefix, target = _prefix_and_target(record, stage)
        segment = encode_ascii_record(prefix + target)
        if len(segment) > seq_len:
            continue
        prefix_ids = encode_ascii_record_without_eos(prefix)
        prepared.append((len(segment), record, segment, len(prefix_ids), order))

    if packing == "shortest":
        ordered = sorted(prepared, key=lambda x: (x[0], x[4]))
    elif packing == "balanced":
        groups: dict[int, list[tuple[int, dict, list[int], int, int]]] = {}
        for item in prepared:
            groups.setdefault(int(item[1]["family_id"]), []).append(item)
        for family in groups:
            groups[family].sort(key=lambda x: (x[0], x[4]))
        ordered = []
        while any(groups.values()):
            for family in sorted(groups):
                if groups[family]:
                    ordered.append(groups[family].pop(0))
    elif packing == "random-fit":
        ordered = sorted(
            prepared,
            key=lambda x: (
                (GLOBAL_SEED ^ (int(x[1]["index"]) * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1),
                x[4],
            ),
        )
    else:
        raise ValueError("packing must be one of: shortest, balanced, random-fit")

    for _segment_len, record, segment, prefix_len, _order in ordered:
        if len(ids) + len(segment) > seq_len:
            continue

        base = len(ids)
        target_start = base + prefix_len - 1
        target_end_exclusive = base + len(segment) - 1

        ids.extend(segment)
        used_indices.append(int(record["index"]))
        family_ids.append(int(record["family_id"]))
        for pos in range(target_start, min(target_end_exclusive, seq_len - 1)):
            loss_mask[pos] = 1
            target_tokens += 1

    if not used_indices:
        raise ValueError("NO_RECORD_FITS_PACKED_CHUNK")

    real_len = len(ids)
    if include_padding_loss and real_len < seq_len:
        for pos in range(max(0, real_len - 1), seq_len - 1):
            loss_mask[pos] = 1
            target_tokens += 1
    ids = ids + [257] * (seq_len - real_len)
    return PackedChunk(
        token_ids=ids,
        labels=ids[1:],
        loss_mask=loss_mask,
        record_indices=used_indices,
        family_ids=family_ids,
        target_tokens=target_tokens,
    )


def generate_packed_chunk(
    stage: int,
    start_index: int,
    *,
    max_records: int = 8,
    candidate_multiplier: int = 4,
    include_padding_loss: bool = False,
    packing: str = "shortest",
) -> PackedChunk:
    candidate_count = max_records * max(1, candidate_multiplier)
    records = [generate_record(stage, start_index + i) for i in range(candidate_count)]
    return build_packed_sequence_and_mask(
        records,
        stage,
        include_padding_loss=include_padding_loss,
        packing=packing,
    )


def _save_jsonl(output_dir: str, row: dict) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(output_dir) / "accelerated_metrics.jsonl").open("a", encoding="ascii") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def train_accelerated_curriculum(
    model,
    kfac_optimizer,
    output_dir: str,
    *,
    max_updates_override: dict[int, int] | None = None,
    eval_every_updates_override: int | None = None,
    val_size_override: int | None = None,
    max_records_per_chunk: int = 8,
    candidate_multiplier: int = 4,
    include_padding_loss: bool = False,
    packing: str = "shortest",
    kfac_warmup_updates: int = 0,
    eta_scale: float = 1.0,
    damping_scale: float = 1.0,
    trust_scale: float = 1.0,
    native_optimizer: str = "kfac",
    checkpoint_every_evals: int = 0,
    include_kfac_in_checkpoints: bool = False,
    require_thresholds: bool = True,
    generate_eval_outputs: bool | None = None,
    enable_ssd: bool | None = None,
    stages: tuple[int, ...] = (1, 2, 3),
):
    if not hasattr(model, "train_chunk_cuda"):
        raise ValueError("accelerated curriculum requires native train_chunk_cuda")

    torch.manual_seed(123456789)
    torch.cuda.manual_seed_all(123456789)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    val_size = VAL_SIZE_PER_STAGE if val_size_override is None else int(val_size_override)
    eval_every = (
        EVAL_EVERY_UPDATES
        if eval_every_updates_override is None
        else int(eval_every_updates_override)
    )
    max_updates = (
        MAX_UPDATES
        if max_updates_override is None
        else {
            stage: int(max_updates_override.get(stage, MAX_UPDATES[stage])) for stage in (1, 2, 3)
        }
    )
    use_ssd_stage3 = enable_ssd if enable_ssd is not None else require_thresholds
    do_generate_eval = (
        require_thresholds if generate_eval_outputs is None else bool(generate_eval_outputs)
    )
    validation_indices = {
        1: [10_000_000 + i for i in range(val_size)],
        2: [20_000_000 + i for i in range(val_size)],
        3: [30_000_000 + i for i in range(val_size)],
    }

    workspace = model.allocate_train_workspace()
    global_update = 0
    teacher_cache = None
    ssd_pool: list[dict] = []

    for stage in stages:
        cfg = STAGE_CONFIG[stage]
        stage_eta = float(cfg["eta"]) * float(eta_scale)
        stage_damping = float(cfg["damping"]) * float(damping_scale)
        stage_delta = float(cfg["delta"]) * float(trust_scale)
        if hasattr(kfac_optimizer, "reset_curvature"):
            kfac_optimizer.reset_curvature(
                active_layers=cfg["active_layers"], damping=stage_damping
            )
        if hasattr(kfac_optimizer, "set_hyperparams"):
            kfac_optimizer.set_hyperparams(
                eta=stage_eta, damping=stage_damping, trust_region_delta=stage_delta
            )

        consecutive_ready = 0
        record_cursor = stage * 1_000_000_000
        eval_count = 0

        if stage == 3:
            teacher_cache_path = str(Path(output_dir) / "teacher_stage3_logits")
            precompute_stage3_teacher_cache(model, validation_indices[3], teacher_cache_path)
            teacher_cache = open_teacher_cache_reader(teacher_cache_path)
            if use_ssd_stage3:
                ssd_pool = build_ssd_pool(
                    model, [40_000_000 + i for i in range(256)], global_update
                )

        for local_update in range(max_updates[stage]):
            use_ssd = False
            if stage == 3 and ssd_pool:
                use_ssd = RNG(GLOBAL_SEED ^ global_update ^ 0xACCE1A7E).randint(0, 99) < 30

            if use_ssd:
                base = global_update % len(ssd_pool)
                records = [
                    ssd_pool[(base + i) % len(ssd_pool)]
                    for i in range(min(max_records_per_chunk, len(ssd_pool)))
                ]
                chunk = build_packed_sequence_and_mask(
                    records,
                    stage,
                    include_padding_loss=include_padding_loss,
                    packing=packing,
                )
            else:
                chunk = generate_packed_chunk(
                    stage,
                    record_cursor,
                    max_records=max_records_per_chunk,
                    candidate_multiplier=candidate_multiplier,
                    include_padding_loss=include_padding_loss,
                    packing=packing,
                )
                record_cursor += max(1, max_records_per_chunk * max(1, candidate_multiplier))

            token_ids_t = torch.tensor(chunk.token_ids, device="cuda", dtype=torch.long)
            labels_t = torch.tensor(chunk.labels, device="cuda", dtype=torch.long)
            mask_t = torch.tensor(chunk.loss_mask, device="cuda", dtype=torch.float32)

            teacher_logits = None
            if stage == 3 and teacher_cache is not None and len(chunk.record_indices) == 1:
                cached = teacher_cache.lookup(chunk.record_indices[0])
                if cached is not None:
                    teacher_logits = cached.to("cuda", dtype=torch.float32)

            step_start = time.perf_counter()
            effective_eta = 0.0 if local_update < int(kfac_warmup_updates) else stage_eta
            metrics, _state = model.train_chunk_cuda(
                token_ids_t,
                workspace=workspace,
                labels=labels_t,
                loss_mask=mask_t,
                stage=stage,
                kfac_optimizer=kfac_optimizer,
                active_layers=cfg["active_layers"],
                eta=effective_eta,
                damping=stage_damping,
                trust_region_delta=stage_delta,
                teacher_logits_final=teacher_logits,
                native_optimizer=native_optimizer,
            )
            step_seconds = time.perf_counter() - step_start

            global_update += 1
            if local_update % eval_every == 0:
                val_metrics = evaluate_stage(
                    model,
                    stage,
                    validation_indices[stage],
                    teacher_cache,
                    generate_outputs=do_generate_eval,
                )
                ready = stage_ready(stage, val_metrics) if require_thresholds else False
                consecutive_ready = consecutive_ready + 1 if ready else 0
                _save_jsonl(
                    output_dir,
                    {
                        "stage": stage,
                        "global_update": global_update,
                        "local_update": local_update,
                        "train": {
                            **{k: float(v) for k, v in metrics.items() if k != "new_state"},
                            "step_seconds": float(step_seconds),
                            "updates_per_minute": float(60.0 / max(step_seconds, 1.0e-9)),
                            "eta": float(effective_eta),
                            "damping": float(stage_damping),
                            "trust_region_delta": float(stage_delta),
                            "kfac_warmup_active": float(local_update < int(kfac_warmup_updates)),
                            "max_memory_allocated_mb": float(
                                torch.cuda.max_memory_allocated() / (1024**2)
                            ),
                        },
                        "validation": val_metrics,
                        "packed_records": len(chunk.record_indices),
                        "family_ids": chunk.family_ids,
                        "target_tokens": chunk.target_tokens,
                        "target_fraction": chunk.target_tokens / float(SEQ_LEN - 1),
                        "consecutive_ready": consecutive_ready,
                    },
                )
                eval_count += 1
                if checkpoint_every_evals > 0 and eval_count % checkpoint_every_evals == 0:
                    ckpt = Path(output_dir) / f"stage_{stage}_update_{global_update}.plastic.silex"
                    export_plastic_checkpoint(
                        model,
                        ckpt,
                        kfac_optimizer=kfac_optimizer,
                        include_kfac=include_kfac_in_checkpoints,
                        metadata={
                            "stage": stage,
                            "global_update": global_update,
                            "local_update": local_update,
                            "include_kfac": include_kfac_in_checkpoints,
                        },
                    )
                if require_thresholds and consecutive_ready >= ADVANCE_CONSECUTIVE_EVALS:
                    break

        if consecutive_ready < ADVANCE_CONSECUTIVE_EVALS:
            if require_thresholds:
                raise RuntimeError(f"ACCELERATED_CURRICULUM_STAGE_{stage}_FAILED_THRESHOLDS")
            _save_jsonl(
                output_dir,
                {
                    "stage": stage,
                    "global_update": global_update,
                    "local_update": max_updates[stage],
                    "dry_run_complete": True,
                    "consecutive_ready": consecutive_ready,
                },
            )

    export_plastic_checkpoint(
        model,
        Path(output_dir) / "accelerated_latest.plastic.silex",
        kfac_optimizer=kfac_optimizer,
        include_kfac=include_kfac_in_checkpoints,
        metadata={
            "global_update": global_update,
            "stages": list(stages),
            "optimizer": native_optimizer,
            "include_kfac": include_kfac_in_checkpoints,
        },
    )
    return model


def train_output_adapter_curriculum(
    model,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    *,
    stages: tuple[int, ...] = (1,),
    max_updates_override: dict[int, int] | None = None,
    eval_every_updates_override: int | None = None,
    val_size_override: int | None = None,
    max_records_per_chunk: int = 8,
    candidate_multiplier: int = 4,
    include_padding_loss: bool = False,
    packing: str = "shortest",
    checkpoint_every_evals: int = 0,
    require_thresholds: bool = False,
    generate_eval_outputs: bool = False,
):
    if not bool(getattr(model, "output_adapter_enabled", False)):
        raise ValueError("output adapter curriculum requires enable_output_adapter=True")

    torch.manual_seed(123456789)
    torch.cuda.manual_seed_all(123456789)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    val_size = VAL_SIZE_PER_STAGE if val_size_override is None else int(val_size_override)
    eval_every = (
        EVAL_EVERY_UPDATES
        if eval_every_updates_override is None
        else int(eval_every_updates_override)
    )
    max_updates = (
        MAX_UPDATES
        if max_updates_override is None
        else {
            stage: int(max_updates_override.get(stage, MAX_UPDATES[stage])) for stage in (1, 2, 3)
        }
    )
    validation_indices = {
        1: [10_000_000 + i for i in range(val_size)],
        2: [20_000_000 + i for i in range(val_size)],
        3: [30_000_000 + i for i in range(val_size)],
    }

    global_update = 0
    for stage in stages:
        record_cursor = stage * 1_000_000_000
        eval_count = 0
        consecutive_ready = 0
        for local_update in range(max_updates[stage]):
            chunk = generate_packed_chunk(
                stage,
                record_cursor,
                max_records=max_records_per_chunk,
                candidate_multiplier=candidate_multiplier,
                include_padding_loss=include_padding_loss,
                packing=packing,
            )
            record_cursor += max(1, max_records_per_chunk * max(1, candidate_multiplier))
            input_ids_t = torch.tensor(chunk.token_ids[:-1], device="cuda", dtype=torch.long)
            labels_t = torch.tensor(chunk.labels, device="cuda", dtype=torch.long)
            mask_t = torch.tensor(chunk.loss_mask, device="cuda", dtype=torch.float32)

            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits_by_k = model.forward_train(
                input_ids=input_ids_t, k_train=K_TRAIN, return_logits_by_depth=True
            )
            depth = compute_depth_losses(logits_by_k, labels_t, mask_t)
            loss = depth["nll"] + 0.10 * depth["mono"]
            loss.backward()
            optimizer.step()
            step_seconds = time.perf_counter() - step_start
            global_update += 1

            if local_update % eval_every == 0:
                val_metrics = evaluate_stage(
                    model,
                    stage,
                    validation_indices[stage],
                    teacher_cache=None,
                    generate_outputs=generate_eval_outputs,
                )
                ready = stage_ready(stage, val_metrics) if require_thresholds else False
                consecutive_ready = consecutive_ready + 1 if ready else 0
                _save_jsonl(
                    output_dir,
                    {
                        "stage": stage,
                        "global_update": global_update,
                        "local_update": local_update,
                        "train": {
                            "loss": float(loss.detach().cpu()),
                            "nll": float(depth["nll"].detach().cpu()),
                            "mono": float(depth["mono"].detach().cpu()),
                            "nll4": float(depth["nll_by_k"][4].detach().cpu()),
                            "latent_gain": float(depth["latent_gain"].detach().cpu()),
                            "step_seconds": float(step_seconds),
                            "updates_per_minute": float(60.0 / max(step_seconds, 1.0e-9)),
                            "max_memory_allocated_mb": float(
                                torch.cuda.max_memory_allocated() / (1024**2)
                            ),
                        },
                        "validation": val_metrics,
                        "packed_records": len(chunk.record_indices),
                        "family_ids": chunk.family_ids,
                        "target_tokens": chunk.target_tokens,
                        "target_fraction": chunk.target_tokens / float(SEQ_LEN - 1),
                        "consecutive_ready": consecutive_ready,
                    },
                )
                eval_count += 1
                if checkpoint_every_evals > 0 and eval_count % checkpoint_every_evals == 0:
                    export_plastic_checkpoint(
                        model,
                        Path(output_dir) / f"stage_{stage}_update_{global_update}.plastic.silex",
                        kfac_optimizer=None,
                        include_kfac=False,
                        metadata={
                            "stage": stage,
                            "global_update": global_update,
                            "local_update": local_update,
                            "optimizer": "output_adapter_adamw",
                        },
                    )
                if require_thresholds and consecutive_ready >= ADVANCE_CONSECUTIVE_EVALS:
                    break

        if require_thresholds and consecutive_ready < ADVANCE_CONSECUTIVE_EVALS:
            raise RuntimeError(f"OUTPUT_ADAPTER_CURRICULUM_STAGE_{stage}_FAILED_THRESHOLDS")

    export_plastic_checkpoint(
        model,
        Path(output_dir) / "output_adapter_latest.plastic.silex",
        metadata={
            "global_update": global_update,
            "stages": list(stages),
            "optimizer": "output_adapter_adamw",
        },
    )
    return model
