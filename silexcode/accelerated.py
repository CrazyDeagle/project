from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset import GLOBAL_SEED, RNG, encode_ascii_record, encode_ascii_record_without_eos, generate_record
from .train import (
    ADVANCE_CONSECUTIVE_EVALS,
    EVAL_EVERY_UPDATES,
    K_TRAIN,
    MAX_UPDATES,
    SEQ_LEN,
    STAGE_CONFIG,
    VAL_SIZE_PER_STAGE,
    build_ssd_pool,
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
) -> PackedChunk:
    if seq_len != SEQ_LEN:
        raise ValueError("accelerated native training requires SEQ_LEN=512")
    if not records:
        raise ValueError("at least one record is required")

    ids: list[int] = []
    loss_mask = [0] * (seq_len - 1)
    used_indices: list[int] = []
    target_tokens = 0

    for record in records:
        prefix, target = _prefix_and_target(record, stage)
        segment = encode_ascii_record(prefix + target)
        if len(segment) > seq_len:
            continue
        if len(ids) + len(segment) > seq_len:
            break

        base = len(ids)
        prefix_ids = encode_ascii_record_without_eos(prefix)
        target_start = base + len(prefix_ids) - 1
        target_end_exclusive = base + len(segment) - 1

        ids.extend(segment)
        used_indices.append(int(record["index"]))
        for pos in range(target_start, min(target_end_exclusive, seq_len - 1)):
            loss_mask[pos] = 1
            target_tokens += 1

    if not used_indices:
        raise ValueError("NO_RECORD_FITS_PACKED_CHUNK")

    real_len = len(ids)
    if real_len < seq_len:
        for pos in range(max(0, real_len - 1), seq_len - 1):
            loss_mask[pos] = 1
            target_tokens += 1
    ids = ids + [257] * (seq_len - real_len)
    return PackedChunk(
        token_ids=ids,
        labels=ids[1:],
        loss_mask=loss_mask,
        record_indices=used_indices,
        target_tokens=target_tokens,
    )


def generate_packed_chunk(stage: int, start_index: int, *, max_records: int = 8) -> PackedChunk:
    records = [generate_record(stage, start_index + i) for i in range(max_records)]
    return build_packed_sequence_and_mask(records, stage)


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
    require_thresholds: bool = True,
    generate_eval_outputs: bool | None = None,
    enable_ssd: bool | None = None,
):
    if not hasattr(model, "train_chunk_cuda"):
        raise ValueError("accelerated curriculum requires native train_chunk_cuda")

    torch.manual_seed(123456789)
    torch.cuda.manual_seed_all(123456789)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    val_size = VAL_SIZE_PER_STAGE if val_size_override is None else int(val_size_override)
    eval_every = EVAL_EVERY_UPDATES if eval_every_updates_override is None else int(eval_every_updates_override)
    max_updates = MAX_UPDATES if max_updates_override is None else {
        stage: int(max_updates_override.get(stage, MAX_UPDATES[stage])) for stage in (1, 2, 3)
    }
    use_ssd_stage3 = enable_ssd if enable_ssd is not None else require_thresholds
    do_generate_eval = require_thresholds if generate_eval_outputs is None else bool(generate_eval_outputs)
    validation_indices = {
        1: [10_000_000 + i for i in range(val_size)],
        2: [20_000_000 + i for i in range(val_size)],
        3: [30_000_000 + i for i in range(val_size)],
    }

    workspace = model.allocate_train_workspace()
    global_update = 0
    teacher_cache = None
    ssd_pool: list[dict] = []

    for stage in (1, 2, 3):
        cfg = STAGE_CONFIG[stage]
        if hasattr(kfac_optimizer, "reset_curvature"):
            kfac_optimizer.reset_curvature(active_layers=cfg["active_layers"], damping=cfg["damping"])
        if hasattr(kfac_optimizer, "set_hyperparams"):
            kfac_optimizer.set_hyperparams(eta=cfg["eta"], damping=cfg["damping"], trust_region_delta=cfg["delta"])

        consecutive_ready = 0
        record_cursor = stage * 1_000_000_000

        if stage == 3:
            teacher_cache_path = str(Path(output_dir) / "teacher_stage3_logits")
            precompute_stage3_teacher_cache(model, validation_indices[3], teacher_cache_path)
            teacher_cache = open_teacher_cache_reader(teacher_cache_path)
            if use_ssd_stage3:
                ssd_pool = build_ssd_pool(model, [40_000_000 + i for i in range(256)], global_update)

        for local_update in range(max_updates[stage]):
            use_ssd = False
            if stage == 3 and ssd_pool:
                use_ssd = RNG(GLOBAL_SEED ^ global_update ^ 0xACCE1A7E).randint(0, 99) < 30

            if use_ssd:
                base = global_update % len(ssd_pool)
                records = [ssd_pool[(base + i) % len(ssd_pool)] for i in range(min(max_records_per_chunk, len(ssd_pool)))]
                chunk = build_packed_sequence_and_mask(records, stage)
            else:
                chunk = generate_packed_chunk(stage, record_cursor, max_records=max_records_per_chunk)
                record_cursor += max(1, len(chunk.record_indices))

            token_ids_t = torch.tensor(chunk.token_ids, device="cuda", dtype=torch.long)
            labels_t = torch.tensor(chunk.labels, device="cuda", dtype=torch.long)
            mask_t = torch.tensor(chunk.loss_mask, device="cuda", dtype=torch.float32)

            teacher_logits = None
            if stage == 3 and teacher_cache is not None and len(chunk.record_indices) == 1:
                cached = teacher_cache.lookup(chunk.record_indices[0])
                if cached is not None:
                    teacher_logits = cached.to("cuda", dtype=torch.float32)

            metrics, _state = model.train_chunk_cuda(
                token_ids_t,
                workspace=workspace,
                labels=labels_t,
                loss_mask=mask_t,
                stage=stage,
                kfac_optimizer=kfac_optimizer,
                active_layers=cfg["active_layers"],
                eta=cfg["eta"],
                damping=cfg["damping"],
                trust_region_delta=cfg["delta"],
                teacher_logits_final=teacher_logits,
            )

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
                        "train": {k: float(v) for k, v in metrics.items() if k != "new_state"},
                        "validation": val_metrics,
                        "packed_records": len(chunk.record_indices),
                        "target_tokens": chunk.target_tokens,
                        "consecutive_ready": consecutive_ready,
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

    return model
