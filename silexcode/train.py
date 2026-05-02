from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .dataset import (
    GLOBAL_SEED,
    RNG,
    compile_restricted_function,
    encode_ascii_record,
    encode_ascii_record_without_eos,
    extract_code_between_C_tags,
    generate_record,
    make_cases,
    run_reference_interpreter,
    run_restricted,
    serialize_value,
    verify_candidate_code,
)
from .losses import plastic_mdl
from .tokenizer import ByteLevelTokenizer
from .training import plastic_named_parameters

BATCH_SIZE = 1
SEQ_LEN = 512
K_TRAIN = 4
VOCAB_SIZE = 258
EVAL_EVERY_UPDATES = 2048
VAL_SIZE_PER_STAGE = 4096
MAX_UPDATES = {1: 200_000, 2: 300_000, 3: 500_000}
ADVANCE_CONSECUTIVE_EVALS = 3

SSD_ENABLED_STAGE = 3
SSD_REFRESH_UPDATES = 16384
SSD_PROBLEMS_PER_REFRESH = 4096
SSD_CANDIDATES_PER_PROBLEM = 12
SSD_MAX_ACCEPTED_PER_PROBLEM = 2
SSD_MAX_CODE_BYTES = 384
SSD_CONSENSUS_MIN = 2

TEMPERATURE_CYCLE = [0.35, 0.55, 0.75, 0.95]
TOP_P_CYCLE = [0.80, 0.90, 0.95]
TOP_K = 32

STAGE_CONFIG = {
    1: {"eta": 0.080, "damping": 1e-3, "delta": 1e-3, "active_layers": list(range(1, 17)), "nll_threshold": 0.080, "mono_threshold": 0.0010, "latent_gain_threshold": 0.010},
    2: {"eta": 0.060, "damping": 1e-3, "delta": 1e-3, "active_layers": list(range(1, 49)), "nll_threshold": 0.120, "mono_threshold": 0.0015, "latent_gain_threshold": 0.020},
    3: {"eta": 0.040, "damping": 3e-4, "delta": 5e-4, "active_layers": list(range(1, 65)), "nll_threshold": 0.180, "mono_threshold": 0.0020, "latent_gain_threshold": 0.040},
}


def extract_code_body_with_closing_C(C: str) -> str:
    if not C.startswith("<C>\n") or not C.endswith("</C>\n"):
        raise ValueError("INVALID_C_BLOCK")
    return C[len("<C>\n"):]


def extract_trace_input_line(R: str) -> str:
    for line in R.splitlines(keepends=True):
        if line.startswith("I="):
            return line
    raise ValueError("TRACE_INPUT_LINE_NOT_FOUND")


def extract_trace_lines_with_closing_R(R: str) -> str:
    lines = R.splitlines(keepends=True)
    out = []
    seen_input = False
    for line in lines:
        if seen_input:
            out.append(line)
        if line.startswith("I="):
            seen_input = True
    if not out or out[-1] != "</R>\n":
        raise ValueError("INVALID_R_BLOCK")
    return "".join(out)


def build_sequence_and_mask(record: dict, stage: int):
    if stage == 1:
        prefix = "<S1>\n" + record["R"] + "<C>\n"
        target = extract_code_body_with_closing_C(record["C"])
    elif stage == 2:
        prefix = "<S2>\n" + record["P"] + record["C"] + "<R>\n" + extract_trace_input_line(record["R"])
        target = extract_trace_lines_with_closing_R(record["R"])
    elif stage == 3:
        prefix = "<S3>\n" + record["P"] + "<C>\n"
        target = extract_code_body_with_closing_C(record["C"])
    else:
        raise ValueError("INVALID_STAGE")

    ids = encode_ascii_record(prefix + target)
    if len(ids) > SEQ_LEN:
        raise ValueError("SEQUENCE_TOO_LONG")
    ids = ids + [257] * (SEQ_LEN - len(ids))
    prefix_ids = encode_ascii_record_without_eos(prefix)
    target_start = len(prefix_ids) - 1
    labels = ids[1:]
    input_ids = ids[:-1]
    loss_mask = [0] * (SEQ_LEN - 1)
    for pos in range(target_start, len(ids) - 1):
        if pos < SEQ_LEN - 1:
            loss_mask[pos] = 1
    return input_ids, labels, loss_mask


def compute_depth_losses(logits_by_k, labels, loss_mask):
    if isinstance(logits_by_k, list):
        logits_by_k = torch.stack([x[:-1] if x.shape[0] == SEQ_LEN else x for x in logits_by_k], dim=0)
    ce_by_k = [F.cross_entropy(logits_by_k[k].float(), labels, reduction="none") for k in range(5)]
    denom = torch.clamp(loss_mask.sum(), min=1.0)
    nll_by_k = [(ce_by_k[k] * loss_mask).sum() / denom for k in range(5)]
    omega = torch.tensor([2 * (k + 1) / ((K_TRAIN + 1) * (K_TRAIN + 2)) for k in range(5)], device=logits_by_k.device, dtype=torch.float32)
    nll = sum(omega[k] * nll_by_k[k] for k in range(5))
    mono_acc = sum((torch.relu(ce_by_k[k + 1] - ce_by_k[k]) * loss_mask).sum() for k in range(4))
    mono = mono_acc / (4.0 * denom)
    return {"nll": nll, "mono": mono, "nll_by_k": nll_by_k, "latent_gain": nll_by_k[0] - nll_by_k[4]}


def compute_kd_loss(logits_by_k, teacher_logits_final, loss_mask):
    if isinstance(logits_by_k, list):
        logits_by_k = torch.stack([x[:-1] if x.shape[0] == SEQ_LEN else x for x in logits_by_k], dim=0)
    denom = torch.clamp(loss_mask.sum(), min=1.0)
    with torch.no_grad():
        q = torch.softmax(teacher_logits_final.float(), dim=-1)
        log_q = torch.log_softmax(teacher_logits_final.float(), dim=-1)
    kd = 0.0
    for k in range(5):
        log_p = torch.log_softmax(logits_by_k[k].float(), dim=-1)
        kd = kd + (q * (log_q - log_p)).sum(dim=-1).mul(loss_mask).sum()
    return kd / (5.0 * denom)


def compute_token_diagnostics(logits_by_k, labels, loss_mask) -> dict[str, float]:
    if isinstance(logits_by_k, list):
        logits_by_k = torch.stack([x[:-1] if x.shape[0] == SEQ_LEN else x for x in logits_by_k], dim=0)
    final_logits = logits_by_k[4].float()
    mask = loss_mask.float()
    denom = torch.clamp(mask.sum(), min=1.0)
    ce = F.cross_entropy(final_logits, labels, reduction="none")
    pred = torch.argmax(final_logits, dim=-1)
    out = {
        "token_acc4": float(((pred == labels).float() * mask).sum().detach().cpu() / denom.detach().cpu()),
        "target_tokens": float(mask.sum().detach().cpu()),
    }

    def masked_mean(name: str, class_mask: torch.Tensor) -> None:
        effective = mask * class_mask.float()
        count = effective.sum()
        if float(count.detach().cpu()) > 0.0:
            out[name] = float((ce * effective).sum().detach().cpu() / count.detach().cpu())

    masked_mean("nll4_eos", labels == 257)
    masked_mean("nll4_non_eos", labels != 257)
    masked_mean("nll4_newline", labels == 10)
    masked_mean("nll4_space", labels == 32)
    masked_mean("nll4_digit", (labels >= 48) & (labels <= 57))
    masked_mean("nll4_alpha", ((labels >= 65) & (labels <= 90)) | ((labels >= 97) & (labels <= 122)))
    masked_mean("nll4_tag_chars", (labels == ord("<")) | (labels == ord(">")) | (labels == ord("/")))
    return out


def compute_stage_loss(stage: int, logits_by_k, labels, loss_mask, mdl_loss, teacher_logits_final=None):
    depth = compute_depth_losses(logits_by_k, labels, loss_mask)
    if stage in (1, 2):
        total = depth["nll"] + 0.10 * depth["mono"] + 1e-6 * mdl_loss
    elif stage == 3:
        kd = torch.tensor(0.0, device=labels.device) if teacher_logits_final is None else compute_kd_loss(logits_by_k, teacher_logits_final, loss_mask)
        total = depth["nll"] + 0.10 * depth["mono"] + 0.25 * kd + 1e-6 * mdl_loss
        depth["kd"] = kd
    else:
        raise ValueError("INVALID_STAGE")
    return total, depth


def stage_ready(stage: int, metrics: dict) -> bool:
    cfg = STAGE_CONFIG[stage]
    if metrics["nll4"] > cfg["nll_threshold"] or metrics["mono"] > cfg["mono_threshold"] or metrics["latent_gain"] < cfg["latent_gain_threshold"]:
        return False
    if stage == 1:
        return metrics["compile_pass"] >= 0.995
    if stage == 2:
        return metrics["var_exact"] >= 0.990 and metrics["line_exact"] >= 0.970
    if stage == 3:
        return metrics["compile_pass"] >= 0.990 and metrics["unit_pass"] >= 0.920
    raise ValueError("INVALID_STAGE")


def model_forward_train(model, input_ids_t: torch.Tensor):
    if hasattr(model, "forward_native") and not torch.is_grad_enabled():
        logits, _state = model.forward_native(input_ids_t, k=K_TRAIN, return_all_depths=True)
        return torch.stack([(x[:-1] if x.shape[0] == SEQ_LEN else x).float() for x in logits], dim=0)
    if hasattr(model, "forward_train"):
        return model.forward_train(input_ids=input_ids_t, k_train=K_TRAIN, return_logits_by_depth=True)
    logits, _ = model.forward_python_reference(input_ids_t, k=K_TRAIN, return_all_depths=True)
    return torch.stack([x[:-1].float() for x in logits], dim=0)


def compile_and_unit_check_generated_code(generated_code: str, record: dict, tests_count: int):
    ok_compile = False
    try:
        fn = compile_restricted_function(generated_code)
        ok_compile = True
    except Exception:
        return False, False
    rng = RNG(GLOBAL_SEED ^ 0xA11CE ^ int(record["index"]))
    for case in make_cases(record["family_id"], record["stage"], rng, tests_count):
        try:
            if run_restricted(fn, case) != run_reference_interpreter(record["family_id"], record["params"], case):
                return ok_compile, False
        except Exception:
            return ok_compile, False
    return ok_compile, True


def greedy_generate_text(model, prompt: str, max_new_bytes: int) -> str:
    if hasattr(model, "greedy_generate_code"):
        return model.greedy_generate_code(prompt=prompt, max_new_bytes=max_new_bytes)
    tok = ByteLevelTokenizer()
    ids = tok.encode_prompt(prompt)
    out_ids = model.generate(ids, max_new_bytes)
    generated = tok.decode_generated_tokens(out_ids[len(ids):])
    return generated


def greedy_generate_trace_text(model, prompt: str, max_new_bytes: int) -> str:
    if hasattr(model, "greedy_generate_trace"):
        return model.greedy_generate_trace(prompt=prompt, max_new_bytes=max_new_bytes)
    return greedy_generate_text(model, prompt, max_new_bytes)


def variable_exact_counts(generated_trace: str, ref_trace: str):
    num = den = 0
    for g, r in zip(generated_trace.splitlines(), ref_trace.splitlines()):
        if "|" not in r:
            continue
        gs = dict(item.split("=", 1) for item in g.split("|", 1)[1].split(",") if "=" in item) if "|" in g else {}
        for item in r.split("|", 1)[1].split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                den += 1
                num += int(gs.get(k) == v)
    return num, den


def line_exact_counts(generated_trace: str, ref_trace: str):
    ref = ref_trace.splitlines()
    gen = generated_trace.splitlines()
    return sum(int(i < len(gen) and gen[i] == line) for i, line in enumerate(ref)), len(ref)


@torch.no_grad()
def evaluate_stage(model, stage: int, validation_indices: list[int], teacher_cache=None, *, generate_outputs: bool = True):
    nll4_sum = mono_sum = gain_sum = 0.0
    compile_pass = unit_pass = count = 0
    var_exact_num = var_exact_den = line_exact_num = line_exact_den = 0
    diag_sums: dict[str, float] = {}
    diag_counts: dict[str, int] = {}
    for idx in validation_indices:
        record = generate_record(stage, idx)
        input_ids, labels, loss_mask = build_sequence_and_mask(record, stage)
        input_ids_t = torch.tensor(input_ids, device="cuda", dtype=torch.long)
        labels_t = torch.tensor(labels, device="cuda", dtype=torch.long)
        mask_t = torch.tensor(loss_mask, device="cuda", dtype=torch.float32)
        logits_by_k = model_forward_train(model, input_ids_t)
        depth = compute_depth_losses(logits_by_k, labels_t, mask_t)
        token_diag = compute_token_diagnostics(logits_by_k, labels_t, mask_t)
        for key, value in token_diag.items():
            diag_sums[key] = diag_sums.get(key, 0.0) + float(value)
            diag_counts[key] = diag_counts.get(key, 0) + 1
        nll4_sum += float(depth["nll_by_k"][4].detach().cpu())
        mono_sum += float(depth["mono"].detach().cpu())
        gain_sum += float(depth["latent_gain"].detach().cpu())
        count += 1
        if generate_outputs and stage in (1, 3) and (hasattr(model, "greedy_generate_code") or hasattr(model, "generate")):
            prompt = ("<S1>\n" + record["R"] + "<C>\n") if stage == 1 else ("<S3>\n" + record["P"] + "<C>\n")
            ok_c, ok_u = compile_and_unit_check_generated_code(greedy_generate_text(model, prompt, 384), record, 256)
            compile_pass += int(ok_c)
            unit_pass += int(ok_u)
        if generate_outputs and stage == 2 and (hasattr(model, "greedy_generate_trace") or hasattr(model, "generate")):
            prompt = "<S2>\n" + record["P"] + record["C"] + "<R>\n" + extract_trace_input_line(record["R"])
            generated = greedy_generate_trace_text(model, prompt, 384)
            ref = extract_trace_lines_with_closing_R(record["R"])
            ve_num, ve_den = variable_exact_counts(generated, ref)
            le_num, le_den = line_exact_counts(generated, ref)
            var_exact_num += ve_num
            var_exact_den += ve_den
            line_exact_num += le_num
            line_exact_den += le_den
    metrics = {"nll4": nll4_sum / count, "mono": mono_sum / count, "latent_gain": gain_sum / count}
    for key, total in diag_sums.items():
        metrics[key] = total / max(1, diag_counts[key])
    if stage in (1, 3):
        metrics["compile_pass"] = compile_pass / count
        metrics["unit_pass"] = unit_pass / count
    if stage == 2:
        metrics["var_exact"] = var_exact_num / max(1, var_exact_den)
        metrics["line_exact"] = line_exact_num / max(1, line_exact_den)
    return metrics


class TeacherCacheWriter:
    def __init__(self, cache_path: str):
        self.root = Path(cache_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = (self.root / "manifest.jsonl").open("a", encoding="ascii")

    def write(self, index: int, logits: torch.Tensor) -> None:
        path = self.root / f"{index}.pt"
        torch.save(logits.cpu(), path)
        self.manifest.write(json.dumps({"index": index, "file": path.name}, sort_keys=True) + "\n")
        self.manifest.flush()

    def close(self) -> None:
        self.manifest.close()


class TeacherCacheReader:
    def __init__(self, cache_path: str):
        self.root = Path(cache_path)
        self.files: dict[int, str] = {}
        manifest = self.root / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="ascii").splitlines():
                row = json.loads(line)
                self.files[int(row["index"])] = str(row["file"])

    def lookup(self, index: int) -> torch.Tensor | None:
        name = self.files.get(int(index))
        if name is None:
            return None
        return torch.load(self.root / name, map_location="cpu")


def open_cache_writer(cache_path: str) -> TeacherCacheWriter:
    return TeacherCacheWriter(cache_path)


def close_cache_writer(writer: TeacherCacheWriter) -> None:
    writer.close()


def open_teacher_cache_reader(cache_path: str) -> TeacherCacheReader:
    return TeacherCacheReader(cache_path)


def write_teacher_logits(cache_path: str, index: int, logits: torch.Tensor) -> None:
    writer = TeacherCacheWriter(cache_path)
    try:
        writer.write(index, logits)
    finally:
        writer.close()


@torch.no_grad()
def precompute_stage3_teacher_cache(model, stage3_indices: list[int], cache_path: str):
    writer = open_cache_writer(cache_path)
    try:
        for idx in stage3_indices:
            record = generate_record(stage=3, index=idx)
            teacher_text = "<S2>\n" + record["P"] + record["C"] + record["R"]
            ids = encode_ascii_record(teacher_text)
            if len(ids) > SEQ_LEN:
                continue
            ids = ids + [257] * (SEQ_LEN - len(ids))
            input_ids = torch.tensor(ids[:-1], device="cuda", dtype=torch.long)
            logits_by_k = model_forward_train(model, input_ids)
            final_logits = logits_by_k[4].detach().to(torch.float16).cpu()
            writer.write(idx, final_logits)
    finally:
        close_cache_writer(writer)


def build_ssd_pool(
    model,
    stage3_problem_indices: list[int],
    global_step: int,
    *,
    candidates_per_problem: int = SSD_CANDIDATES_PER_PROBLEM,
    max_accepted_per_problem: int = SSD_MAX_ACCEPTED_PER_PROBLEM,
    max_code_bytes: int = SSD_MAX_CODE_BYTES,
    consensus_min: int = SSD_CONSENSUS_MIN,
    tests_count: int = 512,
):
    accepted = []
    if not hasattr(model, "generate_bytes"):
        return accepted
    for pidx in stage3_problem_indices:
        record = generate_record(stage=3, index=pidx)
        prompt_ids = encode_ascii_record_without_eos("<S3>\n" + record["P"] + "<C>\n")
        candidates = []
        for c in range(candidates_per_problem):
            out_ids = model.generate_bytes(prompt_ids=prompt_ids, max_new_bytes=max_code_bytes, temperature=TEMPERATURE_CYCLE[c % 4], top_p=TOP_P_CYCLE[c % 3], top_k=32, seed=GLOBAL_SEED ^ (3 << 56) ^ (pidx << 16) ^ global_step ^ c, stop_bytes=b"</C>\n")
            code = extract_code_between_C_tags(out_ids)
            if code is None:
                continue
            ok, signature, canonical_ast = verify_candidate_code(code, record, tests_count)
            if ok:
                candidates.append({"code": code, "signature": signature, "canonical_ast": canonical_ast, "length": len(code.encode("ascii"))})
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in candidates:
            groups.setdefault(item["signature"], []).append(item)
        selected = []
        for signature in sorted(groups):
            if len(groups[signature]) >= consensus_min:
                selected.append(sorted(groups[signature], key=lambda z: (z["length"], z["canonical_ast"], z["code"]))[0])
        for item in selected[:max_accepted_per_problem]:
            C = "<C>\n" + item["code"] + "</C>\n"
            ids = encode_ascii_record("<S3>\n" + record["P"] + C)
            if len(ids) <= 512:
                accepted.append({"stage": 3, "P": record["P"], "R": record["R"], "C": C, "token_ids": ids, "source": "SSD_F", "index": record["index"], "family_id": record["family_id"], "params": record["params"]})
    return accepted


def save_metrics_jsonl(output_dir: str, **row) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(output_dir) / "metrics.jsonl").open("a", encoding="ascii") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def train_curriculum(
    model,
    kfac_optimizer,
    output_dir: str,
    *,
    max_updates_override: dict[int, int] | None = None,
    eval_every_updates_override: int | None = None,
    val_size_override: int | None = None,
    require_thresholds: bool = True,
    enable_ssd: bool | None = None,
):
    torch.manual_seed(123456789)
    torch.cuda.manual_seed_all(123456789)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    val_size = VAL_SIZE_PER_STAGE if val_size_override is None else int(val_size_override)
    eval_every = EVAL_EVERY_UPDATES if eval_every_updates_override is None else int(eval_every_updates_override)
    max_updates = MAX_UPDATES if max_updates_override is None else {stage: int(max_updates_override.get(stage, MAX_UPDATES[stage])) for stage in (1, 2, 3)}
    use_ssd_stage3 = (enable_ssd if enable_ssd is not None else require_thresholds)
    validation_indices = {1: [10_000_000 + i for i in range(val_size)], 2: [20_000_000 + i for i in range(val_size)], 3: [30_000_000 + i for i in range(val_size)]}
    global_update = 0
    ssd_pool = []
    teacher_cache = None
    for stage in [1, 2, 3]:
        cfg = STAGE_CONFIG[stage]
        if hasattr(kfac_optimizer, "reset_curvature"):
            kfac_optimizer.reset_curvature(active_layers=cfg["active_layers"], damping=cfg["damping"])
        if hasattr(kfac_optimizer, "set_hyperparams"):
            kfac_optimizer.set_hyperparams(eta=cfg["eta"], damping=cfg["damping"], trust_region_delta=cfg["delta"])
        consecutive_ready = 0
        if stage == 3:
            teacher_cache_path = str(Path(output_dir) / "teacher_stage3_logits")
            precompute_stage3_teacher_cache(model, validation_indices[3], teacher_cache_path)
            teacher_cache = open_teacher_cache_reader(teacher_cache_path)
            if use_ssd_stage3:
                ssd_pool = build_ssd_pool(model, [40_000_000 + i for i in range(SSD_PROBLEMS_PER_REFRESH)], global_update)
            else:
                ssd_pool = []
        for local_update in range(max_updates[stage]):
            train_index = stage * 1_000_000_000 + local_update
            use_ssd = stage == 3 and len(ssd_pool) > 0 and RNG(GLOBAL_SEED ^ global_update ^ 0x55D15A11).randint(0, 99) < 30
            record = ssd_pool[global_update % len(ssd_pool)] if use_ssd else generate_record(stage, train_index)
            input_ids, labels, loss_mask = build_sequence_and_mask(record, stage)
            chunk_ids = input_ids + [labels[-1]]
            input_ids_t = torch.tensor(input_ids, device="cuda", dtype=torch.long)
            labels_t = torch.tensor(labels, device="cuda", dtype=torch.long)
            mask_t = torch.tensor(loss_mask, device="cuda", dtype=torch.float32)
            teacher_logits = None
            if stage == 3 and teacher_cache is not None and "index" in record:
                cached_logits = teacher_cache.lookup(record["index"])
                if cached_logits is not None:
                    teacher_logits = cached_logits.to("cuda", dtype=torch.float32)
            if hasattr(model, "train_chunk_cuda") and hasattr(kfac_optimizer, "state"):
                chunk_t = torch.tensor(chunk_ids, device="cuda", dtype=torch.long)
                model.train_chunk_cuda(
                    chunk_t,
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
            else:
                logits_by_k = model_forward_train(model, input_ids_t)
                mdl_loss = plastic_mdl((p for _, p in plastic_named_parameters(model)))
                loss, _depth = compute_stage_loss(stage, logits_by_k, labels_t, mask_t, mdl_loss, teacher_logits)
                if hasattr(kfac_optimizer, "zero_grad"):
                    kfac_optimizer.zero_grad()
                loss.backward()
                kfac_optimizer.step(active_layers=cfg["active_layers"], eta=cfg["eta"], damping=cfg["damping"], trust_region_delta=cfg["delta"])
            global_update += 1
            if use_ssd_stage3 and stage == 3 and global_update % SSD_REFRESH_UPDATES == 0:
                ssd_pool = build_ssd_pool(model, [50_000_000 + global_update + i for i in range(SSD_PROBLEMS_PER_REFRESH)], global_update)
            if local_update % eval_every == 0:
                metrics = evaluate_stage(model, stage, validation_indices[stage], teacher_cache, generate_outputs=require_thresholds)
                consecutive_ready = consecutive_ready + 1 if (not require_thresholds or stage_ready(stage, metrics)) else 0
                save_metrics_jsonl(output_dir, stage=stage, global_update=global_update, local_update=local_update, metrics=metrics, consecutive_ready=consecutive_ready)
                if consecutive_ready >= ADVANCE_CONSECUTIVE_EVALS:
                    break
        if consecutive_ready < ADVANCE_CONSECUTIVE_EVALS:
            if require_thresholds:
                raise RuntimeError(f"CURRICULUM_STAGE_{stage}_FAILED_THRESHOLDS")
            save_metrics_jsonl(output_dir, stage=stage, global_update=global_update, local_update=max_updates[stage], metrics={"dry_run_complete": True}, consecutive_ready=consecutive_ready)
    return model
