from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_jsonl")
    args = parser.parse_args()

    rows = [row for row in _read_rows(Path(args.metrics_jsonl)) if "validation" in row]
    if not rows:
        raise SystemExit("NO_VALIDATION_ROWS")

    print("stage,update,nll4,mono,latent_gain,token_acc4,compile_pass,unit_pass,packed_records,target_tokens,target_fraction,natural_norm,trust_chi,step_seconds")
    for row in rows:
        val = row["validation"]
        train = row.get("train", {})
        print(
            ",".join(
                [
                    str(row.get("stage", "")),
                    str(row.get("global_update", "")),
                    f"{float(val.get('nll4', 0.0)):.6f}",
                    f"{float(val.get('mono', 0.0)):.6f}",
                    f"{float(val.get('latent_gain', 0.0)):.6f}",
                    f"{float(val.get('token_acc4', 0.0)):.6f}" if "token_acc4" in val else "",
                    f"{float(val.get('compile_pass', 0.0)):.6f}" if "compile_pass" in val else "",
                    f"{float(val.get('unit_pass', 0.0)):.6f}" if "unit_pass" in val else "",
                    str(row.get("packed_records", "")),
                    str(row.get("target_tokens", "")),
                    f"{float(row.get('target_fraction', 0.0)):.6f}" if "target_fraction" in row else "",
                    f"{float(train.get('natural_norm', 0.0)):.6f}" if "natural_norm" in train else "",
                    f"{float(train.get('trust_chi', 0.0)):.6f}" if "trust_chi" in train else "",
                    f"{float(train.get('step_seconds', 0.0)):.6f}" if "step_seconds" in train else "",
                ]
            )
        )

    print()
    latest_nll = None
    for stage in sorted({int(row["stage"]) for row in rows if "stage" in row}):
        stage_rows = [row for row in rows if int(row["stage"]) == stage]
        first = stage_rows[0]["validation"]
        last = stage_rows[-1]["validation"]
        best = min(float(row["validation"]["nll4"]) for row in stage_rows)
        latest_nll = float(last["nll4"])
        nll_delta = float(first["nll4"]) - latest_nll
        target_fracs = [float(row.get("target_fraction", 0.0)) for row in stage_rows if "target_fraction" in row]
        natural_norms = [float(row.get("train", {}).get("natural_norm", 0.0)) for row in stage_rows if "natural_norm" in row.get("train", {})]
        chis = [float(row.get("train", {}).get("trust_chi", 1.0)) for row in stage_rows if "trust_chi" in row.get("train", {})]
        step_seconds = [float(row.get("train", {}).get("step_seconds", 0.0)) for row in stage_rows if "step_seconds" in row.get("train", {})]
        print(f"stage_{stage}_first_nll4={float(first['nll4']):.6f}")
        print(f"stage_{stage}_last_nll4={latest_nll:.6f}")
        print(f"stage_{stage}_best_nll4={best:.6f}")
        print(f"stage_{stage}_nll4_delta={nll_delta:.6f}")
        if target_fracs:
            print(f"stage_{stage}_avg_target_fraction={statistics.fmean(target_fracs):.6f}")
        if natural_norms:
            print(f"stage_{stage}_max_natural_norm={max(natural_norms):.6f}")
            print(f"stage_{stage}_natural_norm_spikes_gt_1e5={sum(x > 1.0e5 for x in natural_norms)}")
        if chis:
            print(f"stage_{stage}_min_trust_chi={min(chis):.8f}")
        if step_seconds:
            print(f"stage_{stage}_avg_step_seconds={statistics.fmean(step_seconds):.6f}")
    print("target_stage1_nll4=0.080000")
    if latest_nll is not None and latest_nll > 1.0:
        print("recommendation=do_not_run_full_threshold_training_yet")
    else:
        print("recommendation=run_longer_probe_or_strict_curriculum")


if __name__ == "__main__":
    main()
