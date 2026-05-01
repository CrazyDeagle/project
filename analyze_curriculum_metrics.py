from __future__ import annotations

import argparse
import json
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

    print("stage,update,nll4,mono,latent_gain,compile_pass,unit_pass,packed_records,target_tokens")
    for row in rows:
        val = row["validation"]
        print(
            ",".join(
                [
                    str(row.get("stage", "")),
                    str(row.get("global_update", "")),
                    f"{float(val.get('nll4', 0.0)):.6f}",
                    f"{float(val.get('mono', 0.0)):.6f}",
                    f"{float(val.get('latent_gain', 0.0)):.6f}",
                    f"{float(val.get('compile_pass', 0.0)):.6f}" if "compile_pass" in val else "",
                    f"{float(val.get('unit_pass', 0.0)):.6f}" if "unit_pass" in val else "",
                    str(row.get("packed_records", "")),
                    str(row.get("target_tokens", "")),
                ]
            )
        )

    print()
    latest_nll = None
    for stage in sorted({int(row["stage"]) for row in rows if "stage" in row}):
        stage_rows = [row for row in rows if int(row["stage"]) == stage]
        first = stage_rows[0]["validation"]
        last = stage_rows[-1]["validation"]
        latest_nll = float(last["nll4"])
        nll_delta = float(first["nll4"]) - latest_nll
        print(f"stage_{stage}_first_nll4={float(first['nll4']):.6f}")
        print(f"stage_{stage}_last_nll4={latest_nll:.6f}")
        print(f"stage_{stage}_nll4_delta={nll_delta:.6f}")
    print("target_stage1_nll4=0.080000")
    if latest_nll is not None and latest_nll > 1.0:
        print("recommendation=do_not_run_full_threshold_training_yet")
    else:
        print("recommendation=run_longer_probe_or_strict_curriculum")


if __name__ == "__main__":
    main()
