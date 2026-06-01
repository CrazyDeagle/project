from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_jsonl")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.metrics_jsonl).read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if "validation" in row]
    if not rows:
        raise SystemExit("NO_BOOTSTRAP_VALIDATION_ROWS")

    print("level,update,nll4,token_acc4,target_fraction,natural_norm,trust_chi,step_seconds")
    for row in rows:
        val = row["validation"]
        train = row["train"]
        print(
            ",".join(
                [
                    str(row["bootstrap_level"]),
                    str(row["global_update"]),
                    f"{float(val['nll4']):.6f}",
                    f"{float(val.get('token_acc4', 0.0)):.6f}",
                    f"{float(row.get('target_fraction', 0.0)):.6f}",
                    f"{float(train.get('natural_norm', 0.0)):.6f}",
                    f"{float(train.get('trust_chi', 1.0)):.8f}",
                    f"{float(train.get('step_seconds', 0.0)):.6f}",
                ]
            )
        )

    print()
    for level in sorted({int(row["bootstrap_level"]) for row in rows}):
        level_rows = [row for row in rows if int(row["bootstrap_level"]) == level]
        nlls = [float(row["validation"]["nll4"]) for row in level_rows]
        accs = [float(row["validation"].get("token_acc4", 0.0)) for row in level_rows]
        naturals = [float(row["train"].get("natural_norm", 0.0)) for row in level_rows]
        print(f"level_{level}_first_nll4={nlls[0]:.6f}")
        print(f"level_{level}_last_nll4={nlls[-1]:.6f}")
        print(f"level_{level}_best_nll4={min(nlls):.6f}")
        print(f"level_{level}_last_token_acc4={accs[-1]:.6f}")
        print(f"level_{level}_max_natural_norm={max(naturals):.6f}")
    print(
        f"avg_step_seconds={statistics.fmean(float(row['train'].get('step_seconds', 0.0)) for row in rows):.6f}"
    )


if __name__ == "__main__":
    main()
