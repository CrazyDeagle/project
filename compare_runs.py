from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "validation" in row:
            rows.append(row)
    return rows


def summarize(path: Path) -> dict[str, float | str]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"NO_VALIDATION_ROWS:{path}")
    nlls = [float(row["validation"]["nll4"]) for row in rows]
    updates = [int(row.get("global_update", 0)) for row in rows]
    naturals = [float(row.get("train", {}).get("natural_norm", 0.0)) for row in rows if "natural_norm" in row.get("train", {})]
    targets = [float(row.get("target_tokens", 0.0)) for row in rows if "target_tokens" in row]
    steps = [float(row.get("train", {}).get("step_seconds", 0.0)) for row in rows if "step_seconds" in row.get("train", {})]
    span = max(1, updates[-1] - updates[0])
    return {
        "path": str(path),
        "first_nll4": nlls[0],
        "last_nll4": nlls[-1],
        "best_nll4": min(nlls),
        "delta_nll4": nlls[0] - nlls[-1],
        "slope_per_1k_updates": (nlls[-1] - nlls[0]) * 1000.0 / float(span),
        "avg_target_tokens": statistics.fmean(targets) if targets else 0.0,
        "max_natural_norm": max(naturals) if naturals else 0.0,
        "natural_norm_spikes_gt_1e5": float(sum(x > 1.0e5 for x in naturals)),
        "avg_step_seconds": statistics.fmean(steps) if steps else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    args = parser.parse_args()

    summaries = [summarize(Path(path)) for path in args.metrics]
    keys = [
        "path",
        "first_nll4",
        "last_nll4",
        "best_nll4",
        "delta_nll4",
        "slope_per_1k_updates",
        "avg_target_tokens",
        "max_natural_norm",
        "natural_norm_spikes_gt_1e5",
        "avg_step_seconds",
    ]
    print(",".join(keys))
    for row in summaries:
        print(",".join(str(row[key]) if key == "path" else f"{float(row[key]):.6f}" for key in keys))

    best = min(summaries, key=lambda row: float(row["best_nll4"]))
    print()
    print(f"best_run={best['path']}")
    print(f"best_run_best_nll4={float(best['best_nll4']):.6f}")


if __name__ == "__main__":
    main()
