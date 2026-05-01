from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from silexcode.checkpoint import export_plastic_checkpoint, import_plastic_checkpoint


def _write_metrics(path: Path) -> None:
    rows = [
        {
            "stage": 1,
            "global_update": 1,
            "packed_records": 3,
            "target_tokens": 240,
            "target_fraction": 0.47,
            "train": {"natural_norm": 10.0, "trust_chi": 0.5, "step_seconds": 2.0},
            "validation": {"nll4": 12.0, "mono": 0.01, "latent_gain": 0.02, "token_acc4": 0.1},
        },
        {
            "stage": 1,
            "global_update": 101,
            "packed_records": 4,
            "target_tokens": 260,
            "target_fraction": 0.51,
            "train": {"natural_norm": 20.0, "trust_chi": 0.25, "step_seconds": 3.0},
            "validation": {"nll4": 9.0, "mono": 0.02, "latent_gain": 0.03, "token_acc4": 0.2},
        },
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="ascii")


def test_analyze_curriculum_metrics_outputs_decision(tmp_path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics)
    out = subprocess.check_output([sys.executable, "analyze_curriculum_metrics.py", str(metrics)], text=True)
    assert "stage_1_best_nll4=9.000000" in out
    assert "recommendation=do_not_run_full_threshold_training_yet" in out


def test_compare_runs_selects_best(tmp_path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_metrics(a)
    _write_metrics(b)
    rows = [json.loads(line) for line in b.read_text(encoding="ascii").splitlines()]
    rows[-1]["validation"]["nll4"] = 7.0
    b.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="ascii")
    out = subprocess.check_output([sys.executable, "compare_runs.py", str(a), str(b)], text=True)
    assert f"best_run={b}" in out
    assert "best_run_best_nll4=7.000000" in out


def test_plastic_checkpoint_roundtrip_with_small_module(tmp_path) -> None:
    class Tiny(torch.nn.Module):
        name = "tiny"

        def __init__(self) -> None:
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
            self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    src = Tiny()
    path = tmp_path / "tiny.plastic.silex"
    export_plastic_checkpoint(src, path, metadata={"stage": 1})

    dst = Tiny()
    dst.adapter.data.zero_()
    meta = import_plastic_checkpoint(dst, path)

    assert meta["metadata"] == {"stage": 1}
    assert torch.equal(dst.adapter, src.adapter)
    assert torch.equal(dst.frozen, torch.ones(1))
