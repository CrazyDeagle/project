from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from silexcode.kfac import BlockKFACOptimizer
from silexcode.model import SilexCodeT18_6B_R64
from silexcode.train import train_curriculum
from silexcode.training import plastic_named_parameters


def main() -> None:
    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out = Path("runs") / "curriculum_dry_run"
    if out.exists():
        shutil.rmtree(out)
    model = SilexCodeT18_6B_R64(device="cuda")
    optimizer = BlockKFACOptimizer(plastic_named_parameters(model), lr=0.04, damping=3e-4, trust_region=5e-4)
    train_curriculum(
        model,
        optimizer,
        str(out),
        max_updates_override={1: 1, 2: 1, 3: 1},
        eval_every_updates_override=1,
        val_size_override=1,
        require_thresholds=False,
        enable_ssd=False,
    )
    for stage in (1, 2, 3):
        print(f"stage={stage} dry_run_complete", flush=True)
    print("curriculum_dry_run=PASS", flush=True)


if __name__ == "__main__":
    main()
