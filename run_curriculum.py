from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from silexcode.checkpoint import import_silex_checkpoint
from silexcode.kfac import BlockKFACOptimizer
from silexcode.model import SilexCodeT18_6B_R64
from silexcode.train import train_curriculum
from silexcode.training import plastic_named_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--include-kfac", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = SilexCodeT18_6B_R64(device="cuda")
    optimizer = BlockKFACOptimizer(
        plastic_named_parameters(model), lr=0.04, damping=3e-4, trust_region=5e-4
    )
    if args.resume:
        import_silex_checkpoint(model, args.resume, kfac_optimizer=optimizer)

    max_updates = None
    if args.dry_run:
        max_updates = {1: 1, 2: 1, 3: 1}
    elif args.max_updates is not None:
        max_updates = {1: args.max_updates, 2: args.max_updates, 3: args.max_updates}

    train_curriculum(
        model,
        optimizer,
        args.output_dir,
        max_updates_override=max_updates,
        eval_every_updates_override=(1 if args.dry_run else args.eval_every),
        val_size_override=(1 if args.dry_run else args.val_size),
        require_thresholds=not args.dry_run,
        enable_ssd=(False if args.dry_run else None),
    )
    print("run_curriculum=PASS", flush=True)


if __name__ == "__main__":
    main()
