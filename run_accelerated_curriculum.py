from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from silexcode.accelerated import train_accelerated_curriculum
from silexcode.checkpoint import import_silex_checkpoint
from silexcode.kfac import BlockKFACOptimizer
from silexcode.model import SilexCodeT18_6B_R64
from silexcode.training import plastic_named_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=16)
    parser.add_argument("--max-records-per-chunk", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--include-padding-loss", action="store_true")
    parser.add_argument("--require-thresholds", action="store_true")
    parser.add_argument("--generate-eval-outputs", action="store_true")
    parser.add_argument("--enable-ssd", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stages", default="1,2,3")
    args = parser.parse_args()

    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = SilexCodeT18_6B_R64(device="cuda")
    optimizer = BlockKFACOptimizer(plastic_named_parameters(model), lr=0.04, damping=3e-4, trust_region=5e-4)
    if args.resume:
        import_silex_checkpoint(model, args.resume, kfac_optimizer=optimizer)

    if args.dry_run:
        max_updates = {1: 1, 2: 1, 3: 1}
        eval_every = 1
        val_size = 1
        require_thresholds = False
    else:
        max_updates = None if args.max_updates is None else {1: args.max_updates, 2: args.max_updates, 3: args.max_updates}
        eval_every = args.eval_every
        val_size = args.val_size
        require_thresholds = args.require_thresholds

    stages = tuple(int(x) for x in args.stages.split(",") if x.strip())
    train_accelerated_curriculum(
        model,
        optimizer,
        args.output_dir,
        stages=stages,
        max_updates_override=max_updates,
        eval_every_updates_override=eval_every,
        val_size_override=val_size,
        max_records_per_chunk=args.max_records_per_chunk,
        candidate_multiplier=args.candidate_multiplier,
        include_padding_loss=args.include_padding_loss,
        require_thresholds=require_thresholds,
        generate_eval_outputs=args.generate_eval_outputs,
        enable_ssd=args.enable_ssd,
    )
    print("run_accelerated_curriculum=PASS", flush=True)


if __name__ == "__main__":
    main()
