from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from silexcode.bootstrap import BOOTSTRAP_LEVELS, train_bootstrap
from silexcode.checkpoint import import_plastic_checkpoint
from silexcode.kfac import BlockKFACOptimizer
from silexcode.model import SilexCodeT18_6B_R64
from silexcode.training import plastic_named_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--levels", default=",".join(str(x) for x in BOOTSTRAP_LEVELS))
    parser.add_argument("--updates-per-level", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-size", type=int, default=16)
    parser.add_argument("--max-records-per-chunk", type=int, default=16)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--damping", type=float, default=1.0e-2)
    parser.add_argument("--trust-region-delta", type=float, default=3.0e-5)
    parser.add_argument("--kfac-warmup-updates", type=int, default=100)
    parser.add_argument("--checkpoint-every-evals", type=int, default=0)
    parser.add_argument("--include-kfac", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model = SilexCodeT18_6B_R64(device="cuda")
    optimizer = BlockKFACOptimizer(plastic_named_parameters(model), lr=args.eta, damping=args.damping, trust_region=args.trust_region_delta)
    if args.resume:
        import_plastic_checkpoint(model, args.resume, kfac_optimizer=optimizer)

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    updates_per_level = 1 if args.dry_run else args.updates_per_level
    eval_every = 1 if args.dry_run else args.eval_every
    val_size = 1 if args.dry_run else args.val_size

    train_bootstrap(
        model,
        optimizer,
        args.output_dir,
        levels=levels,
        updates_per_level=updates_per_level,
        eval_every=eval_every,
        val_size=val_size,
        max_records_per_chunk=args.max_records_per_chunk,
        candidate_multiplier=args.candidate_multiplier,
        eta=args.eta,
        damping=args.damping,
        trust_region_delta=args.trust_region_delta,
        kfac_warmup_updates=args.kfac_warmup_updates,
        checkpoint_every_evals=args.checkpoint_every_evals,
        include_kfac_in_checkpoints=args.include_kfac,
    )
    print("run_bootstrap=PASS", flush=True)


if __name__ == "__main__":
    main()
