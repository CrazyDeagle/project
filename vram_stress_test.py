from __future__ import annotations

import argparse
import os
import time

import torch

from silexcode.kfac import BlockKFACOptimizer
from silexcode.model import SilexCodeT18_6B_R64
from silexcode.training import plastic_named_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("deterministic", "packed"), default="deterministic")
    args = parser.parse_args()

    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VRAM stress test")

    t0 = time.perf_counter()
    print("phase=setup_cuda", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    print("phase=model_init_begin", flush=True)
    model = SilexCodeT18_6B_R64(device=args.device)
    if args.mode == "packed":
        model.mark_checkpoint_backbone_loaded()
    torch.cuda.synchronize()
    print(
        f"phase=model_init_done mode={args.mode} seconds={time.perf_counter() - t0:.3f}", flush=True
    )

    model.eval()
    print("phase=kfac_init_begin", flush=True)
    optimizer = BlockKFACOptimizer(
        plastic_named_parameters(model), lr=0.04, damping=3e-4, trust_region=5e-4
    )
    optimizer.reset_curvature(active_layers=list(range(1, 65)), damping=3e-4)
    torch.cuda.synchronize()
    print(f"phase=kfac_init_done seconds={time.perf_counter() - t0:.3f}", flush=True)

    print("phase=workspace_init_begin", flush=True)
    workspace = model.allocate_train_workspace()
    state = model.initial_state()
    tokens = (torch.arange(512, device=args.device, dtype=torch.int64) % 258).to(torch.uint16)
    labels = tokens[1:].to(torch.long)
    loss_mask = torch.ones(511, device=args.device, dtype=torch.float32)
    torch.cuda.synchronize()
    print(f"phase=workspace_init_done seconds={time.perf_counter() - t0:.3f}", flush=True)

    with torch.no_grad():
        for step in range(args.steps):
            step_t0 = time.perf_counter()
            print(f"phase=train_step_begin step={step}", flush=True)
            _metrics, state = model.train_chunk_cuda(
                tokens,
                state=state,
                workspace=workspace,
                labels=labels,
                loss_mask=loss_mask,
                stage=3,
                kfac_optimizer=optimizer,
                active_layers=list(range(1, 65)),
                eta=0.04,
                damping=3e-4,
                trust_region_delta=5e-4,
            )
            state.zero_()
            torch.cuda.synchronize()
            print(
                f"phase=train_step_done step={step} seconds={time.perf_counter() - step_t0:.3f}",
                flush=True,
            )
        torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    current_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"current_allocated_mb={current_mb}", flush=True)
    print(peak_mb)
    if peak_mb > 7256.25598526001:
        raise RuntimeError(f"VRAM_LIMIT_EXCEEDED:{peak_mb:.6f}MB")


if __name__ == "__main__":
    main()
