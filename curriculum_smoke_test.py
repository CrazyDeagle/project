from __future__ import annotations

import math
import os

import torch

from silexcode.dataset import generate_record
from silexcode.kfac import BlockKFACOptimizer
from silexcode.model import SilexCodeT18_6B_R64
from silexcode.train import STAGE_CONFIG, build_sequence_and_mask
from silexcode.training import plastic_named_parameters


def _finite_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"NON_FINITE_METRIC:{key}")


def main() -> None:
    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = SilexCodeT18_6B_R64(device="cuda")
    optimizer = BlockKFACOptimizer(plastic_named_parameters(model), lr=0.04, damping=3e-4, trust_region=5e-4)
    workspace = model.allocate_train_workspace()
    state = model.initial_state()

    for stage in (1, 2, 3):
        cfg = STAGE_CONFIG[stage]
        optimizer.reset_curvature(active_layers=cfg["active_layers"], damping=cfg["damping"])
        record = generate_record(stage, stage * 1_000_000_000)
        input_ids, labels, loss_mask = build_sequence_and_mask(record, stage)
        chunk_ids = input_ids + [labels[-1]]
        metrics, state = model.train_chunk_cuda(
            torch.tensor(chunk_ids, device="cuda", dtype=torch.long),
            state=state,
            workspace=workspace,
            labels=torch.tensor(labels, device="cuda", dtype=torch.long),
            loss_mask=torch.tensor(loss_mask, device="cuda", dtype=torch.float32),
            stage=stage,
            kfac_optimizer=optimizer,
            active_layers=cfg["active_layers"],
            eta=cfg["eta"],
            damping=cfg["damping"],
            trust_region_delta=cfg["delta"],
        )
        torch.cuda.synchronize()
        for required in ("nll", "nll4", "mono", "latent_gain", "natural_norm", "updated_matrices"):
            if required not in metrics:
                raise RuntimeError(f"MISSING_METRIC:{required}")
        _finite_metrics(metrics)
        state.zero_()
        print(f"stage={stage} ok", flush=True)

    print("curriculum_smoke_test=PASS", flush=True)


if __name__ == "__main__":
    main()
