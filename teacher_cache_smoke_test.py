from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from silexcode.model import SilexCodeT18_6B_R64
from silexcode.train import open_teacher_cache_reader, precompute_stage3_teacher_cache


def main() -> None:
    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cache_dir = Path("runs") / "teacher_cache_smoke"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    model = SilexCodeT18_6B_R64(device="cuda")
    indices = [30_000_000, 30_000_001]
    precompute_stage3_teacher_cache(model, indices, str(cache_dir))
    reader = open_teacher_cache_reader(str(cache_dir))
    found = 0
    for idx in indices:
        logits = reader.lookup(idx)
        if logits is None:
            continue
        if tuple(logits.shape) != (511, 258):
            raise RuntimeError(f"BAD_TEACHER_SHAPE:{tuple(logits.shape)}")
        if logits.dtype is not torch.float16:
            raise RuntimeError(f"BAD_TEACHER_DTYPE:{logits.dtype}")
        found += 1
    if found == 0:
        raise RuntimeError("NO_TEACHER_LOGITS_WRITTEN")
    print("teacher_cache_smoke_test=PASS", flush=True)


if __name__ == "__main__":
    main()
