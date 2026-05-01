from __future__ import annotations

import os

import pytest
import torch

from silexcode.checkpoint import export_silex_checkpoint, import_silex_checkpoint
from silexcode.model import SilexCodeT18_6B_R64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(os.environ.get("SILEX_RUN_FULL_CHECKPOINT_TEST") != "1", reason="full checkpoint roundtrip is opt-in")
def test_full_silex_checkpoint_roundtrip(tmp_path) -> None:
    path = tmp_path / "roundtrip.silex"
    model = SilexCodeT18_6B_R64(device="cuda")
    expected = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    export_silex_checkpoint(model, path)
    del model
    torch.cuda.empty_cache()

    loaded = SilexCodeT18_6B_R64(device="cuda")
    meta = import_silex_checkpoint(loaded, path)
    assert meta == {"version": 1, "has_kfac": False}
    assert loaded.deterministic_backbone is True
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor.detach().cpu(), expected[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_checkpoint_loaded_backbone_uses_packed_native_runtime() -> None:
    model = SilexCodeT18_6B_R64(device="cuda")
    model.mark_checkpoint_backbone_loaded()
    assert model.deterministic_backbone is False
    assert model.use_native_runtime is True
    for layer in model.layers[:2]:
        assert layer.w_i.deterministic is False
        assert layer.w_c.deterministic is False
