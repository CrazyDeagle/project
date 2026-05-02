from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .constants import s5
from .model import SilexCodeT18_6B_R64, TLinear, TernaryEmbedding


SILEX_MAGIC = "SILEXCODE_T18_6B_R64"
SILEX_PLASTIC_MAGIC = "SILEXCODE_T18_6B_R64_PLASTIC"
SILEX_VERSION = 1


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def load_tlinear_from_checkpoint(module: TLinear, root: str | Path, name: str) -> None:
    root = Path(root)
    wpack = _load_tensor(root / f"{name}.wpack.pt")
    alpha = _load_tensor(root / f"{name}.alpha.pt")
    expected_w = (module.d_out, s5(module.d_in))
    if wpack.dtype is not torch.uint8 or tuple(wpack.shape) != expected_w:
        raise ValueError(f"{name}.wpack.pt must be uint8 with shape {expected_w}")
    if alpha.dtype is not torch.bfloat16 or tuple(alpha.shape) != (module.d_out,):
        raise ValueError(f"{name}.alpha.pt must be bf16 with shape {(module.d_out,)}")
    module.wpack.copy_(wpack.to(device=module.wpack.device, non_blocking=True))
    module.alpha.copy_(alpha.to(device=module.alpha.device, non_blocking=True))


def load_embedding_from_checkpoint(module: TernaryEmbedding, root: str | Path) -> None:
    root = Path(root)
    wpack = _load_tensor(root / "embedding.wpack.pt")
    alpha = _load_tensor(root / "embedding.alpha.pt")
    expected_w = (258, s5(4096))
    if wpack.dtype is not torch.uint8 or tuple(wpack.shape) != expected_w:
        raise ValueError(f"embedding.wpack.pt must be uint8 with shape {expected_w}")
    if alpha.dtype is not torch.bfloat16 or tuple(alpha.shape) != (258,):
        raise ValueError("embedding.alpha.pt must be bf16 with shape (258,)")
    module.wpack.copy_(wpack.to(device=module.wpack.device, non_blocking=True))
    module.alpha.copy_(alpha.to(device=module.alpha.device, non_blocking=True))


def load_silex_checkpoint(model: SilexCodeT18_6B_R64, root: str | Path) -> None:
    root = Path(root)
    load_embedding_from_checkpoint(model.embedding, root)
    for idx, layer in enumerate(model.layers, start=1):
        prefix = f"layers.{idx:02d}"
        for attr in ("w_i", "w_f", "w_v", "w_r", "w_o", "w_a", "w_b", "w_c"):
            load_tlinear_from_checkpoint(getattr(layer, attr), root, f"{prefix}.{attr}")
    load_tlinear_from_checkpoint(model.reasoner.w_z1, root, "reasoner.w_z1")
    load_tlinear_from_checkpoint(model.reasoner.w_z2, root, "reasoner.w_z2")
    load_tlinear_from_checkpoint(model.reasoner.w_z3, root, "reasoner.w_z3")
    model.mark_checkpoint_backbone_loaded()


def _tensor_manifest(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": str(t.dtype), "shape": list(t.shape)}
        for name, t in sorted(tensors.items())
    }


def _output_adapter_metadata(model: SilexCodeT18_6B_R64) -> dict[str, Any]:
    enabled = bool(getattr(model, "output_adapter_enabled", False))
    return {
        "enabled": enabled,
        "rank": int(getattr(model, "output_adapter_rank", 0)) if enabled else 0,
    }


def _validate_output_adapter_metadata(model: SilexCodeT18_6B_R64, meta: dict[str, Any] | None, *, plastic_names: set[str] | None = None) -> None:
    if not isinstance(meta, dict):
        meta = None
    plastic_has_output = bool(plastic_names and any(name.startswith("output_adapter_") for name in plastic_names))
    enabled = bool((meta or {}).get("enabled", plastic_has_output))
    if not enabled and not plastic_has_output:
        return
    if not bool(getattr(model, "output_adapter_enabled", False)):
        raise ValueError("SILEX_OUTPUT_ADAPTER_CHECKPOINT_REQUIRES_ENABLE_OUTPUT_ADAPTER")
    rank = int((meta or {}).get("rank", getattr(model, "output_adapter_rank", 0)))
    if rank != int(getattr(model, "output_adapter_rank", 0)):
        raise ValueError(f"SILEX_OUTPUT_ADAPTER_RANK_MISMATCH:{rank}!={model.output_adapter_rank}")


def _validate_state_dict(model: SilexCodeT18_6B_R64, state: dict[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing:
        raise ValueError(f"SILEX_CHECKPOINT_MISSING_TENSORS: {missing[:8]}")
    if extra:
        raise ValueError(f"SILEX_CHECKPOINT_EXTRA_TENSORS: {extra[:8]}")
    for name, ref in expected.items():
        got = state[name]
        if got.dtype != ref.dtype:
            raise ValueError(f"SILEX_CHECKPOINT_DTYPE_MISMATCH:{name}:{got.dtype}!={ref.dtype}")
        if tuple(got.shape) != tuple(ref.shape):
            raise ValueError(f"SILEX_CHECKPOINT_SHAPE_MISMATCH:{name}:{tuple(got.shape)}!={tuple(ref.shape)}")


def _collect_kfac_state(kfac_optimizer) -> dict[str, dict[str, torch.Tensor]] | None:
    if kfac_optimizer is None or not hasattr(kfac_optimizer, "state"):
        return None
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, state in kfac_optimizer.state.items():
        out[name] = {
            "a_cov": state.a_cov.detach().cpu(),
            "g_cov": state.g_cov.detach().cpu(),
            "a_inv": state.a_inv.detach().cpu(),
            "g_inv": state.g_inv.detach().cpu(),
        }
    return out


def _plastic_state_dict(model: SilexCodeT18_6B_R64) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def export_silex_checkpoint(
    model: SilexCodeT18_6B_R64,
    path: str | Path,
    *,
    kfac_optimizer=None,
    include_kfac: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    kfac_state = _collect_kfac_state(kfac_optimizer) if include_kfac else None
    payload = {
        "magic": SILEX_MAGIC,
        "version": SILEX_VERSION,
        "model_name": model.name,
        "config": vars(model.config),
        "deterministic_backbone": bool(model.deterministic_backbone),
        "output_adapter": _output_adapter_metadata(model),
        "manifest": _tensor_manifest(tensors),
        "state_dict": tensors,
        "kfac_state": kfac_state,
    }
    torch.save(payload, path)


def import_silex_checkpoint(
    model: SilexCodeT18_6B_R64,
    path: str | Path,
    *,
    kfac_optimizer=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("SILEX_CHECKPOINT_NOT_A_CONTAINER")
    if payload.get("magic") != SILEX_MAGIC:
        raise ValueError("SILEX_CHECKPOINT_BAD_MAGIC")
    if payload.get("version") != SILEX_VERSION:
        raise ValueError("SILEX_CHECKPOINT_UNSUPPORTED_VERSION")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("SILEX_CHECKPOINT_MISSING_STATE_DICT")
    _validate_output_adapter_metadata(model, payload.get("output_adapter"))
    _validate_state_dict(model, state)
    model.load_state_dict(
        {name: tensor.to(device=model.gamma_out.device, non_blocking=True) for name, tensor in state.items()},
        strict=True,
    )
    if bool(payload.get("deterministic_backbone", False)):
        model.deterministic_backbone = True
        model.use_native_runtime = True
    else:
        model.mark_checkpoint_backbone_loaded()

    kfac_state = payload.get("kfac_state")
    if kfac_optimizer is not None and kfac_state is not None:
        for name, src in kfac_state.items():
            if name not in kfac_optimizer.state:
                raise ValueError(f"SILEX_CHECKPOINT_UNKNOWN_KFAC_PARAM:{name}")
            dst = kfac_optimizer.state[name]
            dst.a_cov.copy_(src["a_cov"].to(device=dst.a_cov.device))
            dst.g_cov.copy_(src["g_cov"].to(device=dst.g_cov.device))
            dst.a_inv.copy_(src["a_inv"].to(device=dst.a_inv.device))
            dst.g_inv.copy_(src["g_inv"].to(device=dst.g_inv.device))
    return {"version": payload["version"], "has_kfac": kfac_state is not None}


def export_plastic_checkpoint(
    model: SilexCodeT18_6B_R64,
    path: str | Path,
    *,
    kfac_optimizer=None,
    include_kfac: bool = False,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plastic = _plastic_state_dict(model)
    meta = dict(metadata or {})
    adapter_meta = _output_adapter_metadata(model)
    if (adapter_meta["enabled"] or "output_adapter" in meta) and not isinstance(meta.get("output_adapter"), dict):
        meta["output_adapter"] = _output_adapter_metadata(model)
    payload = {
        "magic": SILEX_PLASTIC_MAGIC,
        "version": SILEX_VERSION,
        "model_name": model.name,
        "metadata": meta,
        "manifest": _tensor_manifest(plastic),
        "plastic_state": plastic,
        "kfac_state": _collect_kfac_state(kfac_optimizer) if include_kfac else None,
    }
    torch.save(payload, path)


def import_plastic_checkpoint(
    model: SilexCodeT18_6B_R64,
    path: str | Path,
    *,
    kfac_optimizer=None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or payload.get("magic") != SILEX_PLASTIC_MAGIC:
        raise ValueError("SILEX_PLASTIC_CHECKPOINT_BAD_MAGIC")
    if payload.get("version") != SILEX_VERSION:
        raise ValueError("SILEX_PLASTIC_CHECKPOINT_UNSUPPORTED_VERSION")
    plastic = payload.get("plastic_state")
    if not isinstance(plastic, dict):
        raise ValueError("SILEX_PLASTIC_CHECKPOINT_MISSING_STATE")
    metadata = payload.get("metadata", {})
    _validate_output_adapter_metadata(
        model,
        metadata.get("output_adapter") if isinstance(metadata, dict) else None,
        plastic_names=set(plastic),
    )
    params = dict(model.named_parameters())
    for name, tensor in plastic.items():
        if name not in params:
            raise ValueError(f"SILEX_PLASTIC_CHECKPOINT_UNKNOWN_PARAM:{name}")
        param = params[name]
        if not param.requires_grad:
            raise ValueError(f"SILEX_PLASTIC_CHECKPOINT_NON_PLASTIC_PARAM:{name}")
        if tensor.dtype != param.dtype or tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"SILEX_PLASTIC_CHECKPOINT_PARAM_MISMATCH:{name}")
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype, non_blocking=True))

    kfac_state = payload.get("kfac_state")
    if kfac_optimizer is not None and kfac_state is not None:
        for name, src in kfac_state.items():
            if name not in kfac_optimizer.state:
                raise ValueError(f"SILEX_PLASTIC_CHECKPOINT_UNKNOWN_KFAC_PARAM:{name}")
            dst = kfac_optimizer.state[name]
            dst.a_cov.copy_(src["a_cov"].to(device=dst.a_cov.device))
            dst.g_cov.copy_(src["g_cov"].to(device=dst.g_cov.device))
            dst.a_inv.copy_(src["a_inv"].to(device=dst.a_inv.device))
            dst.g_inv.copy_(src["g_inv"].to(device=dst.g_inv.device))
    return {
        "version": payload["version"],
        "has_kfac": kfac_state is not None,
        "metadata": payload.get("metadata", {}),
    }
