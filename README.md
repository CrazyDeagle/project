# SilexCode-T18.6B-R64

CUDA/C++ and PyTorch implementation of the fixed SilexCode-T18.6B-R64 TDD architecture.

## Requirements

- Windows Developer Command Prompt with MSVC available.
- NVIDIA CUDA toolkit matching the installed PyTorch CUDA runtime.
- PyTorch with CUDA and BF16 support.
- Editable install must use `--no-build-isolation` because `setup.py` imports PyTorch build helpers from the active environment.

## Install

From a Visual Studio Developer Command Prompt:

```bat
cd /d D:\silexcode
set CUDA_HOME=%CUDA_PATH%
set DISTUTILS_USE_SDK=1
pip install -e . --no-build-isolation
```

If the extension must be rebuilt from scratch:

```bat
del silexcode\_C*.pyd
pip install -e . --no-build-isolation
```

## Test

```powershell
python -m pytest -q
```

Full checkpoint roundtrip is large and opt-in:

```powershell
$env:SILEX_RUN_FULL_CHECKPOINT_TEST="1"
python -m pytest -q tests\test_checkpoint_roundtrip.py
Remove-Item Env:\SILEX_RUN_FULL_CHECKPOINT_TEST
```

## VRAM Stress

Deterministic fast path:

```powershell
python -u vram_stress_test.py --steps 10 --mode deterministic
```

Packed checkpoint path:

```powershell
python -u vram_stress_test.py --steps 1 --mode packed
```

## Smoke Tests

```powershell
python -u curriculum_smoke_test.py
python -u teacher_cache_smoke_test.py
python -u ssd_smoke_test.py
python -u curriculum_dry_run.py
python -u run_curriculum.py --dry-run --output-dir runs\final_dry_run
```

## Real Curriculum Run

```powershell
python -u run_curriculum.py --output-dir runs\silex_curriculum_001
```

## Checkpoint Runtime Modes

- `deterministic_backbone=True`: native runtime uses the FWHT fast path that exactly matches the deterministic TDD initialization.
- `deterministic_backbone=False`: native runtime uses packed `Wpack` kernels so arbitrary checkpoint weights are respected.
