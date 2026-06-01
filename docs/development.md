# Development Guide

This guide describes the day-to-day developer workflow. For project policy,
see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Prerequisites

| Component              | Version              | Notes                                |
| ---------------------- | -------------------- | ------------------------------------ |
| Python                 | 3.10 – 3.12          | 3.11 is the reference version.       |
| PyTorch                | >= 2.1 with CUDA     | Must match installed CUDA toolkit.   |
| CUDA toolkit           | >= 12.1              | `nvcc` must be on `PATH`.            |
| MSVC (Windows only)    | Visual Studio 2022   | Use the Developer Command Prompt.    |
| GNU Make               | any recent           | Optional but recommended.            |

On Windows you must launch the install from a **Developer Command Prompt for
VS 2022**, otherwise `cl.exe` will not be found and the CUDA extension build
will fail with a cryptic linker error.

## First-time Setup

```bash
git clone https://github.com/CrazyDeagle/project.git silexcode
cd silexcode
python -m venv .venv

# Linux / macOS
. .venv/bin/activate
# Windows PowerShell
. .venv\Scripts\Activate.ps1

make install-dev
```

`make install-dev` does three things:

1. installs the package in editable mode with the `[dev]` extras
   (`ruff`, `mypy`, `pytest`, `pre-commit`);
2. compiles the CUDA extension into `silexcode/_C*.{pyd,so}`;
3. registers the pre-commit hooks.

## Inner Loop

```bash
make lint        # ruff check + ruff format --check
make format      # apply autoformat + autofixes
make typecheck   # mypy
make test        # CPU-only pytest
make test-all    # full pytest including CUDA-marked tests
```

The pre-commit hook runs `ruff`, `ruff-format`, and a small set of
hygiene checks on every commit. If a hook autofixes a file, just `git add`
the result and re-commit.

## Running the CUDA Extension Locally

If you change anything under `silexcode/cuda/`, you must rebuild the
extension:

```bash
rm -f silexcode/_C*.pyd silexcode/_C*.so
pip install -e . --no-build-isolation
```

The `--no-build-isolation` flag is **required** because `setup.py` imports
build helpers (`torch.utils.cpp_extension`) from the active environment.

## Debugging Tips

- Set `TORCH_CUDA_LAUNCH_BLOCKING=1` to get accurate stack traces from CUDA
  errors. Expect a real slowdown.
- Set `CUDA_VISIBLE_DEVICES=""` to force the CPU-fallback path; useful for
  rapid iteration when you're not touching kernels.
- The smoke tests under the repo root (`curriculum_smoke_test.py`,
  `ssd_smoke_test.py`, `teacher_cache_smoke_test.py`) are designed to run in
  under a minute and exercise the full pipeline. Use them before launching a
  real training run.

## Releasing

Releases are cut from `main` only. Steps:

1. Update `[project].version` in `pyproject.toml` and the `version:` field in
   `CITATION.cff`.
2. Move the *Unreleased* block of `CHANGELOG.md` under a new `## [X.Y.Z]`
   heading with today's date.
3. Open a PR titled `Release vX.Y.Z`. After it merges, tag the commit:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
4. GitHub Actions will produce the sdist artefact attached to the workflow
   run. (Wheel publishing requires a manual rebuild on a CUDA host.)
