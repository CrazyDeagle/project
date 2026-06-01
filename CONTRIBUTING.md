# Contributing to SilexCode

Thanks for taking the time to contribute. This document describes the workflow
and the quality bar expected for changes.

## Ground Rules

- Be civil. The [Code of Conduct](CODE_OF_CONDUCT.md) applies in every space
  related to this project.
- Discuss large changes in an issue before opening a pull request.
- Keep PRs focused — one logical change per PR. Refactors and feature work
  should not share a commit.

## Development Environment

See [docs/development.md](docs/development.md) for the full setup. Quick path:

```bash
python -m venv .venv
. .venv/Scripts/activate   # PowerShell: . .venv\Scripts\Activate.ps1
pip install -e . --no-build-isolation
pip install -r requirements-dev.txt
pre-commit install
```

The CUDA extension is required for runtime training but **not** required for
running the lint/static-analysis tests in CI.

## Branches and Commits

- Branch from `main`. Use short, descriptive names: `fix/checkpoint-roundtrip`,
  `feat/output-adapter-sgd`.
- Write commit messages in the imperative mood:
  *"Add output adapter SGD update"*, not *"Added"* or *"Adds"*.
- Keep the subject line under 72 characters. Wrap the body at 80.

## Tests

Every behavioural change should ship with a test. Run the suite before
pushing:

```bash
python -m pytest -q
```

The CUDA-dependent tests are skipped automatically when no GPU is present, so
CPU contributors can still iterate.

## Style

Python code is linted with [Ruff](https://docs.astral.sh/ruff/) and formatted
with `ruff format`. Both are wired into the pre-commit hook and into CI:

```bash
ruff check .
ruff format --check .
```

C++/CUDA sources follow `clang-format` with the project `.clang-format` file
once it is added; until then, match the surrounding style.

## Pull Request Checklist

- [ ] The PR description explains *what* changed and *why*.
- [ ] Tests are added or updated.
- [ ] `ruff check` and `pytest -q` both pass locally.
- [ ] The [CHANGELOG](CHANGELOG.md) has an entry under *Unreleased*.
- [ ] No unrelated reformatting churn.

## Reporting Bugs

Use the bug report template under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/). Please include:

- The exact command you ran and the full traceback.
- The output of `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- Your GPU model and driver version (`nvidia-smi`).

## Proposing Features

Open a *Feature request* issue first. For larger architectural changes,
consider drafting an ADR under `docs/adr/` (see ADR-0001 for the template).
