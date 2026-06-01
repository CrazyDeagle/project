# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Apache-2.0 license and `CITATION.cff` for academic citation.
- `pyproject.toml` with project metadata, Ruff, pytest and mypy configuration.
- GitHub Actions CI: lint, type-check, and CPU-only unit tests on push and PR.
- CodeQL static analysis workflow.
- Dependabot configuration for Python and GitHub Actions updates.
- Issue and pull request templates under `.github/`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- Pre-commit hooks (`ruff`, `ruff-format`, generic hygiene).
- `Makefile` with common developer entry points.
- `Dockerfile` and `.devcontainer/devcontainer.json` for a reproducible
  CUDA-enabled environment.
- `docs/architecture.md` and `docs/adr/` with the first architectural
  decision record.

### Changed

- `README.md` now surfaces project badges and links the docs and contribution
  guides.

## [0.1.0] - 2025-01-01

Initial development version. Curriculum, bootstrap and accelerated trainers,
native CUDA extension, K-FAC-style updates, and checkpoint roundtrip.

[Unreleased]: https://github.com/CrazyDeagle/project/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CrazyDeagle/project/releases/tag/v0.1.0
