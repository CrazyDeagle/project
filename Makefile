# SilexCode developer entry points.
# Run `make help` to list available targets.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' \
	      $(MAKEFILE_LIST)

.PHONY: install
install:  ## Editable install with the CUDA extension.
	$(PIP) install -e . --no-build-isolation

.PHONY: install-dev
install-dev:  ## Install dev dependencies (ruff, mypy, pytest, pre-commit).
	$(PIP) install -e ".[dev]" --no-build-isolation
	pre-commit install

.PHONY: lint
lint:  ## Run ruff lint + format check.
	ruff check .
	ruff format --check .

.PHONY: format
format:  ## Apply ruff formatter and autofixable lints.
	ruff format .
	ruff check --fix .

.PHONY: typecheck
typecheck:  ## Run mypy on the silexcode package.
	mypy silexcode

.PHONY: test
test:  ## Run the pytest suite (CPU-safe tests only).
	$(PYTEST) -q -m "not cuda"

.PHONY: test-all
test-all:  ## Run the full pytest suite, including CUDA-marked tests.
	$(PYTEST) -q

.PHONY: smoke
smoke:  ## Run the smoke-test scripts (no GPU writes).
	$(PYTHON) -u curriculum_smoke_test.py
	$(PYTHON) -u teacher_cache_smoke_test.py
	$(PYTHON) -u ssd_smoke_test.py

.PHONY: clean
clean:  ## Remove build artefacts and caches.
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: docker
docker:  ## Build the project Docker image.
	docker build -t silexcode:dev -f Dockerfile .
