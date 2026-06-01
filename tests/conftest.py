"""Shared pytest configuration.

Auto-skips tests that require the compiled CUDA extension (``silexcode._C``)
when it is not available — for example in CI, where the package is installed
without building the native extension. Tests can still opt in explicitly with
``@pytest.mark.cuda``; those are filtered out at the CI command line.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_EXT_AVAILABLE = importlib.util.find_spec("silexcode._C") is not None
_SKIP_CUDA_BUILD = os.environ.get("SILEX_SKIP_CUDA_BUILD", "").lower() in {"1", "true", "yes"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark any explicitly CUDA-using test as skipped when the extension is missing.

    Tests already carry their own ``torch.cuda.is_available()`` skipif guards
    where appropriate; this hook is a safety net so a missing extension does
    not produce hard errors during collection or fixture setup.
    """
    if _EXT_AVAILABLE and not _SKIP_CUDA_BUILD:
        return

    skip_cuda = pytest.mark.skip(reason="silexcode._C CUDA extension not built in this environment")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
