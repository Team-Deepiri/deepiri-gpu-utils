"""Lock down that every public module imports cleanly and exposes ``__version__``.

Regression guard: importing any submodule must not require GPU hardware,
optional extras (``torch``), or network access.
"""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "deepiri_gpu_utils",
    "deepiri_gpu_utils._version",
    "deepiri_gpu_utils.build_args",
    "deepiri_gpu_utils.cli",
    "deepiri_gpu_utils.compose_gpu",
    "deepiri_gpu_utils.detect",
    "deepiri_gpu_utils.doctor",
    "deepiri_gpu_utils.env_hints",
    "deepiri_gpu_utils.export_env",
    "deepiri_gpu_utils.gpu_top",
    "deepiri_gpu_utils.hardware",
    "deepiri_gpu_utils.health",
    "deepiri_gpu_utils.install_check",
    "deepiri_gpu_utils.inventory",
    "deepiri_gpu_utils.profiles",
    "deepiri_gpu_utils.setup",
    "deepiri_gpu_utils.snapshot",
    "deepiri_gpu_utils.stress_test",
    "deepiri_gpu_utils.summary",
    "deepiri_gpu_utils.system_info",
    "deepiri_gpu_utils.torch_device",
    "deepiri_gpu_utils.visualize",
]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_version_is_exposed_and_consistent() -> None:
    import deepiri_gpu_utils
    from deepiri_gpu_utils import _version

    assert isinstance(deepiri_gpu_utils.__version__, str)
    assert deepiri_gpu_utils.__version__
    assert deepiri_gpu_utils.__version__ == _version.__version__
    assert all(part != "" for part in deepiri_gpu_utils.__version__.split("."))
