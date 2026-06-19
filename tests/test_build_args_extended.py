"""Regression tests covering every ``build_args`` backend path.

The existing ``test_build_args.py`` covers gpu/mpsos/auto-cuda; this adds the
cpu, rocm, unknown, and explicit-override paths so all branches are locked.
"""

from __future__ import annotations

import pytest

from deepiri_gpu_utils.build_args import (
    DEFAULT_CPU_IMAGE,
    DEFAULT_PYTORCH_GPU_IMAGE,
    build_args_from_detection,
)
from deepiri_gpu_utils.detect import DetectResult


def test_auto_cpu_path() -> None:
    b = build_args_from_detection(
        device_type="auto", detect_result=DetectResult(backend="cpu")
    )
    assert b.device_type == "cpu"
    assert b.base_image == DEFAULT_CPU_IMAGE
    assert b.cuda_version is None
    assert b.warnings == []


def test_auto_rocm_falls_back_to_cpu_with_warning() -> None:
    b = build_args_from_detection(
        device_type="auto", detect_result=DetectResult(backend="rocm")
    )
    assert b.device_type == "cpu"
    assert b.base_image == DEFAULT_CPU_IMAGE
    assert any("ROCm" in w for w in b.warnings)


def test_auto_unknown_falls_back_to_cpu_with_warning() -> None:
    b = build_args_from_detection(
        device_type="auto", detect_result=DetectResult(backend="unknown")
    )
    assert b.device_type == "cpu"
    assert any("Unknown detection" in w for w in b.warnings)


def test_explicit_cpu() -> None:
    b = build_args_from_detection(device_type="cpu")
    assert b.device_type == "cpu"
    assert b.base_image == DEFAULT_CPU_IMAGE
    assert b.build_args["BUILD_TYPE"] == "prebuilt"


def test_explicit_gpu_uses_pytorch_image() -> None:
    b = build_args_from_detection(device_type="gpu")
    assert b.device_type == "gpu"
    assert b.base_image == DEFAULT_PYTORCH_GPU_IMAGE
    assert b.cuda_version == "12.8"


def test_invalid_device_type_raises() -> None:
    with pytest.raises(ValueError):
        build_args_from_detection(device_type="tpu")
