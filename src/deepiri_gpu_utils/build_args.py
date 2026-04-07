from __future__ import annotations

import re
from dataclasses import dataclass, field

from .detect import DetectResult, detect

# Aligned with diri-cyrex/scripts/utils/detect_gpu.sh and Dockerfile defaults.
DEFAULT_PYTORCH_GPU_IMAGE = "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"
DEFAULT_CPU_IMAGE = "python:3.11-slim"
DEFAULT_PYTHON_VERSION = "3.11"
DEFAULT_CUDA_VERSION_GPU = "12.8"
_MIN_GPU_MEMORY_GB = 4

_SLIM_CPU_RE = re.compile(r"cpu|slim", re.IGNORECASE)
_MPS_IMAGE_RE = re.compile(r"mps|macos|darwin", re.IGNORECASE)


@dataclass(frozen=True)
class BuildArgs:
    """Docker build arguments for Cyrex-style hybrid images."""

    device_type: str
    """Resolved ``DEVICE_TYPE`` for ``docker build`` (``gpu``, ``cpu``, ``mpsos``)."""

    base_image: str
    cuda_version: str | None = None
    python_version: str = DEFAULT_PYTHON_VERSION
    build_args: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def infer_device_type_from_base_image(base_image: str) -> str:
    """Mirror ``diri-cyrex/Dockerfile`` logic when ``DEVICE_TYPE=auto``."""

    if _SLIM_CPU_RE.search(base_image):
        return "cpu"
    if _MPS_IMAGE_RE.search(base_image):
        return "mpsos"
    return "gpu"


def _base_and_device_for_result(dr: DetectResult) -> tuple[str, str, str | None, list[str]]:
    """Return ``(base_image, device_type, cuda_version, extra_warnings)``."""

    warnings: list[str] = []
    if dr.backend == "cuda":
        nv = dr.details.get("nvidia") or {}
        meets = bool(nv.get("meets_min_vram", False))
        if meets:
            return DEFAULT_PYTORCH_GPU_IMAGE, "gpu", DEFAULT_CUDA_VERSION_GPU, warnings
        warnings.append(
            f"NVIDIA GPU present but VRAM below {_MIN_GPU_MEMORY_GB}GB; "
            "using CPU base image (matches detect_gpu.sh)."
        )
        return DEFAULT_CPU_IMAGE, "cpu", None, warnings

    if dr.backend == "mps":
        # Dockerfile maps ``python:*-slim`` to ``cpu`` when DEVICE_TYPE=auto; pass mpsos explicitly.
        return DEFAULT_CPU_IMAGE, "mpsos", None, warnings

    if dr.backend == "cpu":
        return DEFAULT_CPU_IMAGE, "cpu", None, warnings

    if dr.backend == "rocm":
        warnings.append(
            "ROCm: Docker base image policy is not finalized; using CPU slim base. "
            "Override with --device-type gpu when your pipeline supports ROCm images."
        )
        return DEFAULT_CPU_IMAGE, "cpu", None, warnings

    # Unknown / defensive
    warnings.append("Unknown detection result; defaulting to CPU base image.")
    return DEFAULT_CPU_IMAGE, "cpu", None, warnings


def _explicit_base_and_device(device_type: str) -> tuple[str, str, str | None]:
    dt = device_type.lower().strip()
    if dt == "gpu":
        return DEFAULT_PYTORCH_GPU_IMAGE, "gpu", DEFAULT_CUDA_VERSION_GPU
    if dt == "cpu":
        return DEFAULT_CPU_IMAGE, "cpu", None
    if dt == "mpsos":
        return DEFAULT_CPU_IMAGE, "mpsos", None
    raise ValueError(f"Unsupported device_type: {device_type!r}")


def build_args_from_detection(
    *,
    device_type: str = "auto",
    detect_result: DetectResult | None = None,
) -> BuildArgs:
    """Return docker ``--build-arg`` values aligned with Cyrex Dockerfile + detect_gpu.sh.

    When ``device_type`` is ``auto`` (default), uses :func:`detect` unless
    ``detect_result`` is provided (mainly for tests).
    """

    extra_warnings: list[str]
    if device_type.lower().strip() == "auto":
        dr = detect_result if detect_result is not None else detect()
        base_image, resolved_dt, cuda_ver, extra_warnings = _base_and_device_for_result(dr)
    else:
        base_image, resolved_dt, cuda_ver = _explicit_base_and_device(device_type)
        extra_warnings = []

    ba = {
        "DEVICE_TYPE": resolved_dt,
        "BASE_IMAGE": base_image,
        "BUILD_TYPE": "prebuilt",
    }

    return BuildArgs(
        device_type=resolved_dt,
        cuda_version=cuda_ver,
        python_version=DEFAULT_PYTHON_VERSION,
        base_image=base_image,
        build_args=ba,
        warnings=extra_warnings,
    )
