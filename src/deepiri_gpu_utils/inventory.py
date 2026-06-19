"""Read-only GPU inventory and suitability helpers.

This module reports what GPUs are visible through tools that are already on the
host (``nvidia-smi``, ``rocm-smi``) plus the platform signal used by
:func:`deepiri_gpu_utils.detect.detect`. It is intentionally lightweight and
strictly read-only:

* It never mutates the environment (e.g. ``CUDA_VISIBLE_DEVICES``).
* It never hides, selects, or reconfigures hardware.
* It never crashes on missing tools or malformed output — callers always get a
  well-formed result, empty on CPU-only / no-GPU hosts.

It reuses the existing safe subprocess runner and ROCm probe rather than adding
new dependencies or duplicating subprocess handling.
"""

from __future__ import annotations

import csv
import io
import platform
import shutil
from dataclasses import dataclass, field

from . import detect
from ._subprocess import run_text


@dataclass(frozen=True)
class GPUInfo:
    """A single GPU as reported by an available read-only source."""

    backend: str
    index: int | None = None
    name: str | None = None
    memory_mib: int | None = None
    memory_gb: float | None = None
    utilization_percent: int | None = None
    driver_version: str | None = None
    source: str = "unknown"
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GPUInventoryResult:
    """All GPUs discovered, plus any non-fatal warnings and the winning source."""

    gpus: list[GPUInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str | None = None


@dataclass(frozen=True)
class GPUSelection:
    """Outcome of a VRAM-threshold suitability check (read-only; selects nothing)."""

    selected: GPUInfo | None
    suitable: bool
    reason: str
    candidates: list[GPUInfo] = field(default_factory=list)
    min_memory_gb: float = 0.0


_NVIDIA_QUERY = "index,name,memory.total,memory.free,utilization.gpu,driver_version"


def _safe_int(value: str) -> int | None:
    """Parse an int from a possibly-noisy nvidia-smi cell (``[N/A]`` -> None)."""

    v = value.strip()
    if not v or v.startswith("["):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _parse_nvidia_inventory_row(line: str) -> GPUInfo | None:
    """Parse one nvidia-smi CSV row; return None if it cannot describe a GPU."""

    reader = csv.reader(io.StringIO(line.strip()))
    try:
        row = next(reader)
    except StopIteration:
        return None
    cells = [c.strip() for c in row]
    # Need at least an index slot and a non-empty name to describe a GPU.
    if len(cells) < 2 or not cells[1]:
        return None
    while len(cells) < 6:
        cells.append("")

    memory_mib = _safe_int(cells[2])
    memory_free_mib = _safe_int(cells[3])

    details: dict[str, object] = {}
    if memory_free_mib is not None:
        details["memory_free_mib"] = memory_free_mib
        details["memory_free_gb"] = round(memory_free_mib / 1024.0, 2)

    return GPUInfo(
        backend="cuda",
        index=_safe_int(cells[0]),
        name=cells[1] or None,
        memory_mib=memory_mib,
        memory_gb=round(memory_mib / 1024.0, 2) if memory_mib is not None else None,
        utilization_percent=_safe_int(cells[4]),
        driver_version=cells[5].strip() or None,
        source="nvidia-smi",
        details=details,
    )


def _nvidia_inventory() -> tuple[list[GPUInfo], list[str]]:
    """Enumerate all NVIDIA GPUs via nvidia-smi; empty when unavailable."""

    if not shutil.which("nvidia-smi"):
        return [], []
    res = run_text(
        ["nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
        timeout=15,
    )
    if not res.ok:
        return [], []

    gpus: list[GPUInfo] = []
    warnings: list[str] = []
    for raw_line in res.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        gpu = _parse_nvidia_inventory_row(line)
        if gpu is None:
            warnings.append(f"Ignored unparsable nvidia-smi row: {line!r}")
        else:
            gpus.append(gpu)
    return gpus, warnings


def _rocm_product_name(raw: str) -> str | None:
    """Best-effort product name from ``rocm-smi --showproductname`` text."""

    for line in raw.splitlines():
        if "card series" in line.lower() and ":" in line:
            name = line.split(":")[-1].strip()
            if name:
                return name
    return None


def gpu_inventory() -> GPUInventoryResult:
    """Return all GPUs visible from available read-only tools.

    Precedence mirrors :func:`detect.detect`: NVIDIA, then ROCm, then Apple MPS,
    then an empty inventory for CPU-only hosts. Never raises.
    """

    warnings: list[str] = []

    gpus, nv_warnings = _nvidia_inventory()
    warnings.extend(nv_warnings)
    if gpus:
        return GPUInventoryResult(gpus=gpus, warnings=warnings, source="nvidia-smi")

    rocm = detect.query_rocm_smi()
    if rocm is not None:
        raw = str(rocm.get("raw", "")) if isinstance(rocm, dict) else ""
        gpu = GPUInfo(
            backend="rocm",
            name=_rocm_product_name(raw),
            source="rocm-smi",
            details={"raw": raw[:500]} if raw else {},
        )
        return GPUInventoryResult(gpus=[gpu], warnings=warnings, source="rocm-smi")

    if platform.system() == "Darwin":
        gpu = GPUInfo(
            backend="mps",
            name="Apple MPS",
            source="platform",
            details={"platform": platform.system(), "machine": platform.machine()},
        )
        return GPUInventoryResult(gpus=[gpu], warnings=warnings, source="platform")

    warnings.append(
        "No GPU inventory source found (nvidia-smi/rocm-smi unavailable or reported no GPUs); "
        "treating as CPU-only."
    )
    return GPUInventoryResult(gpus=[], warnings=warnings, source=None)


def _memory_metric_gb(gpu: GPUInfo) -> float | None:
    """Free VRAM (GB) when known, else total VRAM (GB), else None."""

    free = gpu.details.get("memory_free_gb") if isinstance(gpu.details, dict) else None
    if isinstance(free, (int, float)):
        return float(free)
    if gpu.memory_gb is not None:
        return float(gpu.memory_gb)
    return None


def choose_suitable_gpu(min_memory_gb: float, *, backend: str | None = None) -> GPUSelection:
    """Report whether any detected GPU satisfies a minimum VRAM threshold.

    Read-only: ranks candidates by free VRAM (falling back to total VRAM) and
    reports the best match without touching the environment or hiding GPUs.
    Returns ``suitable=False`` (never raises) on CPU-only / no-GPU hosts.
    """

    inventory = gpu_inventory()
    candidates = list(inventory.gpus)
    if backend is not None:
        candidates = [g for g in candidates if g.backend == backend]

    if not candidates:
        if backend is not None:
            reason = f"No GPUs found for backend={backend!r}."
        else:
            reason = "No GPUs detected; CPU-only or no GPU tooling available."
        return GPUSelection(
            selected=None,
            suitable=False,
            reason=reason,
            candidates=candidates,
            min_memory_gb=min_memory_gb,
        )

    measured = [(g, _memory_metric_gb(g)) for g in candidates]
    qualifying = [(g, m) for g, m in measured if m is not None and m >= min_memory_gb]
    if qualifying:
        best = max(qualifying, key=lambda pair: pair[1])[0]
        best_mem = _memory_metric_gb(best)
        reason = (
            f"{best.name or 'GPU'} has ~{best_mem} GB available "
            f"(>= {min_memory_gb} GB minimum)."
        )
        return GPUSelection(
            selected=best,
            suitable=True,
            reason=reason,
            candidates=candidates,
            min_memory_gb=min_memory_gb,
        )

    known = [m for _, m in measured if m is not None]
    if not known:
        reason = (
            f"GPU memory not reported by available tooling; "
            f"cannot confirm {min_memory_gb} GB minimum."
        )
    else:
        reason = (
            f"No GPU meets the {min_memory_gb} GB minimum "
            f"(best available ~{max(known)} GB)."
        )
    return GPUSelection(
        selected=None,
        suitable=False,
        reason=reason,
        candidates=candidates,
        min_memory_gb=min_memory_gb,
    )
