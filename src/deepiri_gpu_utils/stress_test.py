"""Bounded GPU/CPU stress testing with read-only telemetry sampling.

Runs a time-limited workload and samples inventory in parallel. Uses PyTorch
when the optional ``[torch]`` extra is installed; otherwise falls back to a
stdlib CPU numeric loop. A ``probes`` mode hammers read-only detect/inventory
calls for CI smoke testing without compute load.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from .detect import detect
from .doctor import doctor
from .inventory import gpu_inventory
from .torch_device import resolve_torch_device

StressMode = Literal["compute", "probes"]
StressBackend = Literal["auto", "cuda", "mps", "cpu", "probes"]

_MAX_DURATION_S = 120.0
_DEFAULT_DURATION_S = 5.0
_DEFAULT_MATRIX_SIZE = 1024
_MAX_MATRIX_SIZE = 4096


@dataclass(frozen=True)
class TelemetrySample:
    """One read-only GPU telemetry snapshot during stress."""

    elapsed_s: float
    gpu_utilization_percent: int | None
    memory_used_gb: float | None
    memory_total_gb: float | None


@dataclass(frozen=True)
class StressTestResult:
    """Outcome of a bounded stress run."""

    mode: StressMode
    backend: str
    device: str
    duration_requested_s: float
    duration_actual_s: float
    iterations: int
    ops_per_sec: float
    matrix_size: int | None
    telemetry_samples: list[TelemetrySample] = field(default_factory=list)
    peak_utilization_percent: int | None = None
    notes: list[str] = field(default_factory=list)
    success: bool = True


def _clamp_duration(seconds: float) -> float:
    return max(0.5, min(float(seconds), _MAX_DURATION_S))


def _inventory_telemetry() -> TelemetrySample:
    inv = gpu_inventory()
    util_values: list[int] = []
    used_values: list[float] = []
    total_values: list[float] = []

    for gpu in inv.gpus:
        if gpu.utilization_percent is not None:
            util_values.append(gpu.utilization_percent)
        total = float(gpu.memory_gb or 0)
        if total > 0:
            total_values.append(total)
            free = gpu.details.get("memory_free_gb") if isinstance(gpu.details, dict) else None
            if isinstance(free, (int, float)):
                used_values.append(max(total - float(free), 0.0))

    util = max(util_values) if util_values else None
    used = max(used_values) if used_values else None
    total = max(total_values) if total_values else None
    return TelemetrySample(
        elapsed_s=0.0,
        gpu_utilization_percent=util,
        memory_used_gb=round(used, 2) if used is not None else None,
        memory_total_gb=round(total, 2) if total is not None else None,
    )


def _sample_telemetry(elapsed_s: float) -> TelemetrySample:
    base = _inventory_telemetry()
    return TelemetrySample(
        elapsed_s=round(elapsed_s, 2),
        gpu_utilization_percent=base.gpu_utilization_percent,
        memory_used_gb=base.memory_used_gb,
        memory_total_gb=base.memory_total_gb,
    )


def _run_cpu_stress(duration_s: float) -> tuple[int, float]:
    """Stdlib CPU numeric loop (no torch required)."""

    start = time.perf_counter()
    iterations = 0
    x = 1.0001
    while time.perf_counter() - start < duration_s:
        for _ in range(10_000):
            x = math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
        iterations += 1
    elapsed = time.perf_counter() - start
    return iterations, elapsed


def _run_torch_stress(device: str, duration_s: float, matrix_size: int) -> tuple[int, float]:
    import torch

    dev = torch.device(device)
    size = max(64, min(matrix_size, _MAX_MATRIX_SIZE))
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)
    start = time.perf_counter()
    iterations = 0
    try:
        while time.perf_counter() - start < duration_s:
            _ = a @ b
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            iterations += 1
    finally:
        del a, b
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed = time.perf_counter() - start
    return iterations, elapsed


def _resolve_device(backend: StressBackend) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    if backend == "probes":
        return "probes", "probes", notes

    decision = resolve_torch_device("auto" if backend == "auto" else backend)  # type: ignore[arg-type]
    device = decision.device
    detected = detect().backend

    if backend == "auto":
        resolved = device
    elif backend == "cuda":
        resolved = "cuda" if decision.torch_available else "cpu"
        if resolved != "cuda":
            notes.append("torch CUDA unavailable; falling back to CPU stress.")
    elif backend == "mps":
        resolved = "mps" if decision.torch_available and device == "mps" else "cpu"
        if resolved != "mps":
            notes.append("torch MPS unavailable; falling back to CPU stress.")
    else:
        resolved = "cpu"

    return detected, resolved, notes


def _run_with_sampling(
    worker: Callable[[], tuple[int, float]],
) -> tuple[int, float, list[TelemetrySample]]:
    samples = [_sample_telemetry(0.0)]
    iterations, elapsed = worker()
    samples.append(_sample_telemetry(elapsed))
    return iterations, elapsed, samples


def _probe_stress(duration_s: float) -> tuple[int, float, list[TelemetrySample]]:
    start = time.perf_counter()
    iterations = 0
    samples: list[TelemetrySample] = [_sample_telemetry(0.0)]
    next_sample = start + 1.0
    while time.perf_counter() - start < duration_s:
        detect()
        gpu_inventory()
        doctor()
        iterations += 1
        now = time.perf_counter()
        if now >= next_sample:
            samples.append(_sample_telemetry(now - start))
            next_sample += 1.0
    elapsed = time.perf_counter() - start
    samples.append(_sample_telemetry(elapsed))
    return iterations, elapsed, samples


def _peak_util(samples: list[TelemetrySample]) -> int | None:
    values = [s.gpu_utilization_percent for s in samples if s.gpu_utilization_percent is not None]
    return max(values) if values else None


def run_stress_test(
    *,
    duration_s: float = _DEFAULT_DURATION_S,
    mode: StressMode = "compute",
    backend: StressBackend = "auto",
    matrix_size: int = _DEFAULT_MATRIX_SIZE,
) -> StressTestResult:
    """Run a bounded stress test and return structured results."""

    duration = _clamp_duration(duration_s)
    notes: list[str] = []
    detected, device, resolve_notes = _resolve_device(backend)
    notes.extend(resolve_notes)

    matrix = max(128, min(int(matrix_size), _MAX_MATRIX_SIZE))
    iterations = 0
    elapsed = 0.0
    samples: list[TelemetrySample] = []

    if mode == "probes" or backend == "probes":
        iterations, elapsed, samples = _probe_stress(duration)
        ops = iterations / elapsed if elapsed > 0 else 0.0
        return StressTestResult(
            mode="probes",
            backend=detected,
            device="probes",
            duration_requested_s=duration,
            duration_actual_s=round(elapsed, 3),
            iterations=iterations,
            ops_per_sec=round(ops, 2),
            matrix_size=None,
            telemetry_samples=samples,
            peak_utilization_percent=_peak_util(samples),
            notes=notes + ["Read-only probe loop (detect/inventory/doctor)."],
        )

    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None and device in ("cuda", "mps"):
        try:
            iterations, elapsed, samples = _run_with_sampling(
                lambda: _run_torch_stress(device, duration, matrix),
            )
            notes.append(f"PyTorch compute stress on device={device!r}.")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"PyTorch stress failed ({exc}); using CPU fallback.")
            iterations, elapsed, samples = _run_with_sampling(
                lambda: _run_cpu_stress(duration),
            )
            device = "cpu"
    else:
        if device != "cpu":
            notes.append("torch not installed; using stdlib CPU numeric stress.")
        iterations, elapsed, samples = _run_with_sampling(lambda: _run_cpu_stress(duration))
        device = "cpu"

    ops = iterations / elapsed if elapsed > 0 else 0.0
    return StressTestResult(
        mode="compute",
        backend=detected,
        device=device,
        duration_requested_s=duration,
        duration_actual_s=round(elapsed, 3),
        iterations=iterations,
        ops_per_sec=round(ops, 2),
        matrix_size=matrix,
        telemetry_samples=samples,
        peak_utilization_percent=_peak_util(samples),
        notes=notes,
    )


def render_stress_text(result: StressTestResult) -> str:
    """Human-readable stress summary."""

    lines = [
        f"stress: mode={result.mode} backend={result.backend} device={result.device}",
        (
            f"  duration={result.duration_actual_s}s "
            f"iterations={result.iterations} ops/s={result.ops_per_sec}"
        ),
    ]
    if result.matrix_size is not None:
        lines.append(f"  matrix_size={result.matrix_size}")
    if result.peak_utilization_percent is not None:
        lines.append(f"  peak_gpu_util={result.peak_utilization_percent}%")
    if result.telemetry_samples:
        last = result.telemetry_samples[-1]
        if last.memory_used_gb is not None and last.memory_total_gb is not None:
            lines.append(f"  vram_used={last.memory_used_gb}/{last.memory_total_gb} GB")
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
