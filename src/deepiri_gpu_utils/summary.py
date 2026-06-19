"""Aggregate hardware snapshot from detect, doctor, inventory, and build-args.

Read-only: composes existing probes into one JSON-friendly structure for CI,
compose preflight, and operator dashboards. Never mutates the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .build_args import build_args_from_detection
from .detect import DetectResult, detect
from .doctor import DoctorReport, doctor
from .inventory import GPUInventoryResult, gpu_inventory
from .ollama import OllamaRecommendation, recommend_models
from .system_info import system_ram_gb
from .torch_device import DeviceDecision, resolve_torch_device


@dataclass(frozen=True)
class HardwareSummary:
    """Single snapshot of host GPU readiness and build posture."""

    detect: DetectResult
    doctor: DoctorReport
    inventory: GPUInventoryResult
    build_args: Any
    ollama: OllamaRecommendation
    torch_device: DeviceDecision
    system_ram_gb: int = 0
    gpu_count: int = 0
    total_vram_gb: float | None = None
    notes: list[str] = field(default_factory=list)


def _total_vram_gb(inventory: GPUInventoryResult) -> float | None:
    values = [g.memory_gb for g in inventory.gpus if g.memory_gb is not None]
    if not values:
        return None
    return round(sum(values), 2)


def hardware_summary() -> HardwareSummary:
    """Return an aggregate hardware snapshot (read-only; never raises)."""

    d = detect()
    rep = doctor()
    inv = gpu_inventory()
    ba = build_args_from_detection(device_type="auto", detect_result=d)
    ollama_rec = recommend_models()
    torch_dec = resolve_torch_device("auto")
    ram = system_ram_gb()

    notes: list[str] = []
    if rep.status == "warn":
        notes.append(f"doctor status={rep.status}; see runbook for remediation hints.")
    if not inv.gpus and d.backend != "cpu":
        notes.append(f"detect backend={d.backend} but inventory reported no GPUs.")

    return HardwareSummary(
        detect=d,
        doctor=rep,
        inventory=inv,
        build_args=ba,
        ollama=ollama_rec,
        torch_device=torch_dec,
        system_ram_gb=ram,
        gpu_count=len(inv.gpus),
        total_vram_gb=_total_vram_gb(inv),
        notes=notes,
    )
