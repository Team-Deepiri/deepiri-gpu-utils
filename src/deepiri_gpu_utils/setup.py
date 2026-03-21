from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DeviceArg = Literal["auto", "nvidia", "amd", "apple", "cpu"]


@dataclass(frozen=True)
class SetupPlan:
    device: DeviceArg
    dry_run: bool = True
    runbook: list[str] = field(default_factory=list)


def setup_device(device: DeviceArg = "auto", *, dry_run: bool = True) -> SetupPlan:
    """Prepare the system for the requested device (stub)."""

    return SetupPlan(
        device=device,
        dry_run=dry_run,
        runbook=[f"setup_device({device}) stub (dry_run={dry_run})"],
    )


def setup_device_mac(*, dry_run: bool = True) -> SetupPlan:
    """macOS/MPS + Ollama setup helper (stub)."""

    return SetupPlan(device="apple", dry_run=dry_run, runbook=["setup_device_mac stub"])

