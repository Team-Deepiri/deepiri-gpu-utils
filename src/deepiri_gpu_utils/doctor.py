from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Literal

from .detect import DetectResult, detect
from .system_info import (
    dmidecode_inventory,
    docker_cli_available,
    is_wsl,
    nvidia_container_toolkit_hint,
    system_ram_gb,
)

LiteralStatus = Literal["ok", "warn", "unknown"]


@dataclass(frozen=True)
class DoctorReport:
    detect: DetectResult
    status: LiteralStatus = "unknown"
    findings: dict[str, Any] = field(default_factory=dict)
    runbook: list[str] = field(default_factory=list)


def doctor() -> DoctorReport:
    """Run readiness checks: detection, Docker/toolkit hints, DMI (best-effort), WSL notes."""

    d = detect()
    findings: dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "wsl": is_wsl(),
        "system_ram_gb": system_ram_gb(),
        "docker_cli": docker_cli_available(),
        "nvidia_container_toolkit": nvidia_container_toolkit_hint(),
        "dmi": dmidecode_inventory(),
    }

    runbook: list[str] = []
    status: LiteralStatus = "ok" if d.backend != "unknown" else "unknown"

    if d.backend == "mps":
        runbook.append(
            "Apple Silicon: run `ollama serve` natively for local LLM; ai-team start.sh "
            "excludes cyrex/ollama in Docker on MPS."
        )
        runbook.append("Docker Desktop: use arm64 images where possible.")
    elif d.backend == "cpu" and platform.system() == "Linux":
        runbook.append("CPU-only: inference will be slower; GPU optional for dev.")

    if is_wsl() and d.backend == "cuda":
        runbook.append(
            "WSL2 + CUDA: verify `nvidia-smi` in this distro; on host Windows, install "
            "NVIDIA Game Ready/Studio drivers with WSL support."
        )
        status = "warn"

    if d.backend == "cuda" and docker_cli_available():
        hints = findings["nvidia_container_toolkit"]
        if (
            isinstance(hints, dict)
            and not hints.get("nvidia_ctk_on_path")
            and not hints.get("nvidia_container_binary")
        ):
            runbook.append(
                "Docker + NVIDIA: install NVIDIA Container Toolkit, then "
                "`sudo nvidia-ctk runtime configure --runtime=docker --set-as-default` "
                "and restart Docker (see Nvidia install guide)."
            )
            status = "warn"

    if d.details.get("nvidia_drivers_missing"):
        runbook.append(
            "PCI shows NVIDIA but drivers are missing: install OS-appropriate NVIDIA "
            "drivers until `nvidia-smi` works."
        )
        status = "warn"

    if d.backend == "rocm":
        runbook.append(
            "ROCm: install AMD ROCm stack for your distro; Docker ROCm images differ "
            "from NVIDIA — see AMD docs."
        )

    dmi = findings.get("dmi")
    if isinstance(dmi, dict) and not dmi.get("available"):
        runbook.append(
            "DMI/SMBIOS: run `sudo dmidecode -s system-product-name` for inventory "
            f"({dmi.get('reason', 'unavailable')})."
        )

    return DoctorReport(detect=d, status=status, findings=findings, runbook=runbook)
