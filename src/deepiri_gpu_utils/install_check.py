"""Read-only install readiness checks for every supported GPU backend.

Reports which driver/tooling is present vs missing for NVIDIA, AMD/ROCm,
Apple/MPS, and CPU-only hosts. Never runs package managers or sudo.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass, field
from typing import Literal

from .detect import detect
from .profiles import BackendName, all_backend_profiles, backend_profile
from .setup import DeviceArg
from .system_info import docker_cli_available, lspci_amd_present, lspci_nvidia_present

DeviceArgInput = Literal["auto", "nvidia", "amd", "apple", "cpu"]

_DEVICE_TO_BACKEND: dict[DeviceArg, BackendName] = {
    "nvidia": "cuda",
    "amd": "rocm",
    "apple": "mps",
    "cpu": "cpu",
}


@dataclass(frozen=True)
class ToolCheck:
    """Whether a single tool/binary is on PATH."""

    tool: str
    required: bool
    present: bool


@dataclass(frozen=True)
class InstallReadiness:
    """Install posture for one backend profile on this host."""

    backend: BackendName
    device: DeviceArg
    profile_label: str
    ready: bool
    checks: list[ToolCheck] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    install_steps: list[str] = field(default_factory=list)
    verify_commands: list[str] = field(default_factory=list)
    pci_visible: bool | None = None
    drivers_missing: bool = False
    notes: list[str] = field(default_factory=list)


def _resolve_target(device: DeviceArgInput) -> tuple[DeviceArg, BackendName]:
    if device != "auto":
        dev: DeviceArg = device  # type: ignore[assignment]
        return dev, _DEVICE_TO_BACKEND[dev]

    d = detect()
    if d.backend == "cuda":
        return "nvidia", "cuda"
    if d.backend == "rocm":
        return "amd", "rocm"
    if d.backend == "mps":
        return "apple", "mps"
    return "cpu", "cpu"


def _pci_and_driver_flags(backend: BackendName) -> tuple[bool | None, bool]:
    if platform.system() != "Linux":
        return None, False
    if backend == "cuda":
        pci = lspci_nvidia_present()
        drivers_missing = pci is True and shutil.which("nvidia-smi") is None
        return pci, drivers_missing
    if backend == "rocm":
        pci = lspci_amd_present()
        drivers_missing = pci is True and shutil.which("rocm-smi") is None
        return pci, drivers_missing
    return None, False


def install_readiness(*, device: DeviceArgInput = "auto") -> InstallReadiness:
    """Check whether required tooling is installed for the target GPU profile."""

    dev, backend = _resolve_target(device)
    profile = backend_profile(backend)
    checks: list[ToolCheck] = []
    missing: list[str] = []

    for tool in profile.required_tools:
        present = shutil.which(tool) is not None
        checks.append(ToolCheck(tool=tool, required=True, present=present))
        if not present:
            missing.append(tool)

    for tool in profile.optional_tools:
        present = shutil.which(tool) is not None
        checks.append(ToolCheck(tool=tool, required=False, present=present))

    pci_visible, drivers_missing = _pci_and_driver_flags(backend)
    notes: list[str] = []
    if drivers_missing:
        notes.append(
            f"PCI reports {profile.label} hardware but required drivers/tools are missing."
        )
    if backend == "cuda" and docker_cli_available() and shutil.which("nvidia-ctk") is None:
        notes.append("Docker is installed but NVIDIA Container Toolkit was not found on PATH.")
    if backend == "mps" and platform.system() != "Darwin":
        notes.append("MPS profile selected but host platform is not Darwin.")

    ready = not missing and not drivers_missing
    return InstallReadiness(
        backend=backend,
        device=dev,
        profile_label=profile.label,
        ready=ready,
        checks=checks,
        missing_required=missing,
        install_steps=list(profile.install_steps),
        verify_commands=list(profile.verify_commands),
        pci_visible=pci_visible,
        drivers_missing=drivers_missing,
        notes=notes,
    )


def install_readiness_all() -> list[InstallReadiness]:
    """Return readiness for every canonical backend profile on this host."""

    return [
        install_readiness(device=profile.device_setup_arg)  # type: ignore[arg-type]
        for profile in all_backend_profiles()
    ]
