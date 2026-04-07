from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .detect import detect
from .system_info import docker_cli_available, is_wsl

DeviceArg = Literal["auto", "nvidia", "amd", "apple", "cpu"]


@dataclass(frozen=True)
class SetupPlan:
    device: DeviceArg
    dry_run: bool = True
    runbook: list[str] = field(default_factory=list)


def _header(device: DeviceArg, dry_run: bool) -> list[str]:
    if dry_run:
        mode = "dry-run (commands not executed)"
    else:
        mode = "confirmed (--yes); run commands manually"
    return [
        f"# deepiri-gpu-utils setup plan: device={device} ({mode})",
        "",
    ]


def _nvidia_runbook() -> list[str]:
    lines = [
        "## NVIDIA + Docker (Linux / WSL2)",
        "1. Install NVIDIA drivers until `nvidia-smi` works in this environment.",
        "2. Install NVIDIA Container Toolkit (distro-specific). Examples:",
        "   - Debian/Ubuntu: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html",
        "3. Configure Docker runtime:",
        "   sudo nvidia-ctk runtime configure --runtime=docker --set-as-default",
        "4. Restart Docker: sudo systemctl restart docker  (or Docker Desktop restart on Mac/Win)",
        "5. Test: docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi",
        "",
    ]
    if is_wsl():
        lines += [
            "## WSL2 notes",
            "- Update WSL: `wsl --update` on Windows.",
            "- Use Windows drivers with WSL CUDA support.",
            "",
        ]
    return lines


def _amd_runbook() -> list[str]:
    return [
        "## AMD / ROCm",
        "1. Install ROCm for your distro: https://rocm.docs.amd.com/",
        "2. Verify: `rocm-smi` when available.",
        "3. Docker: use ROCm-capable images; not interchangeable with NVIDIA CUDA images.",
        "",
    ]


def _apple_runbook() -> list[str]:
    return [
        "## macOS / Apple Silicon",
        "1. Install Ollama: https://ollama.com/download or `brew install ollama`",
        "2. Run locally: `ollama serve` (native Metal).",
        (
            "3. Docker Desktop: prefer arm64; Cyrex/Ollama in compose may be excluded on MPS "
            "(see ai-team start.sh)."
        ),
        "",
    ]


def _cpu_runbook() -> list[str]:
    return [
        "## CPU-only",
        "1. No GPU drivers required; expect slower PyTorch/Ollama.",
        (
            "2. For Ollama in Docker: "
            "`docker compose -f docker-compose.dev.yml up -d ollama` (CPU fallback)."
        ),
        "",
    ]


def setup_device(device: DeviceArg = "auto", *, dry_run: bool = True) -> SetupPlan:
    """Printable runbook aligned with diri-cyrex GPU scripts.

    Non-interactive; does not run sudo.
    """

    dev: DeviceArg = device
    if dev == "auto":
        d = detect()
        if d.backend == "cuda":
            dev = "nvidia"
        elif d.backend == "rocm":
            dev = "amd"
        elif d.backend == "mps":
            dev = "apple"
        else:
            dev = "cpu"

    rb = _header(dev, dry_run)
    if dev == "nvidia":
        rb.extend(_nvidia_runbook())
    elif dev == "amd":
        rb.extend(_amd_runbook())
    elif dev == "apple":
        rb.extend(_apple_runbook())
    else:
        rb.extend(_cpu_runbook())

    if docker_cli_available():
        rb.append("Docker CLI detected on PATH.")
    else:
        rb.append("Docker CLI not found on PATH; install Docker Engine or Docker Desktop.")

    return SetupPlan(device=dev, dry_run=dry_run, runbook=rb)


def setup_device_mac(*, dry_run: bool = True) -> SetupPlan:
    """macOS-focused runbook (MPS + native Ollama)."""

    rb = _header("apple", dry_run)
    rb.extend(_apple_runbook())
    return SetupPlan(device="apple", dry_run=dry_run, runbook=rb)
