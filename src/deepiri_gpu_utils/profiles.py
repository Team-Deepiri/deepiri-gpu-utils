"""Canonical backend profiles for NVIDIA, AMD/ROCm, Apple/MPS, and CPU hosts.

Read-only reference data used by install checks, setup runbooks, and compose
helpers. Does not execute installs or mutate the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .build_args import DEFAULT_CPU_IMAGE, DEFAULT_PYTORCH_GPU_IMAGE

BackendName = Literal["cuda", "rocm", "mps", "cpu"]


@dataclass(frozen=True)
class BackendProfile:
    """Install + docker guidance for one accelerator backend."""

    backend: BackendName
    label: str
    device_setup_arg: str
    required_tools: list[str] = field(default_factory=list)
    optional_tools: list[str] = field(default_factory=list)
    install_steps: list[str] = field(default_factory=list)
    verify_commands: list[str] = field(default_factory=list)
    docker_run_gpu_args: list[str] = field(default_factory=list)
    compose_deploy_devices: list[str] = field(default_factory=list)
    compose_device_requests: list[dict[str, object]] = field(default_factory=list)
    default_base_image: str = DEFAULT_CPU_IMAGE
    docs_url: str = ""


_PROFILES: dict[BackendName, BackendProfile] = {
    "cuda": BackendProfile(
        backend="cuda",
        label="NVIDIA CUDA",
        device_setup_arg="nvidia",
        required_tools=["nvidia-smi"],
        optional_tools=["docker", "nvidia-ctk"],
        install_steps=[
            "Install OS-appropriate NVIDIA drivers until `nvidia-smi` works.",
            "For Docker GPU: install NVIDIA Container Toolkit and configure the runtime.",
            "On WSL2: install Windows NVIDIA drivers with WSL support and run `wsl --update`.",
        ],
        verify_commands=[
            "nvidia-smi",
            "docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi",
        ],
        docker_run_gpu_args=["--gpus", "all"],
        compose_deploy_devices=["nvidia"],
        compose_device_requests=[{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}],
        default_base_image=DEFAULT_PYTORCH_GPU_IMAGE,
        docs_url="https://docs.nvidia.com/cuda/",
    ),
    "rocm": BackendProfile(
        backend="rocm",
        label="AMD ROCm",
        device_setup_arg="amd",
        required_tools=["rocm-smi"],
        optional_tools=["docker", "rocminfo"],
        install_steps=[
            "Install the ROCm stack for your distro: https://rocm.docs.amd.com/",
            "Verify `rocm-smi` and (optionally) `rocminfo` on native Linux.",
            "Docker: use ROCm-capable images; CUDA images are not interchangeable.",
            "ROCm on WSL is limited — prefer native Linux for AMD GPU development.",
        ],
        verify_commands=["rocm-smi --showproductname", "rocminfo | head"],
        docker_run_gpu_args=[
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            "--group-add",
            "video",
        ],
        compose_deploy_devices=[],
        compose_device_requests=[],
        default_base_image=DEFAULT_CPU_IMAGE,
        docs_url="https://rocm.docs.amd.com/",
    ),
    "mps": BackendProfile(
        backend="mps",
        label="Apple Metal (MPS)",
        device_setup_arg="apple",
        required_tools=[],
        optional_tools=["ollama", "docker"],
        install_steps=[
            "Use native arm64 tooling on Apple Silicon (Metal / MPS).",
            "Install Ollama natively: https://ollama.com/download or `brew install ollama`.",
            "Run `ollama serve` locally; prefer arm64 Docker images when using Docker Desktop.",
        ],
        verify_commands=["sysctl -n machdep.cpu.brand_string", "ollama --version"],
        docker_run_gpu_args=[],
        compose_deploy_devices=[],
        compose_device_requests=[],
        default_base_image=DEFAULT_CPU_IMAGE,
        docs_url="https://developer.apple.com/metal/pytorch/",
    ),
    "cpu": BackendProfile(
        backend="cpu",
        label="CPU-only",
        device_setup_arg="cpu",
        required_tools=[],
        optional_tools=["docker", "ollama"],
        install_steps=[
            "No GPU drivers required; expect slower PyTorch/Ollama inference.",
            "For Ollama in Docker: `docker compose up -d ollama` (CPU fallback).",
        ],
        verify_commands=["python3 -c \"import platform; print(platform.machine())\""],
        docker_run_gpu_args=[],
        compose_deploy_devices=[],
        compose_device_requests=[],
        default_base_image=DEFAULT_CPU_IMAGE,
        docs_url="",
    ),
}


def backend_profile(backend: BackendName) -> BackendProfile:
    """Return the canonical profile for ``backend``."""

    return _PROFILES[backend]


def all_backend_profiles() -> list[BackendProfile]:
    """Return profiles for cuda, rocm, mps, and cpu (stable order)."""

    return [_PROFILES[name] for name in ("cuda", "rocm", "mps", "cpu")]
