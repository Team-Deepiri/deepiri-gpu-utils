"""Docker Compose GPU fragments for NVIDIA, AMD/ROCm, Apple/MPS, and CPU.

Read-only helpers that emit compose-friendly device/deploy blocks for
downstream repos (diri-cyrex, diri-helox, deepiri-ollama-utils). Does not
write files or mutate Docker state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detect import detect
from .profiles import BackendName, backend_profile


@dataclass(frozen=True)
class ComposeGPUConfig:
    """Compose service GPU stanza for the active (or requested) backend."""

    backend: BackendName
    deploy_devices: list[str] = field(default_factory=list)
    device_requests: list[dict[str, object]] = field(default_factory=list)
    run_gpu_args: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def compose_gpu_config(*, backend: BackendName | None = None) -> ComposeGPUConfig:
    """Return compose GPU settings for ``backend`` or the detected backend."""

    resolved: BackendName
    if backend is not None:
        resolved = backend
    else:
        d = detect()
        resolved = d.backend if d.backend in ("cuda", "rocm", "mps", "cpu") else "cpu"

    profile = backend_profile(resolved)
    env: dict[str, str] = {}
    notes: list[str] = []

    if resolved == "rocm":
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        notes.append(
            "Set HSA_OVERRIDE_GFX_VERSION per your AMD GPU if ROCm reports unsupported gfx."
        )
    if resolved == "mps":
        notes.append(
            "Compose services on macOS typically run CPU/arm64; use native Ollama for Metal."
        )

    return ComposeGPUConfig(
        backend=resolved,
        deploy_devices=list(profile.compose_deploy_devices),
        device_requests=list(profile.compose_device_requests),
        run_gpu_args=list(profile.docker_run_gpu_args),
        environment=env,
        notes=notes,
    )
