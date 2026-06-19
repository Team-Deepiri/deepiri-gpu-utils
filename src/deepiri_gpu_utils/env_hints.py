"""Recommended runtime environment variables per detected GPU backend.

Read-only: suggests env vars for compose, shell, and downstream services.
Does not set or export variables in the current process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compose_gpu import compose_gpu_config
from .detect import detect
from .torch_device import resolve_torch_device


@dataclass(frozen=True)
class EnvHint:
    """One environment variable recommendation."""

    key: str
    value: str
    reason: str
    required: bool = False


@dataclass(frozen=True)
class EnvHintsResult:
    """Runtime env hints for the active or requested backend."""

    backend: str
    hints: list[EnvHint] = field(default_factory=list)
    export_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def runtime_env_hints(*, backend: str | None = None) -> EnvHintsResult:
    """Return recommended runtime environment variables."""

    d = detect()
    resolved = backend or d.backend
    compose = compose_gpu_config(
        backend=resolved if resolved in ("cuda", "rocm", "mps", "cpu") else "cpu"  # type: ignore[arg-type]
    )
    torch_dec = resolve_torch_device("auto")

    hints: list[EnvHint] = []
    notes: list[str] = list(compose.notes)

    if resolved == "cuda":
        hints.append(
            EnvHint(
                key="NVIDIA_VISIBLE_DEVICES",
                value="all",
                reason="Expose all GPUs to containers (override per service as needed).",
            )
        )
        hints.append(
            EnvHint(
                key="PYTORCH_CUDA_ALLOC_CONF",
                value="expandable_segments:True",
                reason="Reduce CUDA OOM fragmentation for long-running inference.",
                required=False,
            )
        )
    elif resolved == "rocm":
        for key, value in compose.environment.items():
            hints.append(
                EnvHint(
                    key=key,
                    value=str(value),
                    reason="ROCm gfx compatibility override when auto-detection fails.",
                )
            )
        hints.append(
            EnvHint(
                key="HIP_VISIBLE_DEVICES",
                value="0",
                reason="Default first AMD GPU; adjust for multi-GPU hosts.",
            )
        )
    elif resolved == "mps":
        hints.append(
            EnvHint(
                key="PYTORCH_ENABLE_MPS_FALLBACK",
                value="1",
                reason="Allow CPU fallback when an op is unsupported on MPS.",
            )
        )
    else:
        hints.append(
            EnvHint(
                key="OMP_NUM_THREADS",
                value="4",
                reason="Sensible CPU inference default; tune per host.",
            )
        )

    hints.append(
        EnvHint(
            key="DEEPIRI_TORCH_DEVICE",
            value=torch_dec.device,
            reason="Resolved torch device from deepiri-gpu-utils.",
        )
    )

    export_lines = [f"export {h.key}={h.value!r}" for h in hints]
    return EnvHintsResult(
        backend=resolved,
        hints=hints,
        export_lines=export_lines,
        notes=notes,
    )
