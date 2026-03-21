from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BuildArgs:
    """Docker build arguments (stub)."""

    device_type: str = "unknown"
    cuda_version: str | None = None
    python_version: str = "3.11"
    build_args: dict[str, str] = field(default_factory=dict)


def build_args_from_detection(*, device_type: str = "auto") -> BuildArgs:
    """Return docker build args based on detected device (stub)."""

    _ = device_type
    return BuildArgs(
        device_type="unknown",
        cuda_version=None,
        build_args={"DEVICE_TYPE": "unknown"},
    )

