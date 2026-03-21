from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Backend = Literal["cuda", "rocm", "mps", "cpu", "unknown"]


@dataclass(frozen=True)
class DetectResult:
    backend: Backend
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def detect(*, prefer: Backend | None = None) -> DetectResult:
    """Detect best available backend (stub).

    This is intentionally non-invasive: the real implementation will probe
    OS/GPU signals in later branches.
    """

    _ = prefer
    return DetectResult(backend="unknown", confidence=0.0, details={}, warnings=["detect() stub"])

