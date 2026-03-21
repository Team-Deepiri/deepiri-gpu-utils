from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .detect import DetectResult

LiteralStatus = Literal["ok", "unknown"]


@dataclass(frozen=True)
class DoctorReport:
    detect: DetectResult
    status: LiteralStatus = "unknown"
    findings: dict[str, Any] = field(default_factory=dict)
    runbook: list[str] = field(default_factory=list)


def doctor() -> DoctorReport:
    """Run readiness checks (stub).

    Later branches will:
    - validate driver/runtime (NVIDIA container toolkit, etc.)
    - validate MPS availability on macOS
    - emit machine-readable findings
    """

    return DoctorReport(
        detect=DetectResult(backend="unknown"),
        status="unknown",
        findings={},
        runbook=[],
    )

