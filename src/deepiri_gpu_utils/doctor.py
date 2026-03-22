from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Literal

from .detect import DetectResult, detect

LiteralStatus = Literal["ok", "unknown"]


@dataclass(frozen=True)
class DoctorReport:
    detect: DetectResult
    status: LiteralStatus = "unknown"
    findings: dict[str, Any] = field(default_factory=dict)
    runbook: list[str] = field(default_factory=list)


def doctor() -> DoctorReport:
    """Run lightweight readiness checks (detection + platform facts).

    Deeper checks (NVIDIA Container Toolkit, ``dmidecode``, etc.) come in Phase 2.
    """

    d = detect()
    status: LiteralStatus = "ok" if d.backend != "unknown" else "unknown"
    findings = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    runbook: list[str] = []
    if d.backend == "mps":
        runbook.append(
            "Apple Silicon: for local Ollama, run `ollama serve` outside Docker; "
            "see README (ai-team start.sh excludes cyrex/ollama on MPS)."
        )
    elif d.backend == "cpu" and platform.system() == "Linux":
        runbook.append("CPU-only: expect slower inference; GPU optional for development.")

    return DoctorReport(detect=d, status=status, findings=findings, runbook=runbook)
