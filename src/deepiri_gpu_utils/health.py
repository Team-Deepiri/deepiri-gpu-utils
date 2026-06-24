"""CI-friendly health gate aggregating doctor, install, and detection checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .install_check import install_readiness
from .summary import hardware_summary

HealthStatus = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class HealthCheck:
    """One named health check."""

    name: str
    status: HealthStatus
    message: str


@dataclass(frozen=True)
class HealthReport:
    """Aggregate health suitable for CI gates and monitoring."""

    status: HealthStatus
    exit_code: int
    backend: str
    doctor_status: str
    install_ready: bool
    gpu_count: int
    checks: list[HealthCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _worst(*statuses: HealthStatus) -> HealthStatus:
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _exit_for(status: HealthStatus) -> int:
    if status == "ok":
        return 0
    if status == "warn":
        return 1
    return 2


def health_check() -> HealthReport:
    """Run aggregate health checks; never raises."""

    snap = hardware_summary()
    d = snap.detect
    rep = snap.doctor
    install = install_readiness(device="auto")

    checks: list[HealthCheck] = []

    doctor_status: HealthStatus = "ok"
    if rep.status == "warn":
        doctor_status = "warn"
    elif rep.status == "unknown":
        doctor_status = "warn"
    checks.append(
        HealthCheck(
            name="doctor",
            status=doctor_status,
            message=f"doctor status={rep.status}",
        )
    )

    install_status: HealthStatus = "ok"
    if install.drivers_missing:
        install_status = "fail"
    elif not install.ready:
        install_status = "warn"
    checks.append(
        HealthCheck(
            name="install",
            status=install_status,
            message=(
                "install ready"
                if install.ready
                else f"missing={','.join(install.missing_required) or 'drivers'}"
            ),
        )
    )

    inventory_status: HealthStatus = "ok"
    if d.backend in ("cuda", "rocm") and snap.gpu_count == 0:
        inventory_status = "warn"
    checks.append(
        HealthCheck(
            name="inventory",
            status=inventory_status,
            message=f"gpus={snap.gpu_count} backend={d.backend}",
        )
    )

    detect_status: HealthStatus = "ok"
    if d.backend == "unknown":
        detect_status = "fail"
    if d.details.get("nvidia_drivers_missing") or d.details.get("rocm_drivers_missing"):
        detect_status = "fail"
    checks.append(
        HealthCheck(
            name="detect",
            status=detect_status,
            message=f"backend={d.backend} confidence={d.confidence:.2f}",
        )
    )

    overall = _worst(*(c.status for c in checks))
    notes = list(snap.notes)
    if rep.runbook:
        notes.append(f"runbook items={len(rep.runbook)}")

    return HealthReport(
        status=overall,
        exit_code=_exit_for(overall),
        backend=d.backend,
        doctor_status=rep.status,
        install_ready=install.ready,
        gpu_count=snap.gpu_count,
        checks=checks,
        notes=notes,
    )
