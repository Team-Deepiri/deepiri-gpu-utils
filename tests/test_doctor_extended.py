"""Regression tests for ``doctor()`` status + runbook logic.

Detection and all host probes are mocked, so these lock the readiness rules
(ok/warn transitions and runbook hints) without needing real hardware.
"""

from __future__ import annotations

import deepiri_gpu_utils.doctor as docmod
from deepiri_gpu_utils.detect import DetectResult


def _patch_env(
    monkeypatch,
    *,
    detect_result: DetectResult,
    wsl: bool = False,
    docker: bool = True,
    toolkit: dict | None = None,
    dmi: dict | None = None,
    system: str = "Linux",
) -> None:
    monkeypatch.setattr(docmod, "detect", lambda: detect_result)
    monkeypatch.setattr(docmod, "is_wsl", lambda: wsl)
    monkeypatch.setattr(docmod, "system_ram_gb", lambda: 16)
    monkeypatch.setattr(docmod, "docker_cli_available", lambda: docker)
    monkeypatch.setattr(
        docmod,
        "nvidia_container_toolkit_hint",
        lambda: toolkit if toolkit is not None else {"nvidia_ctk_on_path": True},
    )
    monkeypatch.setattr(
        docmod,
        "dmidecode_inventory",
        lambda: dmi if dmi is not None else {"available": True, "system_product_name": "X"},
    )
    monkeypatch.setattr(docmod.platform, "system", lambda: system)


def test_doctor_findings_keys(monkeypatch) -> None:
    _patch_env(monkeypatch, detect_result=DetectResult(backend="cpu", confidence=0.8))
    rep = docmod.doctor()
    assert set(rep.findings) == {
        "platform",
        "platform_release",
        "machine",
        "python",
        "wsl",
        "system_ram_gb",
        "docker_cli",
        "nvidia_container_toolkit",
        "dmi",
    }


def test_doctor_cpu_linux_is_ok(monkeypatch) -> None:
    _patch_env(monkeypatch, detect_result=DetectResult(backend="cpu", confidence=0.82))
    rep = docmod.doctor()
    assert rep.status == "ok"
    assert any("CPU-only" in line for line in rep.runbook)


def test_doctor_mps_runbook(monkeypatch) -> None:
    _patch_env(
        monkeypatch,
        detect_result=DetectResult(backend="mps", confidence=0.88),
        system="Darwin",
    )
    rep = docmod.doctor()
    assert rep.status == "ok"
    assert any("Apple Silicon" in line for line in rep.runbook)


def test_doctor_unknown_backend_status_unknown(monkeypatch) -> None:
    _patch_env(monkeypatch, detect_result=DetectResult(backend="unknown", confidence=0.0))
    rep = docmod.doctor()
    assert rep.status == "unknown"


def test_doctor_wsl_cuda_warns(monkeypatch) -> None:
    _patch_env(
        monkeypatch,
        detect_result=DetectResult(backend="cuda", confidence=0.92),
        wsl=True,
    )
    rep = docmod.doctor()
    assert rep.status == "warn"
    assert any("WSL2 + CUDA" in line for line in rep.runbook)


def test_doctor_cuda_missing_toolkit_warns(monkeypatch) -> None:
    _patch_env(
        monkeypatch,
        detect_result=DetectResult(backend="cuda", confidence=0.92),
        docker=True,
        toolkit={"nvidia_ctk_on_path": False},
    )
    rep = docmod.doctor()
    assert rep.status == "warn"
    assert any("NVIDIA Container Toolkit" in line for line in rep.runbook)


def test_doctor_nvidia_drivers_missing_warns(monkeypatch) -> None:
    _patch_env(
        monkeypatch,
        detect_result=DetectResult(
            backend="cuda", confidence=0.48, details={"nvidia_drivers_missing": True}
        ),
        docker=False,
    )
    rep = docmod.doctor()
    assert rep.status == "warn"
    assert any("drivers are missing" in line for line in rep.runbook)


def test_doctor_rocm_runbook(monkeypatch) -> None:
    _patch_env(monkeypatch, detect_result=DetectResult(backend="rocm", confidence=0.78))
    rep = docmod.doctor()
    assert any("ROCm" in line for line in rep.runbook)


def test_doctor_dmi_unavailable_hint(monkeypatch) -> None:
    _patch_env(
        monkeypatch,
        detect_result=DetectResult(backend="cpu", confidence=0.82),
        dmi={"available": False, "reason": "dmidecode typically requires root"},
    )
    rep = docmod.doctor()
    assert any("DMI/SMBIOS" in line for line in rep.runbook)
