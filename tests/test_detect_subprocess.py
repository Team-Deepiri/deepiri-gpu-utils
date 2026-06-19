"""Subprocess-level regression tests for the detection probes.

Covers ``query_nvidia_smi`` / ``query_rocm_smi`` and the ``detect()`` decision
tree (including confidence values) using fully mocked ``shutil.which`` and
``subprocess.run`` — no GPU required.
"""

from __future__ import annotations

import subprocess

from conftest import FakeProc, run_router, which_map

import deepiri_gpu_utils.detect as det

NVIDIA_CSV = "550.54, 8192, NVIDIA GeForce RTX 3080"


# --- query_nvidia_smi --------------------------------------------------------


def test_query_nvidia_smi_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({}))
    assert det.query_nvidia_smi() is None


def test_query_nvidia_smi_success(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(
        det.subprocess,
        "run",
        run_router({"nvidia-smi": FakeProc(returncode=0, stdout=NVIDIA_CSV + "\n")}),
    )
    out = det.query_nvidia_smi()
    assert out is not None
    assert out["driver_version"] == "550.54"
    assert out["memory_mib"] == 8192
    assert out["memory_gb"] == 8.0
    assert out["name"] == "NVIDIA GeForce RTX 3080"
    assert out["meets_min_vram"] is True
    assert out["blackwell_family"] is False


def test_query_nvidia_smi_nonzero_returncode(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(
        det.subprocess, "run", run_router({"nvidia-smi": FakeProc(returncode=9, stdout="boom")})
    )
    assert det.query_nvidia_smi() is None


def test_query_nvidia_smi_empty_output(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(
        det.subprocess, "run", run_router({"nvidia-smi": FakeProc(returncode=0, stdout="   \n")})
    )
    assert det.query_nvidia_smi() is None


def test_query_nvidia_smi_timeout_is_swallowed(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(
        det.subprocess,
        "run",
        run_router({"nvidia-smi": subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=15)}),
    )
    assert det.query_nvidia_smi() is None


def test_query_nvidia_smi_oserror_is_swallowed(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(det.subprocess, "run", run_router({"nvidia-smi": OSError()}))
    assert det.query_nvidia_smi() is None


def test_parse_nvidia_csv_line_handles_blackwell() -> None:
    out = det._parse_nvidia_csv_line("560.0, 16384, NVIDIA GeForce RTX 5090")
    assert out is not None
    assert out["blackwell_family"] is True
    assert out["memory_gb"] == 16.0


def test_parse_nvidia_csv_line_below_min_vram() -> None:
    out = det._parse_nvidia_csv_line("550.0, 2048, NVIDIA T400")
    assert out is not None
    assert out["meets_min_vram"] is False


def test_parse_nvidia_csv_line_rejects_short_row() -> None:
    assert det._parse_nvidia_csv_line("550.0, 8192") is None


# --- query_rocm_smi ----------------------------------------------------------


def test_query_rocm_smi_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({}))
    assert det.query_rocm_smi() is None


def test_query_rocm_smi_success(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"rocm-smi": "/opt/rocm/bin/rocm-smi"}))
    monkeypatch.setattr(
        det.subprocess,
        "run",
        run_router({"rocm-smi": FakeProc(returncode=0, stdout="GPU0: Radeon RX 6800\n")}),
    )
    out = det.query_rocm_smi()
    assert out is not None
    assert "Radeon RX 6800" in out["raw"]


def test_query_rocm_smi_nonzero_returncode(monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"rocm-smi": "/opt/rocm/bin/rocm-smi"}))
    monkeypatch.setattr(
        det.subprocess, "run", run_router({"rocm-smi": FakeProc(returncode=2, stdout="err")})
    )
    assert det.query_rocm_smi() is None


# --- detect() decision tree (locks backend + confidence) ---------------------


def test_detect_cuda_from_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(det, "query_nvidia_smi", lambda: {"meets_min_vram": True, "memory_gb": 8.0})
    monkeypatch.setattr(det.system_info, "is_wsl", lambda: False)
    monkeypatch.setattr(det.platform, "system", lambda: "Linux")
    r = det.detect()
    assert r.backend == "cuda"
    assert r.confidence == 0.92
    assert r.details["nvidia"]["memory_gb"] == 8.0


def test_detect_cuda_lspci_fallback_confidence(monkeypatch) -> None:
    monkeypatch.setattr(det, "query_nvidia_smi", lambda: None)
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(det.system_info, "lspci_nvidia_present", lambda: True)
    monkeypatch.setattr(det.system_info, "is_wsl", lambda: False)
    monkeypatch.setattr(det.platform, "system", lambda: "Linux")
    r = det.detect()
    assert r.backend == "cuda"
    assert r.confidence == 0.48
    assert r.details["nvidia_drivers_missing"] is True


def test_detect_rocm_confidence(monkeypatch) -> None:
    monkeypatch.setattr(det, "query_nvidia_smi", lambda: None)
    monkeypatch.setattr(det.system_info, "lspci_nvidia_present", lambda: None)
    monkeypatch.setattr(det, "query_rocm_smi", lambda: {"raw": "Radeon"})
    monkeypatch.setattr(det.system_info, "is_wsl", lambda: False)
    monkeypatch.setattr(det.platform, "system", lambda: "Linux")
    r = det.detect()
    assert r.backend == "rocm"
    assert r.confidence == 0.78


def test_detect_mps_confidence(monkeypatch) -> None:
    monkeypatch.setattr(det, "query_nvidia_smi", lambda: None)
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(det.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(det.platform, "machine", lambda: "arm64")
    r = det.detect()
    assert r.backend == "mps"
    assert r.confidence == 0.88


def test_detect_cpu_fallback_confidence(force_cpu) -> None:
    r = det.detect()
    assert r.backend == "cpu"
    assert r.confidence == 0.82


def test_detect_prefer_mismatch_warning(monkeypatch) -> None:
    monkeypatch.setattr(det, "query_nvidia_smi", lambda: {"meets_min_vram": True})
    monkeypatch.setattr(det.system_info, "is_wsl", lambda: False)
    monkeypatch.setattr(det.platform, "system", lambda: "Linux")
    r = det.detect(prefer="cpu")
    assert any("Preferred backend" in w for w in r.warnings)
