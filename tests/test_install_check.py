"""Regression tests for :mod:`deepiri_gpu_utils.install_check`."""

from __future__ import annotations

import json

from conftest import which_map

import deepiri_gpu_utils.detect as det
import deepiri_gpu_utils.install_check as ic
import deepiri_gpu_utils.system_info as si
from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.install_check import install_readiness, install_readiness_all


def test_install_readiness_cpu_always_ready(force_cpu) -> None:
    result = install_readiness(device="cpu")
    assert result.backend == "cpu"
    assert result.ready is True
    assert result.missing_required == []


def test_install_readiness_nvidia_missing_tools(force_cpu, monkeypatch) -> None:
    monkeypatch.setattr(ic.shutil, "which", which_map({}))
    monkeypatch.setattr(si, "lspci_nvidia_present", lambda: None)
    result = install_readiness(device="nvidia")
    assert result.backend == "cuda"
    assert result.ready is False
    assert "nvidia-smi" in result.missing_required


def test_install_readiness_amd_pci_without_drivers(force_cpu, monkeypatch) -> None:
    monkeypatch.setattr(ic.shutil, "which", which_map({"lspci": "/usr/bin/lspci"}))
    monkeypatch.setattr(ic, "lspci_amd_present", lambda: True)
    result = install_readiness(device="amd")
    assert result.backend == "rocm"
    assert result.ready is False
    assert result.drivers_missing is True


def test_install_readiness_all_returns_four_profiles(force_cpu) -> None:
    results = install_readiness_all()
    assert len(results) == 4
    assert {r.backend for r in results} == {"cuda", "rocm", "mps", "cpu"}


def test_install_check_json_cli(force_cpu, capsys) -> None:
    rc = main(["install-check", "--device", "cpu", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "backend",
        "device",
        "profile_label",
        "ready",
        "checks",
        "missing_required",
        "install_steps",
        "verify_commands",
        "pci_visible",
        "drivers_missing",
        "notes",
    }
    assert payload["ready"] is True


def test_detect_amd_lspci_fallback(force_cpu, monkeypatch) -> None:
    monkeypatch.setattr(det.shutil, "which", which_map({"lspci": "/usr/bin/lspci"}))
    monkeypatch.setattr(det, "query_nvidia_smi", lambda: None)
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(det.system_info, "lspci_nvidia_present", lambda: False)
    monkeypatch.setattr(det.system_info, "lspci_amd_present", lambda: True)
    monkeypatch.setattr(det.platform, "system", lambda: "Linux")
    result = det.detect()
    assert result.backend == "rocm"
    assert result.details.get("rocm_drivers_missing") is True
