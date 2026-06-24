"""Regression tests for :mod:`deepiri_gpu_utils.summary`."""

from __future__ import annotations

import json

from conftest import which_map

import deepiri_gpu_utils.detect as det
import deepiri_gpu_utils.inventory as inv
from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.summary import hardware_summary


def _mock_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(inv.platform, "system", lambda: "Linux")


def test_hardware_summary_cpu_shape(force_cpu, monkeypatch) -> None:
    _mock_no_gpu(monkeypatch)
    snap = hardware_summary()
    assert snap.detect.backend == "cpu"
    assert snap.gpu_count == 0
    assert snap.total_vram_gb is None
    assert isinstance(snap.notes, list)
    assert snap.build_args.device_type == "cpu"


def test_summary_json_cli_shape(force_cpu, monkeypatch, capsys) -> None:
    _mock_no_gpu(monkeypatch)
    rc = main(["summary", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "detect",
        "doctor",
        "inventory",
        "build_args",
        "ollama",
        "torch_device",
        "system_ram_gb",
        "gpu_count",
        "total_vram_gb",
        "notes",
    }
    assert payload["detect"]["backend"] == "cpu"
    assert payload["gpu_count"] == 0
