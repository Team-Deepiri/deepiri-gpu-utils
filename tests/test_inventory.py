"""Regression tests for the read-only GPU inventory + suitability helpers.

Every external probe (nvidia-smi via run_text, rocm-smi via detect, platform)
is mocked, so these run on a CPU-only host with no GPU hardware.
"""

from __future__ import annotations

import subprocess

import pytest
from conftest import FakeProc, run_router, which_map

import deepiri_gpu_utils.detect as det
import deepiri_gpu_utils.inventory as inv

TWO_GPU_CSV = (
    "0, NVIDIA GeForce RTX 3090, 24576, 24000, 5, 550.54.15\n"
    "1, NVIDIA GeForce RTX 3080, 10240, 2000, 0, 550.54.15\n"
)

ROCM_TEXT = (
    "GPU[0]\t: Card series:\t\tNavi 21 [Radeon RX 6800/6800 XT/6900 XT]\n"
    "GPU[0]\t: Card model:\t\t0x73bf\n"
)


def _mock_nvidia(monkeypatch, stdout: str) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(subprocess, "run", run_router({"nvidia-smi": FakeProc(0, stdout)}))


def _mock_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(inv.platform, "system", lambda: "Linux")


# --- NVIDIA inventory --------------------------------------------------------


def test_nvidia_multi_gpu_parsing(monkeypatch) -> None:
    _mock_nvidia(monkeypatch, TWO_GPU_CSV)
    result = inv.gpu_inventory()
    assert result.source == "nvidia-smi"
    assert len(result.gpus) == 2

    g0, g1 = result.gpus
    assert g0.backend == "cuda"
    assert g0.index == 0
    assert g0.name == "NVIDIA GeForce RTX 3090"
    assert g0.memory_mib == 24576
    assert g0.memory_gb == 24.0
    assert g0.utilization_percent == 5
    assert g0.driver_version == "550.54.15"
    assert g0.source == "nvidia-smi"
    assert g0.details["memory_free_gb"] == 23.44

    assert g1.index == 1
    assert g1.memory_gb == 10.0
    assert g1.details["memory_free_gb"] == 1.95


def test_nvidia_malformed_rows_are_skipped(monkeypatch) -> None:
    csv = (
        "0, NVIDIA Good, 8192, 8000, 10, 550.0\n"
        "justonetoken\n"
        "2, NVIDIA Partial, [N/A], [N/A], [N/A], 550.0\n"
    )
    _mock_nvidia(monkeypatch, csv)
    result = inv.gpu_inventory()
    assert len(result.gpus) == 2
    assert [g.index for g in result.gpus] == [0, 2]
    partial = result.gpus[1]
    assert partial.name == "NVIDIA Partial"
    assert partial.memory_mib is None
    assert partial.memory_gb is None
    assert partial.utilization_percent is None
    assert any("justonetoken" in w for w in result.warnings)


def test_nvidia_missing_returns_empty(monkeypatch) -> None:
    _mock_no_gpu(monkeypatch)
    result = inv.gpu_inventory()
    assert result.gpus == []
    assert result.source is None
    assert any("CPU-only" in w for w in result.warnings)


@pytest.mark.parametrize(
    "response",
    [
        FakeProc(returncode=9, stdout="driver error"),
        subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=15),
        OSError("boom"),
    ],
)
def test_nvidia_failure_modes_return_empty(monkeypatch, response) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(subprocess, "run", run_router({"nvidia-smi": response}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(inv.platform, "system", lambda: "Linux")
    result = inv.gpu_inventory()
    assert result.gpus == []
    assert result.source is None


# --- ROCm / MPS / CPU --------------------------------------------------------


def test_rocm_simple_product_name(monkeypatch) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: {"raw": ROCM_TEXT})
    result = inv.gpu_inventory()
    assert result.source == "rocm-smi"
    assert len(result.gpus) == 1
    gpu = result.gpus[0]
    assert gpu.backend == "rocm"
    assert gpu.name == "Navi 21 [Radeon RX 6800/6800 XT/6900 XT]"
    assert gpu.memory_gb is None
    assert gpu.utilization_percent is None


def test_mps_inventory(monkeypatch) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(inv.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(inv.platform, "machine", lambda: "arm64")
    result = inv.gpu_inventory()
    assert result.source == "platform"
    assert len(result.gpus) == 1
    gpu = result.gpus[0]
    assert gpu.backend == "mps"
    assert gpu.name == "Apple MPS"
    assert gpu.memory_gb is None
    assert gpu.details["machine"] == "arm64"


def test_cpu_no_gpu_inventory(monkeypatch) -> None:
    _mock_no_gpu(monkeypatch)
    result = inv.gpu_inventory()
    assert result.gpus == []
    assert result.source is None
    assert result.warnings


# --- suitability -------------------------------------------------------------


def test_choose_prefers_highest_free_memory(monkeypatch) -> None:
    # GPU-A has more total VRAM but less *free*; selection must follow free VRAM.
    _mock_nvidia(
        monkeypatch,
        "0, GPU-A, 24576, 6144, 0, 550\n1, GPU-B, 16384, 14336, 0, 550\n",
    )
    sel = inv.choose_suitable_gpu(4.0)
    assert sel.suitable is True
    assert sel.selected is not None
    assert sel.selected.name == "GPU-B"
    assert sel.selected.index == 1
    assert len(sel.candidates) == 2


def test_choose_falls_back_to_total_when_free_unknown(monkeypatch) -> None:
    _mock_nvidia(
        monkeypatch,
        "0, GPU-A, 8192, [N/A], 0, 550\n1, GPU-B, 16384, [N/A], 0, 550\n",
    )
    sel = inv.choose_suitable_gpu(8.0)
    assert sel.suitable is True
    assert sel.selected is not None
    assert sel.selected.name == "GPU-B"


def test_choose_unsuitable_when_threshold_not_met(monkeypatch) -> None:
    _mock_nvidia(monkeypatch, "0, GPU-A, 4096, 3072, 0, 550\n")
    sel = inv.choose_suitable_gpu(8.0)
    assert sel.suitable is False
    assert sel.selected is None
    assert len(sel.candidates) == 1
    assert "8" in sel.reason
    assert sel.min_memory_gb == 8.0


def test_choose_unsuitable_in_cpu_environment(monkeypatch) -> None:
    _mock_no_gpu(monkeypatch)
    sel = inv.choose_suitable_gpu(8.0)
    assert sel.suitable is False
    assert sel.selected is None
    assert sel.candidates == []
    assert "No GPUs detected" in sel.reason


def test_choose_backend_filter_no_match(monkeypatch) -> None:
    _mock_nvidia(monkeypatch, "0, GPU-A, 24576, 24000, 0, 550\n")
    sel = inv.choose_suitable_gpu(4.0, backend="rocm")
    assert sel.suitable is False
    assert sel.selected is None
    assert "rocm" in sel.reason


def test_choose_unsuitable_when_memory_unknown(monkeypatch) -> None:
    # MPS reports no memory metric; suitability cannot be confirmed.
    monkeypatch.setattr(inv.shutil, "which", which_map({}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(inv.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(inv.platform, "machine", lambda: "arm64")
    sel = inv.choose_suitable_gpu(8.0)
    assert sel.suitable is False
    assert sel.selected is None
    assert "not reported" in sel.reason
