"""Regression tests for ``resolve_torch_device`` with torch absent and present.

``torch`` is an optional extra, so both paths must work in a CPU-only env.
``torch_absent`` / ``torch_present`` (from conftest) control ``import torch``
without requiring the real package, and ``detect`` is mocked for determinism.
"""

from __future__ import annotations

from conftest import torch_absent, torch_present

import deepiri_gpu_utils.torch_device as tdmod
from deepiri_gpu_utils.detect import DetectResult
from deepiri_gpu_utils.torch_device import resolve_torch_device


def _mock_detect(monkeypatch, backend: str) -> None:
    monkeypatch.setattr(tdmod, "detect", lambda: DetectResult(backend=backend, confidence=1.0))


# --- torch absent (heuristic) ------------------------------------------------


def test_absent_cpu_policy() -> None:
    with torch_absent():
        d = resolve_torch_device("cpu")
    assert d.device == "cpu"
    assert d.torch_available is False
    assert any("torch not installed" in n for n in d.notes)
    assert d.notes == ["policy=cpu; torch not installed"]


def test_absent_auto_with_cuda_detect(monkeypatch) -> None:
    _mock_detect(monkeypatch, "cuda")
    with torch_absent():
        d = resolve_torch_device("auto")
    assert d.device == "cuda"
    assert d.torch_available is False


def test_absent_auto_with_mps_detect(monkeypatch) -> None:
    _mock_detect(monkeypatch, "mps")
    with torch_absent():
        d = resolve_torch_device("auto")
    assert d.device == "mps"
    assert d.torch_available is False


def test_absent_auto_with_cpu_detect(monkeypatch) -> None:
    _mock_detect(monkeypatch, "cpu")
    with torch_absent():
        d = resolve_torch_device("auto")
    assert d.device == "cpu"
    assert d.torch_available is False


def test_absent_rocm_maps_to_cuda_guess(monkeypatch) -> None:
    _mock_detect(monkeypatch, "rocm")
    with torch_absent():
        d = resolve_torch_device("auto")
    assert d.device == "cuda"


# --- torch present (mocked module) -------------------------------------------


def test_present_cpu_policy() -> None:
    with torch_present():
        d = resolve_torch_device("cpu")
    assert d.device == "cpu"
    assert d.torch_available is True
    assert d.notes == ["policy=cpu"]


def test_present_cuda_policy_available(monkeypatch) -> None:
    _mock_detect(monkeypatch, "cuda")
    with torch_present(cuda=True):
        d = resolve_torch_device("cuda")
    assert d.device == "cuda"
    assert d.torch_available is True


def test_present_cuda_policy_unavailable_falls_back(monkeypatch) -> None:
    _mock_detect(monkeypatch, "cpu")
    with torch_present(cuda=False):
        d = resolve_torch_device("cuda")
    assert d.device == "cpu"
    assert d.torch_available is True


def test_present_mps_policy_available(monkeypatch) -> None:
    _mock_detect(monkeypatch, "mps")
    with torch_present(mps=True):
        d = resolve_torch_device("mps")
    assert d.device == "mps"
    assert d.torch_available is True


def test_present_mps_policy_unavailable_falls_back(monkeypatch) -> None:
    _mock_detect(monkeypatch, "cpu")
    with torch_present(mps=False):
        d = resolve_torch_device("mps")
    assert d.device == "cpu"


def test_present_rocm_policy_maps_to_cuda(monkeypatch) -> None:
    _mock_detect(monkeypatch, "rocm")
    with torch_present(cuda=True):
        d = resolve_torch_device("rocm")
    assert d.device == "cuda"


def test_present_auto_prefers_cuda(monkeypatch) -> None:
    _mock_detect(monkeypatch, "cuda")
    with torch_present(cuda=True):
        d = resolve_torch_device("auto")
    assert d.device == "cuda"


def test_present_auto_prefers_mps(monkeypatch) -> None:
    _mock_detect(monkeypatch, "mps")
    with torch_present(mps=True):
        d = resolve_torch_device("auto")
    assert d.device == "mps"


def test_present_auto_cpu_fallback(monkeypatch) -> None:
    _mock_detect(monkeypatch, "cpu")
    with torch_present():
        d = resolve_torch_device("auto")
    assert d.device == "cpu"
    assert d.torch_available is True
