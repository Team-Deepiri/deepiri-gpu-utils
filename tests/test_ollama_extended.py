"""Regression tests for the Ollama RAM/VRAM tiering logic.

Locks ``setup_tier`` thresholds, ``_effective_vram_gb`` per backend, and the
``recommend_models`` backend-hint overrides. ``detect`` and ``system_ram_gb``
are mocked so results never depend on the host.
"""

from __future__ import annotations

import pytest

import deepiri_gpu_utils.ollama as ollama
from deepiri_gpu_utils.detect import DetectResult
from deepiri_gpu_utils.ollama import recommend_models, setup_tier


@pytest.mark.parametrize(
    ("ram", "vram", "expected"),
    [
        (32, 16, "setup5"),
        (32, 10, "setup4"),
        (32, 8, "setup3"),
        (0, 15, "setup5"),
        (16, 10, "setup2"),
        (16, 8, "setup1"),
        (16, 0, "basic"),
        (0, 8, "basic"),
        (8, 0, "minimal"),
    ],
)
def test_setup_tier_thresholds(ram: int, vram: int, expected: str) -> None:
    assert setup_tier(ram, vram) == expected


def test_effective_vram_mps_uses_ram() -> None:
    d = DetectResult(backend="mps")
    assert ollama._effective_vram_gb(d, ram_gb=24) == 24


def test_effective_vram_cuda_uses_nvidia_memory() -> None:
    d = DetectResult(backend="cuda", details={"nvidia": {"memory_gb": 12.0}})
    assert ollama._effective_vram_gb(d, ram_gb=32) == 12


def test_effective_vram_cpu_is_zero() -> None:
    d = DetectResult(backend="cpu")
    assert ollama._effective_vram_gb(d, ram_gb=32) == 0


def _patch(monkeypatch, *, backend: str, ram: int, details: dict | None = None) -> None:
    monkeypatch.setattr(
        ollama, "detect", lambda: DetectResult(backend=backend, details=details or {})
    )
    monkeypatch.setattr(ollama, "system_ram_gb", lambda: ram)


def test_recommend_cuda_high_end(monkeypatch) -> None:
    _patch(monkeypatch, backend="cuda", ram=32, details={"nvidia": {"memory_gb": 24.0}})
    rec = recommend_models()
    assert rec.setup_tier == "setup5"
    assert rec.system_ram_gb == 32
    assert rec.effective_vram_gb == 24
    assert rec.default_model == "mistral:7b"
    assert "mistral:7b" in rec.recommended_models


def test_recommend_cpu_only(monkeypatch) -> None:
    _patch(monkeypatch, backend="cpu", ram=16)
    rec = recommend_models()
    assert rec.effective_vram_gb == 0
    assert rec.system_ram_gb == 16
    assert rec.category == "hardware_tiered"


def test_recommend_backend_hint_cpu_forces_zero_vram(monkeypatch) -> None:
    _patch(monkeypatch, backend="cuda", ram=32, details={"nvidia": {"memory_gb": 24.0}})
    rec = recommend_models(backend_hint="cpu")
    assert rec.effective_vram_gb == 0
    assert any("CPU-tier" in n for n in rec.notes)


def test_recommend_backend_hint_mps_uses_ram_as_vram(monkeypatch) -> None:
    _patch(monkeypatch, backend="cpu", ram=16)
    rec = recommend_models(backend_hint="mps")
    assert rec.effective_vram_gb == 16
    assert any("unified memory" in n for n in rec.notes)
