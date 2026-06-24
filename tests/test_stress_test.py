"""Regression tests for :mod:`deepiri_gpu_utils.stress_test`."""

from __future__ import annotations

import json

from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.stress_test import run_stress_test


def test_probe_stress_short(force_cpu) -> None:
    result = run_stress_test(duration_s=0.5, mode="probes", backend="probes")
    assert result.mode == "probes"
    assert result.device == "probes"
    assert result.iterations > 0
    assert result.duration_actual_s >= 0.5
    assert result.matrix_size is None
    assert any("probe loop" in n for n in result.notes)


def test_compute_stress_cpu_fallback(force_cpu) -> None:
    result = run_stress_test(duration_s=0.5, mode="compute", backend="cpu", matrix_size=128)
    assert result.mode == "compute"
    assert result.device == "cpu"
    assert result.iterations > 0
    assert result.ops_per_sec > 0
    assert result.matrix_size == 128


def test_stress_duration_clamped() -> None:
    from deepiri_gpu_utils.stress_test import _clamp_duration

    assert _clamp_duration(999.0) == 120.0
    assert _clamp_duration(0.1) == 0.5
    assert _clamp_duration(5.0) == 5.0


def test_stress_json_cli(force_cpu, capsys) -> None:
    rc = main(["stress", "--mode", "probes", "--duration", "0.5", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "mode",
        "backend",
        "device",
        "duration_requested_s",
        "duration_actual_s",
        "iterations",
        "ops_per_sec",
        "matrix_size",
        "telemetry_samples",
        "peak_utilization_percent",
        "notes",
        "success",
    }
    assert payload["mode"] == "probes"
