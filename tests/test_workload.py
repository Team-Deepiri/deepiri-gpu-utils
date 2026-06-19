"""Regression tests for :mod:`deepiri_gpu_utils.workload`."""

from __future__ import annotations

import json

from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.workload import estimate_workload


def test_estimate_workload_small_model_fits_cpu(force_cpu) -> None:
    result = estimate_workload("llama3.2:1b")
    assert result.model == "llama3.2:1b"
    assert result.parameters_b == 1.0
    assert result.estimated_memory_gb > 0
    assert isinstance(result.fits, bool)


def test_estimate_workload_large_model_unlikely_on_cpu(force_cpu) -> None:
    result = estimate_workload("llama3.1:70b", context_tokens=8192)
    assert result.parameters_b == 70.0
    assert result.estimated_memory_gb > 100


def test_workload_json_cli(force_cpu, capsys) -> None:
    rc = main(["workload", "mistral:7b", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "model",
        "parameters_b",
        "estimated_memory_gb",
        "available_memory_gb",
        "memory_source",
        "fits",
        "headroom_gb",
        "notes",
    }
    assert payload["model"] == "mistral:7b"
