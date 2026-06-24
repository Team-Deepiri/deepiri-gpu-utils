"""Regression tests for :mod:`deepiri_gpu_utils.model_fit`."""

from __future__ import annotations

import json

from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.model_fit import model_fit_check


def test_model_fit_known_small_model(force_cpu) -> None:
    result = model_fit_check("llama3.2:1b")
    assert result.model == "llama3.2:1b"
    assert result.fit in {"recommended", "usable", "marginal", "no"}
    assert isinstance(result.suitable, bool)
    assert result.reason


def test_model_fit_large_model_unsuitable_on_cpu(force_cpu) -> None:
    result = model_fit_check("llama3.1:70b")
    assert result.fit == "no"
    assert result.suitable is False


def test_model_fit_json_cli(force_cpu, capsys) -> None:
    rc = main(["model-fit", "mistral:7b", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "model",
        "fit",
        "setup_tier",
        "system_ram_gb",
        "effective_vram_gb",
        "default_model",
        "suitable",
        "reason",
        "notes",
    }
    assert payload["model"] == "mistral:7b"
