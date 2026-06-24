"""Regression tests for :mod:`deepiri_gpu_utils.model_matrix`."""

from __future__ import annotations

import json

from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.model_matrix import model_fit_matrix, render_model_matrix_text
from deepiri_gpu_utils.ollama import curated_model_ids, curated_models


def test_curated_models_public_api() -> None:
    ids = curated_model_ids()
    models = curated_models()
    assert len(ids) == len(models) > 20
    assert ids[0] == "mistral:7b"
    assert models[0][0] == "mistral:7b"


def test_model_fit_matrix_covers_curated_list(force_cpu) -> None:
    matrix = model_fit_matrix()
    assert len(matrix.rows) == len(curated_model_ids())
    assert set(matrix.counts) == {"recommended", "usable", "marginal", "no"}
    assert sum(matrix.counts.values()) == len(matrix.rows)


def test_render_model_matrix_text(force_cpu) -> None:
    text = render_model_matrix_text()
    assert "model matrix" in text
    assert "mistral:7b" in text


def test_model_matrix_json_cli(force_cpu, capsys) -> None:
    rc = main(["model-matrix", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "setup_tier",
        "system_ram_gb",
        "effective_vram_gb",
        "backend",
        "rows",
        "counts",
    }
    assert len(payload["rows"]) == len(curated_model_ids())
