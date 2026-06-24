"""CLI tests for the new read-only ``inventory`` command.

Verifies the parser, the stable JSON shape (gpus/warnings/source), and the
optional suitability ``selection`` block. External probes are mocked.
"""

from __future__ import annotations

import json
import subprocess

from conftest import FakeProc, run_router, which_map

import deepiri_gpu_utils.detect as det
import deepiri_gpu_utils.inventory as inv
from deepiri_gpu_utils.cli import build_parser, main

TWO_GPU_CSV = (
    "0, NVIDIA GeForce RTX 3090, 24576, 24000, 5, 550.54.15\n"
    "1, NVIDIA GeForce RTX 3080, 10240, 9000, 0, 550.54.15\n"
)


def _mock_nvidia(monkeypatch, stdout: str) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({"nvidia-smi": "/usr/bin/nvidia-smi"}))
    monkeypatch.setattr(subprocess, "run", run_router({"nvidia-smi": FakeProc(0, stdout)}))


def _mock_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(inv.platform, "system", lambda: "Linux")


def test_parser_accepts_inventory_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["inventory", "--json"]) is not None
    assert parser.parse_args(["inventory", "--min-memory-gb", "8", "--json"]) is not None
    assert parser.parse_args(["inventory", "--min-memory-gb", "8", "--backend", "cuda"]) is not None


def test_inventory_json_empty_shape(monkeypatch, capsys) -> None:
    _mock_no_gpu(monkeypatch)
    rc = main(["inventory", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"gpus", "warnings", "source"}
    assert payload["gpus"] == []
    assert payload["source"] is None


def test_inventory_json_with_gpus(monkeypatch, capsys) -> None:
    _mock_nvidia(monkeypatch, TWO_GPU_CSV)
    rc = main(["inventory", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"gpus", "warnings", "source"}
    assert payload["source"] == "nvidia-smi"
    assert len(payload["gpus"]) == 2
    assert payload["gpus"][0]["name"] == "NVIDIA GeForce RTX 3090"


def test_inventory_json_with_selection(monkeypatch, capsys) -> None:
    _mock_nvidia(monkeypatch, TWO_GPU_CSV)
    rc = main(["inventory", "--min-memory-gb", "8", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"gpus", "warnings", "source", "selection"}
    selection = payload["selection"]
    assert set(selection) == {
        "selected",
        "suitable",
        "reason",
        "candidates",
        "min_memory_gb",
    }
    assert selection["suitable"] is True
    assert selection["min_memory_gb"] == 8.0
    assert selection["selected"]["name"] == "NVIDIA GeForce RTX 3090"


def test_inventory_human_output_no_gpu(monkeypatch, capsys) -> None:
    _mock_no_gpu(monkeypatch)
    rc = main(["inventory"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No GPUs detected." in out
