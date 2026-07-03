"""Regression tests for env hints, gpu top, and snapshots."""

from __future__ import annotations

import json
from pathlib import Path

from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.env_hints import runtime_env_hints
from deepiri_gpu_utils.gpu_top import gpu_top
from deepiri_gpu_utils.snapshot import capture_snapshot, diff_snapshots, save_snapshot


def test_runtime_env_hints_cpu(force_cpu) -> None:
    hints = runtime_env_hints(backend="cpu")
    assert hints.backend == "cpu"
    assert any(h.key == "OMP_NUM_THREADS" for h in hints.hints)
    assert hints.export_lines

def test_gpu_top_no_nvidia(force_cpu, monkeypatch) -> None:
    monkeypatch.setattr("deepiri_gpu_utils.gpu_top.shutil.which", lambda _: None)
    result = gpu_top()
    assert result.processes == []
    assert result.warnings


def test_snapshot_save_and_diff(force_cpu, tmp_path: Path) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    save_snapshot(path_a)
    save_snapshot(path_b)
    diff = diff_snapshots(json.loads(path_a.read_text()), json.loads(path_b.read_text()))
    assert diff.changed == {}
    snap = capture_snapshot()
    assert snap["schema"] == "deepiri-gpu-snapshot/v1"


def test_env_hints_cli(force_cpu, capsys) -> None:
    assert main(["env-hints", "--backend", "cpu", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "cpu"
