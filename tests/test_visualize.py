"""Regression tests for :mod:`deepiri_gpu_utils.visualize`."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import which_map

import deepiri_gpu_utils.detect as det
import deepiri_gpu_utils.inventory as inv
from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.summary import hardware_summary
from deepiri_gpu_utils.visualize import render_dashboard, render_html_report


def _mock_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(inv.shutil, "which", which_map({}))
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(inv.platform, "system", lambda: "Linux")


def test_render_dashboard_cpu_shape(force_cpu, monkeypatch) -> None:
    _mock_no_gpu(monkeypatch)
    snap = hardware_summary()
    dash = render_dashboard(snap=snap)
    assert dash.backend == "cpu"
    assert dash.gpu_count == 0
    assert "deepiri-gpu dashboard" in dash.text
    assert "VRAM / utilization" in dash.text


def test_render_html_report_contains_sections(force_cpu, monkeypatch) -> None:
    _mock_no_gpu(monkeypatch)
    html = render_html_report(snap=hardware_summary())
    assert "<title>deepiri-gpu report</title>" in html
    assert "GPU inventory" in html
    assert "Ollama model fit" in html


def test_visualize_cli_ascii(force_cpu, monkeypatch, capsys) -> None:
    _mock_no_gpu(monkeypatch)
    rc = main(["visualize"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "deepiri-gpu dashboard" in out


def test_visualize_cli_html(force_cpu, monkeypatch, tmp_path: Path) -> None:
    _mock_no_gpu(monkeypatch)
    out_file = tmp_path / "report.html"
    rc = main(["visualize", "--html", str(out_file)])
    assert rc == 0
    text = out_file.read_text(encoding="utf-8")
    assert "deepiri-gpu hardware report" in text


def test_visualize_json_cli(force_cpu, monkeypatch, capsys) -> None:
    _mock_no_gpu(monkeypatch)
    rc = main(["visualize", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "ascii"
    assert "text" in payload
