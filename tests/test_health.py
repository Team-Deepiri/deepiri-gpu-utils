"""Regression tests for :mod:`deepiri_gpu_utils.health`."""

from __future__ import annotations

import json

from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.health import health_check


def test_health_check_cpu_shape(force_cpu) -> None:
    report = health_check()
    assert report.status in {"ok", "warn", "fail"}
    assert report.exit_code in {0, 1, 2}
    assert len(report.checks) >= 4


def test_health_json_cli(force_cpu, capsys) -> None:
    rc = main(["health", "--json"])
    assert rc in {0, 1, 2}
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "status",
        "exit_code",
        "backend",
        "doctor_status",
        "install_ready",
        "gpu_count",
        "checks",
        "notes",
    }
