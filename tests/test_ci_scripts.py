"""Focused policy tests for the reusable CI shell wrappers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fake_cli(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    cli = bin_dir / "deepiri-gpu"
    cli.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"$DEEPIRI_GPU_TEST_LOG"
if [[ "$1 $2" == "health --json" ]]; then
  printf '{"status":"%s","exit_code":%s}\n' "$DEEPIRI_GPU_TEST_STATUS" "$DEEPIRI_GPU_TEST_EXIT"
  exit "$DEEPIRI_GPU_TEST_EXIT"
fi
if [[ "$1 $2" == "snapshot save" ]]; then
  printf '{"schema":"deepiri-gpu-snapshot/v1"}\n' >"$3"
  printf '{"saved":"%s"}\n' "$3"
  exit 0
fi
if [[ "$1 $2" == "snapshot diff" ]]; then
  printf '{"changed":{}}\n'
  exit 0
fi
exit 99
""",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    return bin_dir, log


def _env(bin_dir: Path, log: Path, **updates: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "DEEPIRI_GPU_TEST_LOG": str(log),
            "DEEPIRI_GPU_TEST_STATUS": "ok",
            "DEEPIRI_GPU_TEST_EXIT": "0",
        }
    )
    env.update(updates)
    return env


def test_health_ok_passes_and_uses_configured_report_path(tmp_path: Path) -> None:
    bin_dir, log = _fake_cli(tmp_path)
    report = tmp_path / "reports" / "health.json"
    report.parent.mkdir()
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci-health-gate.sh")],
        env=_env(
            bin_dir,
            log,
            DEEPIRI_GPU_HEALTH_REPORT=str(report),
        ),
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "status": "ok",
        "exit_code": 0,
    }
    assert log.read_text(encoding="utf-8").splitlines() == ["health --json"]


def test_health_invalid_report_path_fails_without_invoking_cli(tmp_path: Path) -> None:
    bin_dir, log = _fake_cli(tmp_path)
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("regular file", encoding="utf-8")
    report = not_a_directory / "health.json"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci-health-gate.sh"), str(report)],
        env=_env(bin_dir, log),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert not log.exists()
    assert f"Unable to create GPU health report directory: {not_a_directory}" in result.stderr


def test_health_warning_is_preserved_and_non_blocking_by_default(tmp_path: Path) -> None:
    bin_dir, log = _fake_cli(tmp_path)
    report = tmp_path / "health.json"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci-health-gate.sh"), str(report)],
        env=_env(
            bin_dir,
            log,
            DEEPIRI_GPU_TEST_STATUS="warn",
            DEEPIRI_GPU_TEST_EXIT="1",
        ),
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "warn"


def test_health_warning_can_block_and_failure_always_blocks(tmp_path: Path) -> None:
    bin_dir, log = _fake_cli(tmp_path)
    warning = subprocess.run(
        ["bash", str(ROOT / "scripts/ci-health-gate.sh")],
        cwd=tmp_path,
        env=_env(
            bin_dir,
            log,
            DEEPIRI_GPU_TEST_STATUS="warn",
            DEEPIRI_GPU_TEST_EXIT="1",
            DEEPIRI_GPU_FAIL_ON_WARNING="true",
        ),
        check=False,
    )
    failure = subprocess.run(
        ["bash", str(ROOT / "scripts/ci-health-gate.sh")],
        cwd=tmp_path,
        env=_env(
            bin_dir,
            log,
            DEEPIRI_GPU_TEST_STATUS="fail",
            DEEPIRI_GPU_TEST_EXIT="2",
        ),
        check=False,
    )

    assert warning.returncode == 1
    assert failure.returncode == 2


def test_health_unexpected_exit_is_preserved_with_report(tmp_path: Path) -> None:
    bin_dir, log = _fake_cli(tmp_path)
    report = tmp_path / "unexpected.json"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/ci-health-gate.sh"), str(report)],
        env=_env(
            bin_dir,
            log,
            DEEPIRI_GPU_TEST_STATUS="unexpected",
            DEEPIRI_GPU_TEST_EXIT="7",
        ),
        check=False,
    )

    assert result.returncode == 7
    assert json.loads(report.read_text(encoding="utf-8"))["exit_code"] == 7


def test_snapshot_requires_opt_in_and_only_diffs_real_baseline(tmp_path: Path) -> None:
    bin_dir, log = _fake_cli(tmp_path)
    script = str(ROOT / "scripts/save-snapshot.sh")
    candidate = tmp_path / "candidate.json"

    skipped = subprocess.run(
        ["bash", script, str(candidate)],
        env=_env(bin_dir, log),
        check=False,
    )
    assert skipped.returncode == 0
    assert not candidate.exists()
    assert not log.exists()

    saved = subprocess.run(
        ["bash", script, str(candidate)],
        env=_env(bin_dir, log, DEEPIRI_GPU_SNAPSHOT_ENABLED="true"),
        check=False,
    )
    assert saved.returncode == 0
    assert candidate.exists()
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"snapshot save {candidate} --json"
    ]

    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")
    diff_report = tmp_path / "diff.json"
    diffed = subprocess.run(
        ["bash", script, str(candidate)],
        env=_env(
            bin_dir,
            log,
            DEEPIRI_GPU_SNAPSHOT_ENABLED="true",
            DEEPIRI_GPU_SNAPSHOT_BASELINE=str(baseline),
            DEEPIRI_GPU_SNAPSHOT_DIFF_REPORT=str(diff_report),
        ),
        check=False,
    )
    assert diffed.returncode == 0
    assert json.loads(diff_report.read_text(encoding="utf-8")) == {"changed": {}}
    assert log.read_text(encoding="utf-8").splitlines()[-2:] == [
        f"snapshot save {candidate} --json",
        f"snapshot diff {baseline} {candidate} --json",
    ]


def test_snapshot_missing_baseline_fails_without_diff(tmp_path: Path) -> None:
    bin_dir, log = _fake_cli(tmp_path)
    candidate = tmp_path / "candidate.json"
    missing = tmp_path / "missing-baseline.json"
    diff_report = tmp_path / "diff.json"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/save-snapshot.sh"), str(candidate)],
        env=_env(
            bin_dir,
            log,
            DEEPIRI_GPU_SNAPSHOT_ENABLED="TRUE",
            DEEPIRI_GPU_SNAPSHOT_BASELINE=str(missing),
            DEEPIRI_GPU_SNAPSHOT_DIFF_REPORT=str(diff_report),
        ),
        check=False,
    )

    assert result.returncode == 2
    assert candidate.exists()
    assert not diff_report.exists()
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"snapshot save {candidate} --json"
    ]
