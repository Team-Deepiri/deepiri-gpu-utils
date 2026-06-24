"""Regression tests for :mod:`deepiri_gpu_utils.export_env`."""

from __future__ import annotations

import shlex

from deepiri_gpu_utils.export_env import build_args_shell_export


def test_shell_export_lines_are_sourceable(force_cpu) -> None:
    export = build_args_shell_export(device_type="auto")
    assert export.build_args is not None
    assert len(export.lines) == 3
    for line in export.lines:
        assert line.startswith("export ")
        key, _, quoted = line.removeprefix("export ").partition("=")
        assert key in {"DEVICE_TYPE", "BASE_IMAGE", "BUILD_TYPE"}
        assert shlex.split(quoted) == [export.build_args.build_args[key]]


def test_shell_export_prefix(force_cpu) -> None:
    export = build_args_shell_export(device_type="cpu", prefix="CYREX_")
    keys = {
        line.removeprefix("export ").split("=", 1)[0].removeprefix("CYREX_")
        for line in export.lines
    }
    assert keys == {"DEVICE_TYPE", "BASE_IMAGE", "BUILD_TYPE"}


def test_export_env_cli(force_cpu, capsys) -> None:
    from deepiri_gpu_utils.cli import main

    rc = main(["export-env", "--device-type", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 3
    assert all(line.startswith("export ") for line in out)
    keys = {line.removeprefix("export ").split("=", 1)[0] for line in out}
    assert keys == {"DEVICE_TYPE", "BASE_IMAGE", "BUILD_TYPE"}
