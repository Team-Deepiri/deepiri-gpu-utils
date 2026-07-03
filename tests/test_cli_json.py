"""CLI parser smoke tests and JSON-shape regression tests."""

from __future__ import annotations

import json

import pytest

from deepiri_gpu_utils import cli
from deepiri_gpu_utils.cli import build_parser, main

EXPECTED_COMMANDS = [
    ["detect", "--json"],
    ["detect", "--prefer", "cuda"],
    ["doctor", "--json"],
    ["setup", "--device", "auto"],
    ["setup", "--device", "nvidia", "--yes"],
    ["build-args", "--json"],
    ["build-args", "--device-type", "gpu"],
    ["build-args", "--base-image-only"],
    ["validate", "--json"],
    ["torch-device", "--policy", "auto", "--json"],
    ["inventory", "--json"],
    ["inventory", "--min-memory-gb", "8", "--json"],
    ["summary", "--json"],
    ["export-env", "--device-type", "cpu"],
    ["export-env", "--prefix", "CYREX_"],
    ["install-check", "--device", "cpu", "--json"],
    ["install-check", "--all", "--json"],
    ["profile", "--all", "--json"],
    ["profile", "--backend", "rocm", "--json"],
    ["compose-gpu", "--backend", "cuda", "--json"],
    ["visualize", "--json"],
    ["stress", "--mode", "probes", "--duration", "0.5", "--json"],
    ["health", "--json"],
    ["env-hints", "--backend", "cpu", "--json"],
    ["top", "--json"],
    ["snapshot", "save", "/tmp/deepiri-gpu-test-snapshot.json", "--json"],
]


@pytest.mark.parametrize("argv", EXPECTED_COMMANDS)
def test_parser_accepts_known_commands(argv: list[str]) -> None:
    assert build_parser().parse_args(argv) is not None


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize(
    "argv",
    [
        ["setup", "--device", "bogus"],
        ["build-args", "--device-type", "bogus"],
        ["torch-device", "--policy", "bogus"],
        ["totally-unknown"],
    ],
)
def test_parser_rejects_invalid_args(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def _json_out(capsys, argv: list[str]) -> dict:
    rc = main(argv)
    assert rc == 0
    return json.loads(capsys.readouterr().out)


def test_detect_json_shape(force_cpu, capsys) -> None:
    payload = _json_out(capsys, ["detect", "--json"])
    assert set(payload) == {"backend", "confidence", "details", "warnings"}
    assert payload["backend"] == "cpu"
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["details"], dict)


def test_doctor_json_shape(force_cpu, capsys) -> None:
    payload = _json_out(capsys, ["doctor", "--json"])
    assert set(payload) == {"detect", "findings", "runbook", "status"}
    assert set(payload["detect"]) == {"backend", "confidence", "details", "warnings"}
    assert payload["status"] in {"ok", "warn", "unknown"}
    assert isinstance(payload["runbook"], list)


def test_build_args_json_shape(force_cpu, capsys) -> None:
    payload = _json_out(capsys, ["build-args", "--json"])
    assert set(payload) == {
        "device_type",
        "base_image",
        "cuda_version",
        "python_version",
        "build_args",
        "warnings",
    }
    assert payload["device_type"] == "cpu"
    assert payload["base_image"] == "python:3.11-slim"
    assert set(payload["build_args"]) == {"DEVICE_TYPE", "BASE_IMAGE", "BUILD_TYPE"}
    assert payload["build_args"]["BUILD_TYPE"] == "prebuilt"


def test_validate_json_shape(force_cpu, capsys) -> None:
    payload = _json_out(capsys, ["validate", "--json"])
    assert set(payload) == {"detect", "doctor", "build_args", "torch_device"}
    assert payload["detect"]["backend"] == "cpu"


def test_torch_device_json_shape(force_cpu, capsys) -> None:
    payload = _json_out(capsys, ["torch-device", "--policy", "cpu", "--json"])
    assert set(payload) == {"device", "notes", "torch_available"}
    assert payload["device"] == "cpu"
    assert isinstance(payload["torch_available"], bool)


def test_build_args_base_image_only_is_single_line(force_cpu, capsys) -> None:
    rc = main(["build-args", "--base-image-only"])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0] == "python:3.11-slim"


def test_to_jsonable_normalizes_nested_structures() -> None:
    from dataclasses import dataclass

    @dataclass
    class Inner:
        a: int

    @dataclass
    class Outer:
        inner: Inner
        items: tuple

    out = cli._to_jsonable(Outer(inner=Inner(a=1), items=(1, 2)))
    assert out == {"inner": {"a": 1}, "items": [1, 2]}
