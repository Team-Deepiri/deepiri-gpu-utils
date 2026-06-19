"""Regression tests for :mod:`deepiri_gpu_utils.compose_gpu`."""

from __future__ import annotations

import json

from deepiri_gpu_utils.cli import main
from deepiri_gpu_utils.compose_gpu import compose_gpu_config


def test_compose_gpu_cuda_has_nvidia_deploy(force_cpu) -> None:
    cfg = compose_gpu_config(backend="cuda")
    assert cfg.backend == "cuda"
    assert cfg.deploy_devices == ["nvidia"]
    assert cfg.device_requests


def test_compose_gpu_rocm_has_device_args(force_cpu) -> None:
    cfg = compose_gpu_config(backend="rocm")
    assert "/dev/kfd" in cfg.run_gpu_args


def test_compose_gpu_json_cli(force_cpu, capsys) -> None:
    rc = main(["compose-gpu", "--backend", "cpu", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {
        "backend",
        "deploy_devices",
        "device_requests",
        "run_gpu_args",
        "environment",
        "notes",
    }
