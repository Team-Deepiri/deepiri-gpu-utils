"""Lock the public API surface: classes, functions, dataclass fields, constants.

This is a pure regression guard. If a future change renames or removes any of
these public names (or reorders/renames dataclass fields, which are also JSON
keys), this test fails loudly so the change is a deliberate, reviewed decision.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass

from deepiri_gpu_utils import build_args, cli, detect, doctor, ollama, setup, system_info
from deepiri_gpu_utils import torch_device as td


def _field_names(cls: type) -> list[str]:
    assert is_dataclass(cls)
    return [f.name for f in fields(cls)]


def test_detect_public_api() -> None:
    assert callable(detect.detect)
    assert callable(detect.query_nvidia_smi)
    assert callable(detect.query_rocm_smi)
    assert is_dataclass(detect.DetectResult)
    assert _field_names(detect.DetectResult) == ["backend", "confidence", "details", "warnings"]
    # detect() must be keyword-callable with `prefer`
    assert "prefer" in inspect.signature(detect.detect).parameters


def test_build_args_public_api() -> None:
    assert callable(build_args.build_args_from_detection)
    assert callable(build_args.infer_device_type_from_base_image)
    assert _field_names(build_args.BuildArgs) == [
        "device_type",
        "base_image",
        "cuda_version",
        "python_version",
        "build_args",
        "warnings",
    ]
    assert build_args.DEFAULT_PYTORCH_GPU_IMAGE
    assert build_args.DEFAULT_CPU_IMAGE == "python:3.11-slim"
    assert build_args.DEFAULT_PYTHON_VERSION == "3.11"
    assert build_args.DEFAULT_CUDA_VERSION_GPU == "12.8"


def test_doctor_public_api() -> None:
    assert callable(doctor.doctor)
    assert _field_names(doctor.DoctorReport) == ["detect", "status", "findings", "runbook"]


def test_ollama_public_api() -> None:
    assert callable(ollama.recommend_models)
    assert callable(ollama.setup_tier)
    assert callable(ollama.categorize_model)
    assert _field_names(ollama.OllamaRecommendation) == [
        "default_model",
        "recommended_models",
        "usable_models",
        "marginal_models",
        "unsuitable_models",
        "notes",
        "setup_tier",
        "system_ram_gb",
        "effective_vram_gb",
        "category",
    ]


def test_setup_public_api() -> None:
    assert callable(setup.setup_device)
    assert callable(setup.setup_device_mac)
    assert _field_names(setup.SetupPlan) == ["device", "dry_run", "runbook"]


def test_torch_device_public_api() -> None:
    assert callable(td.resolve_torch_device)
    assert _field_names(td.DeviceDecision) == ["device", "notes", "torch_available"]


def test_system_info_public_api() -> None:
    for name in (
        "is_wsl",
        "system_ram_gb",
        "lspci_nvidia_present",
        "docker_cli_available",
        "nvidia_container_toolkit_hint",
        "dmidecode_inventory",
    ):
        assert callable(getattr(system_info, name)), name


def test_cli_public_api() -> None:
    assert callable(cli.build_parser)
    assert callable(cli.main)
