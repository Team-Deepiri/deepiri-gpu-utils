"""Lock the public API surface: classes, functions, dataclass fields, constants."""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass

from deepiri_gpu_utils import (
    build_args,
    cli,
    compose_gpu,
    detect,
    doctor,
    env_hints,
    export_env,
    gpu_top,
    hardware,
    health,
    install_check,
    inventory,
    profiles,
    setup,
    snapshot,
    stress_test,
    summary,
    system_info,
    visualize,
)
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
    assert "prefer" in inspect.signature(detect.detect).parameters


def test_hardware_public_api() -> None:
    assert callable(hardware.effective_vram_gb)
    assert callable(hardware.apply_backend_hint_vram)


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


def test_doctor_public_api() -> None:
    assert callable(doctor.doctor)
    assert _field_names(doctor.DoctorReport) == ["detect", "status", "findings", "runbook"]


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
        "lspci_amd_present",
        "docker_cli_available",
        "nvidia_container_toolkit_hint",
        "dmidecode_inventory",
    ):
        assert callable(getattr(system_info, name)), name


def test_cli_public_api() -> None:
    assert callable(cli.build_parser)
    assert callable(cli.main)


def test_inventory_public_api() -> None:
    assert callable(inventory.gpu_inventory)
    assert callable(inventory.choose_suitable_gpu)


def test_summary_public_api() -> None:
    assert callable(summary.hardware_summary)
    assert _field_names(summary.HardwareSummary) == [
        "detect",
        "doctor",
        "inventory",
        "build_args",
        "torch_device",
        "system_ram_gb",
        "gpu_count",
        "total_vram_gb",
        "notes",
    ]


def test_export_env_public_api() -> None:
    assert callable(export_env.build_args_shell_export)
    assert _field_names(export_env.ShellExport) == ["lines", "build_args"]


def test_profiles_public_api() -> None:
    assert callable(profiles.backend_profile)
    assert callable(profiles.all_backend_profiles)


def test_install_check_public_api() -> None:
    assert callable(install_check.install_readiness)
    assert callable(install_check.install_readiness_all)


def test_compose_gpu_public_api() -> None:
    assert callable(compose_gpu.compose_gpu_config)


def test_visualize_public_api() -> None:
    assert callable(visualize.render_dashboard)
    assert callable(visualize.render_html_report)


def test_stress_test_public_api() -> None:
    assert callable(stress_test.run_stress_test)
    assert callable(stress_test.render_stress_text)


def test_health_public_api() -> None:
    assert callable(health.health_check)


def test_env_hints_public_api() -> None:
    assert callable(env_hints.runtime_env_hints)


def test_gpu_top_public_api() -> None:
    assert callable(gpu_top.gpu_top)


def test_snapshot_public_api() -> None:
    assert callable(snapshot.capture_snapshot)
    assert callable(snapshot.save_snapshot)
    assert callable(snapshot.load_snapshot)
    assert callable(snapshot.diff_snapshots)
    assert callable(snapshot.render_diff_text)
