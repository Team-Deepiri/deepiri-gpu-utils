"""Behavioral coverage for the canonical reusable GPU primitives."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import deepiri_gpu_utils.canonical as api
from deepiri_gpu_utils import (
    GpuBackend,
    GpuDevice,
    GpuHealth,
    GpuHealthStatus,
    GpuInventory,
    GpuMemory,
    GpuSelectionPolicy,
    GpuSelectionResult,
    RuntimeCapabilities,
    check_gpu_health,
    detect_backend,
    discover_gpus,
    resolve_runtime,
    select_gpu,
)
from deepiri_gpu_utils._subprocess import RunResult
from deepiri_gpu_utils.detect import DetectResult


def _result(stdout: str = "", *, ok: bool = True) -> RunResult:
    return RunResult(ok=ok, returncode=0 if ok else 1, stdout=stdout, stderr="")


def _device(
    index: int,
    *,
    total: int | None = 8192,
    free: int | None = 4096,
    utilization: float | None = 20,
    backend: GpuBackend = GpuBackend.CUDA,
) -> GpuDevice:
    return GpuDevice(
        backend=backend,
        index=index,
        name=f"GPU {index}",
        memory=GpuMemory(total_mib=total, free_mib=free),
        utilization_percent=utilization,
        source="test",
    )


def test_primitives_are_immutable_and_metadata_is_deeply_frozen() -> None:
    device = GpuDevice(
        backend=GpuBackend.CUDA,
        metadata={"nested": {"values": [1, 2]}},
    )
    with pytest.raises(FrozenInstanceError):
        device.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        device.metadata["new"] = True  # type: ignore[index]
    assert device.metadata["nested"]["values"] == (1, 2)
    assert GpuInventory().devices == ()


def test_device_hash_is_stable_and_metadata_is_explicitly_json_only() -> None:
    left = GpuDevice(backend=GpuBackend.CUDA, metadata={"kind": GpuBackend.ROCM})
    right = GpuDevice.from_dict(left.to_dict())
    assert left == right
    assert hash(left) == hash(right)
    assert left.to_dict()["metadata"] == {"kind": "rocm"}

    with pytest.raises(TypeError):
        GpuDevice(backend=GpuBackend.CUDA, metadata={"bad": object()})
    with pytest.raises(TypeError):
        GpuDevice(backend=GpuBackend.CUDA, metadata={1: "non-string key"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        GpuDevice(backend=GpuBackend.CUDA, metadata={"bad": float("nan")})


def test_memory_binary_conversions_and_derived_usage() -> None:
    memory = GpuMemory(total_mib=12288, free_mib=4096)
    assert memory.total_gib == 12
    assert memory.free_gib == 4
    assert memory.calculated_used_mib == 8192
    assert GpuMemory.gib_to_mib(1.5) == 1536
    assert GpuMemory.mib_to_gib(None) is None
    with pytest.raises(ValueError):
        GpuMemory.gib_to_mib(float("inf"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_mib": -1},
        {"total_mib": 10, "free_mib": 11},
        {"total_mib": 10, "used_mib": 11},
    ],
)
def test_memory_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        GpuMemory(**kwargs)


def test_nested_serialization_round_trip_preserves_optional_values_and_metadata() -> None:
    device = GpuDevice(
        backend=GpuBackend.ROCM,
        index=2,
        name="MI300X",
        uuid=None,
        memory=GpuMemory(total_mib=196608, used_mib=1024),
        metadata={"raw": {"family": "gfx942"}, "flags": ["a", "b"]},
    )
    inventory = GpuInventory(
        devices=(device,), backend=GpuBackend.ROCM, source="rocm-smi", warnings=("partial",)
    )
    assert GpuInventory.from_dict(inventory.to_dict()) == inventory

    health = GpuHealth(
        healthy=True,
        status=GpuHealthStatus.HEALTHY,
        reason="ok",
        devices=(device,),
    )
    assert GpuHealth.from_dict(health.to_dict()) == health

    selection = GpuSelectionResult(device, True, "selected", (device,))
    assert GpuSelectionResult.from_dict(selection.to_dict()) == selection

    runtime = RuntimeCapabilities(
        backend=GpuBackend.ROCM,
        hardware_detected=True,
        tooling_detected=True,
        torch_installed=False,
        torch_usable=False,
        cuda_usable=False,
        rocm_usable=False,
        mps_usable=False,
        warnings=("torch absent",),
    )
    assert RuntimeCapabilities.from_dict(runtime.to_dict()) == runtime


def test_policy_serialization_round_trip() -> None:
    policy = GpuSelectionPolicy(
        preferred_backend=GpuBackend.CUDA,
        minimum_total_mib=8192,
        minimum_free_mib=4096,
        maximum_utilization_percent=50,
        require_healthy=True,
    )
    assert GpuSelectionPolicy.from_dict(policy.to_dict()) == policy


def test_detect_backend_returns_enum_and_unknown_is_safe(monkeypatch) -> None:
    monkeypatch.setattr(api, "detect", lambda **_: DetectResult("cuda"))
    assert detect_backend() is GpuBackend.CUDA
    assert str(detect_backend()) == "cuda"
    monkeypatch.setattr(api, "detect", lambda **_: DetectResult("future"))
    assert detect_backend() is GpuBackend.UNKNOWN


def test_cuda_discovery_normalizes_multiple_devices(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda name: name if name == "nvidia-smi" else None)
    inventory_rows = (
        "1, RTX 4090, GPU-b, 24576, 20000, 4576, 10, 41, 72.5, 560.1\n"
        "0, RTX 3090, GPU-a, 24576, 12000, 12576, 55, 70, 300.0, 560.1\n"
    )

    def fake_run(command, *, timeout):
        if "compute_cap" in command[1]:
            return _result("0, 8.6\n1, 8.9\n")
        return _result(inventory_rows)

    monkeypatch.setattr(api, "run_text", fake_run)
    found = discover_gpus()
    assert found.backend is GpuBackend.CUDA
    assert found.count == 2
    assert found.devices[0].uuid == "GPU-b"
    assert found.devices[0].memory.free_mib == 20000
    assert found.devices[1].compute_capability == "8.6"
    assert found.devices[1].temperature_c == 70
    assert found.devices[1].power_watts == 300


def test_cuda_discovery_skips_malformed_and_keeps_partial_data(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda name: name if name == "nvidia-smi" else None)

    def fake_run(command, *, timeout):
        if "compute_cap" in command[1]:
            return _result(ok=False)
        return _result("bad\n0, Partial GPU, GPU-0, [N/A], 2000, [N/A], [N/A]\n")

    monkeypatch.setattr(api, "run_text", fake_run)
    found = discover_gpus()
    assert found.count == 1
    assert found.devices[0].memory.total_mib is None
    assert found.devices[0].utilization_percent is None
    assert "unparsable" in found.warnings[0]


def test_cuda_discovery_falls_back_when_optional_metrics_are_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda name: name if name == "nvidia-smi" else None)

    def fake_run(command, *, timeout):
        query = command[1]
        if "power.draw" in query or "compute_cap" in query:
            return _result(ok=False)
        return _result("0, Compatible GPU, 8192, 4096, 12, 550.90\n")

    monkeypatch.setattr(api, "run_text", fake_run)
    found = discover_gpus()
    assert found.count == 1
    assert found.devices[0].memory.total_mib == 8192
    assert found.devices[0].driver_version == "550.90"
    assert found.devices[0].uuid is None
    assert "compatibility inventory query" in found.warnings[0]


def test_missing_and_failed_tooling_return_valid_inventories(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda _: None)
    monkeypatch.setattr(api, "detect_backend", lambda: GpuBackend.CPU)
    cpu = discover_gpus()
    assert cpu.backend is GpuBackend.CPU
    assert not cpu.has_gpu

    monkeypatch.setattr(api.shutil, "which", lambda name: name if name == "nvidia-smi" else None)
    monkeypatch.setattr(api, "run_text", lambda *args, **kwargs: _result(ok=False))
    failed = discover_gpus()
    assert not failed.has_gpu
    assert any("failed or timed out" in warning for warning in failed.warnings)


def test_rocm_structured_discovery_preserves_metadata(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda name: name if name == "rocm-smi" else None)
    payload = """{
      "system": {"Driver version": "6.1.2"},
      "card0": {
        "Card series": "AMD Instinct MI250X",
        "Unique ID": "abc",
        "VRAM Total Memory (B)": 68719476736,
        "VRAM Total Used Memory (B)": 1073741824,
        "GPU use (%)": "25",
        "Temperature (Sensor edge) (C)": "52.0",
        "Average Graphics Package Power (W)": "180.5"
      }
    }"""
    monkeypatch.setattr(api, "run_text", lambda *args, **kwargs: _result(payload))
    found = discover_gpus()
    device = found.devices[0]
    assert found.backend is GpuBackend.ROCM
    assert device.name == "AMD Instinct MI250X"
    assert device.memory.total_mib == 65536
    assert device.memory.free_mib == 64512
    assert device.utilization_percent == 25
    assert device.driver_version == "6.1.2"
    assert device.metadata["Unique ID"] == "abc"


def test_rocm_malformed_output_is_non_throwing(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda name: name if name == "rocm-smi" else None)
    monkeypatch.setattr(api, "run_text", lambda *args, **kwargs: _result("not-json"))
    monkeypatch.setattr(api, "detect_backend", lambda: GpuBackend.ROCM)
    found = discover_gpus()
    assert found.devices == ()
    assert "malformed JSON" in found.warnings[0]


def test_rocm_discovery_falls_back_when_optional_metrics_are_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda name: name if name == "rocm-smi" else None)
    payload = '{"card1": {"Card series": "MI210", "GPU use (%)": "3"}}'

    def fake_run(command, *, timeout):
        return _result(ok=False) if "--showtemp" in command else _result(payload)

    monkeypatch.setattr(api, "run_text", fake_run)
    found = discover_gpus()
    assert found.count == 1
    assert found.devices[0].index == 1
    assert found.devices[0].name == "MI210"
    assert "compatibility inventory query" in found.warnings[0]


def test_mps_is_a_valid_limited_inventory(monkeypatch) -> None:
    monkeypatch.setattr(api.shutil, "which", lambda _: None)
    monkeypatch.setattr(api, "detect_backend", lambda: GpuBackend.MPS)
    monkeypatch.setattr(api.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(api.platform, "machine", lambda: "arm64")
    found = discover_gpus()
    assert found.has_gpu
    assert found.devices[0].backend is GpuBackend.MPS
    assert found.devices[0].memory.total_mib is None


def test_health_distinguishes_no_gpu_tooling_unavailable_and_unhealthy(monkeypatch) -> None:
    no_gpu = check_gpu_health(inventory=GpuInventory(backend=GpuBackend.CPU))
    assert no_gpu.status is GpuHealthStatus.NO_GPU

    unavailable = check_gpu_health(inventory=GpuInventory(backend=GpuBackend.CUDA))
    assert unavailable.status is GpuHealthStatus.TOOLING_UNAVAILABLE

    unknown = check_gpu_health(inventory=GpuInventory(backend=GpuBackend.UNKNOWN))
    assert unknown.status is GpuHealthStatus.TOOLING_UNAVAILABLE

    report = types.SimpleNamespace(
        status="fail",
        checks=(types.SimpleNamespace(status="fail", message="driver failed"),),
    )
    monkeypatch.setattr(api, "_legacy_health_check", lambda: report)
    unhealthy = check_gpu_health(
        inventory=GpuInventory((_device(0),), GpuBackend.CUDA, "nvidia-smi")
    )
    assert unhealthy.status is GpuHealthStatus.UNHEALTHY
    assert unhealthy.errors == ("driver failed",)


def test_health_classifies_malformed_tool_response_as_tooling_unavailable() -> None:
    malformed = GpuInventory(
        backend=GpuBackend.CPU,
        warnings=("rocm-smi returned malformed JSON",),
    )
    health = check_gpu_health(inventory=malformed)
    assert health.status is GpuHealthStatus.TOOLING_UNAVAILABLE
    assert "invalid response" in health.reason


def test_health_reports_healthy_with_warning_details(monkeypatch) -> None:
    report = types.SimpleNamespace(
        status="warn",
        checks=(types.SimpleNamespace(status="warn", message="optional tool missing"),),
    )
    monkeypatch.setattr(api, "_legacy_health_check", lambda: report)
    health = check_gpu_health(
        inventory=GpuInventory((_device(0),), GpuBackend.CUDA, "nvidia-smi")
    )
    assert health.healthy
    assert health.warnings == ("optional tool missing",)


def test_selection_is_deterministic_and_uses_free_memory() -> None:
    inventory = GpuInventory(
        (_device(2, free=6000), _device(0, free=6000), _device(1, free=7000)),
        GpuBackend.CUDA,
    )
    result = select_gpu(inventory=inventory)
    assert result.suitable
    assert result.selected is not None and result.selected.index == 1

    tie = GpuInventory((_device(2), _device(0)), GpuBackend.CUDA)
    assert select_gpu(inventory=tie).selected.index == 0  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("policy", "needle"),
    [
        (GpuSelectionPolicy(minimum_total_mib=9000), "total VRAM"),
        (GpuSelectionPolicy(minimum_free_mib=5000), "free VRAM"),
        (GpuSelectionPolicy(maximum_utilization_percent=10), "utilization"),
    ],
)
def test_selection_rejects_threshold_failures(policy, needle) -> None:
    result = select_gpu(policy, inventory=GpuInventory((_device(0),), GpuBackend.CUDA))
    assert not result.suitable
    assert needle in result.reason


def test_selection_rejects_missing_required_metrics_and_falls_back_from_preference() -> None:
    missing = _device(0, total=None, free=None, utilization=None)
    result = select_gpu(
        GpuSelectionPolicy(minimum_free_mib=1),
        inventory=GpuInventory((missing,), GpuBackend.CUDA),
    )
    assert "unavailable" in result.reason

    fallback = select_gpu(
        GpuSelectionPolicy(preferred_backend=GpuBackend.ROCM),
        inventory=GpuInventory((_device(0),), GpuBackend.CUDA),
    )
    assert fallback.suitable
    assert fallback.selected is not None and fallback.selected.backend is GpuBackend.CUDA
    assert "fallback" in fallback.reason


def test_selection_prioritizes_preferred_backend_before_capacity() -> None:
    cuda = _device(0, total=24576, free=20000)
    rocm = _device(1, free=1000, backend=GpuBackend.ROCM)
    result = select_gpu(
        GpuSelectionPolicy(preferred_backend=GpuBackend.ROCM),
        inventory=GpuInventory((cuda, rocm), GpuBackend.UNKNOWN),
    )
    assert result.selected == rocm


def test_selection_honors_healthy_only() -> None:
    device = _device(0)
    health = GpuHealth(False, GpuHealthStatus.UNHEALTHY, "temperature alarm")
    result = select_gpu(
        GpuSelectionPolicy(require_healthy=True),
        inventory=GpuInventory((device,), GpuBackend.CUDA),
        health=health,
    )
    assert not result.suitable
    assert "temperature alarm" in result.reason


def test_runtime_torch_absent_and_broken_are_explicit(monkeypatch) -> None:
    inventory = GpuInventory(backend=GpuBackend.CPU)
    monkeypatch.setattr(api, "_torch_package_present", lambda: False)
    monkeypatch.setattr(
        api.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(ImportError()),
    )
    absent = resolve_runtime(inventory=inventory)
    assert not absent.torch_installed
    assert not absent.torch_usable

    monkeypatch.setattr(api, "_torch_package_present", lambda: True)
    monkeypatch.setattr(api.importlib, "import_module", lambda _: (_ for _ in ()).throw(OSError()))
    broken = resolve_runtime(inventory=inventory)
    assert broken.torch_installed
    assert not broken.torch_usable
    assert any("OSError" in warning for warning in broken.warnings)


@pytest.mark.parametrize(
    ("hip", "cuda_expected", "rocm_expected"),
    [(None, True, False), ("6.1", False, True)],
)
def test_runtime_distinguishes_cuda_and_rocm_torch(
    monkeypatch, hip, cuda_expected, rocm_expected
) -> None:
    torch = types.SimpleNamespace(
        __version__="2.5",
        version=types.SimpleNamespace(cuda="12.4", hip=hip),
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(api.importlib, "import_module", lambda _: torch)
    runtime = resolve_runtime(
        inventory=GpuInventory((_device(0),), GpuBackend.CUDA, "nvidia-smi")
    )
    assert runtime.torch_usable
    assert runtime.cuda_usable is cuda_expected
    assert runtime.rocm_usable is rocm_expected


def test_runtime_detects_mps_and_survives_broken_probe(monkeypatch) -> None:
    torch = types.SimpleNamespace(
        __version__="2.5",
        version=types.SimpleNamespace(cuda=None, hip=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(api.importlib, "import_module", lambda _: torch)
    inventory = GpuInventory((_device(0, backend=GpuBackend.MPS),), GpuBackend.MPS, "platform")
    assert resolve_runtime(inventory=inventory).mps_usable

    torch.cuda.is_available = lambda: (_ for _ in ()).throw(RuntimeError())
    broken = resolve_runtime(inventory=inventory)
    assert not broken.torch_usable
    assert any("runtime probe failed" in warning for warning in broken.warnings)


def test_canonical_public_imports_are_callable() -> None:
    assert all(
        callable(function)
        for function in (
            detect_backend,
            discover_gpus,
            select_gpu,
            check_gpu_health,
            resolve_runtime,
        )
    )


def test_package_import_never_imports_optional_torch() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "src"), env.get("PYTHONPATH")) if part
    )
    script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("torch imported at package import time")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import deepiri_gpu_utils
print(deepiri_gpu_utils.__version__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.3.0"
