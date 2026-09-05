"""Canonical backend, inventory, health, selection, and runtime APIs."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import io
import json
import math
import platform
import re
import shutil
from collections.abc import Mapping
from typing import Any

from ._subprocess import run_text
from .detect import detect
from .health import health_check as _legacy_health_check
from .primitives import (
    GpuBackend,
    GpuDevice,
    GpuHealth,
    GpuHealthStatus,
    GpuInventory,
    GpuMemory,
    GpuSelectionPolicy,
    GpuSelectionResult,
    RuntimeCapabilities,
)

_NVIDIA_QUERY = (
    "index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu,"
    "temperature.gpu,power.draw,driver_version"
)
_NVIDIA_FALLBACK_QUERY = (
    "index,name,memory.total,memory.free,utilization.gpu,driver_version"
)
_NVIDIA_COMPUTE_QUERY = "index,compute_cap"
_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _safe_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _NUMBER_RE.search(str(value))
        if match is None:
            return None
        try:
            number = float(match.group())
        except ValueError:
            return None
    return number if math.isfinite(number) and number >= 0 else None


def _safe_int(value: Any) -> int | None:
    number = _safe_number(value)
    return None if number is None else int(number)


def _safe_percent(value: Any) -> float | None:
    number = _safe_number(value)
    return number if number is not None and number <= 100 else None


def _parse_csv(line: str) -> list[str]:
    try:
        return [cell.strip() for cell in next(csv.reader(io.StringIO(line)))]
    except (csv.Error, StopIteration):
        return []


def _nvidia_compute_capabilities() -> dict[int, str]:
    result = run_text(
        [
            "nvidia-smi",
            f"--query-gpu={_NVIDIA_COMPUTE_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if not result.ok:
        return {}
    capabilities: dict[int, str] = {}
    for line in result.stdout.splitlines():
        cells = _parse_csv(line)
        index = _safe_int(cells[0]) if cells else None
        if index is not None and len(cells) > 1 and cells[1] and not cells[1].startswith("["):
            capabilities[index] = cells[1]
    return capabilities


def _discover_nvidia() -> tuple[list[GpuDevice], list[str], bool]:
    if shutil.which("nvidia-smi") is None:
        return [], [], False
    result = run_text(
        ["nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
        timeout=15,
    )
    warnings: list[str] = []
    detailed = result.ok
    if not detailed:
        result = run_text(
            [
                "nvidia-smi",
                f"--query-gpu={_NVIDIA_FALLBACK_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
        if not result.ok:
            return [], ["nvidia-smi inventory queries failed or timed out"], True
        warnings.append(
            "nvidia-smi detailed metrics unavailable; used compatibility inventory query"
        )

    capabilities = _nvidia_compute_capabilities()
    devices: list[GpuDevice] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        cells = _parse_csv(raw_line)
        if len(cells) < 2 or not cells[1]:
            warnings.append(f"Ignored unparsable nvidia-smi row: {raw_line.strip()!r}")
            continue
        expected_cells = 10 if detailed else 6
        cells.extend([""] * (expected_cells - len(cells)))
        index = _safe_int(cells[0])
        if detailed:
            uuid = cells[2] or None
            total = _safe_int(cells[3])
            free = _safe_int(cells[4])
            used = _safe_int(cells[5])
            utilization = _safe_percent(cells[6])
            temperature = _safe_number(cells[7])
            power = _safe_number(cells[8])
            driver = cells[9] or None
        else:
            uuid = None
            total = _safe_int(cells[2])
            free = _safe_int(cells[3])
            used = None
            utilization = _safe_percent(cells[4])
            temperature = None
            power = None
            driver = cells[5] or None
        if total is not None:
            free = free if free is None or free <= total else None
            used = used if used is None or used <= total else None
        devices.append(
            GpuDevice(
                backend=GpuBackend.CUDA,
                index=index,
                name=cells[1] or None,
                uuid=uuid,
                memory=GpuMemory(total_mib=total, free_mib=free, used_mib=used),
                utilization_percent=utilization,
                temperature_c=temperature,
                power_watts=power,
                driver_version=driver,
                compute_capability=capabilities.get(index) if index is not None else None,
                source="nvidia-smi",
            )
        )
    if not devices:
        warnings.append("nvidia-smi returned no GPU rows")
    return devices, warnings, True


def _lookup_metric(data: Mapping[str, Any], *required: str) -> Any:
    required_lower = tuple(token.lower() for token in required)
    for key, value in data.items():
        key_lower = str(key).lower()
        if all(token in key_lower for token in required_lower):
            return value
    return None


def _bytes_to_mib(value: Any) -> int | None:
    number = _safe_number(value)
    return None if number is None else int(number // (1024**2))


def _rocm_device(
    card_name: str,
    data: Mapping[str, Any],
    global_driver: str | None,
) -> GpuDevice:
    index_match = re.search(r"\d+", card_name)
    total_bytes = _safe_number(_lookup_metric(data, "vram", "total", "memory"))
    used_bytes = _safe_number(_lookup_metric(data, "vram", "used", "memory"))
    total = _bytes_to_mib(total_bytes)
    used = _bytes_to_mib(used_bytes)
    free = (
        _bytes_to_mib(total_bytes - used_bytes)
        if total_bytes is not None and used_bytes is not None and used_bytes <= total_bytes
        else None
    )
    name = _lookup_metric(data, "card", "series") or _lookup_metric(data, "product", "name")
    uuid = _lookup_metric(data, "unique", "id") or _lookup_metric(data, "uuid")
    driver = _lookup_metric(data, "driver", "version")
    temperature = _lookup_metric(data, "temperature")
    power = _lookup_metric(data, "power")
    return GpuDevice(
        backend=GpuBackend.ROCM,
        index=int(index_match.group()) if index_match else None,
        name=str(name) if name is not None else None,
        uuid=str(uuid) if uuid is not None else None,
        memory=GpuMemory(total_mib=total, free_mib=free, used_mib=used),
        utilization_percent=_safe_percent(_lookup_metric(data, "gpu", "use")),
        temperature_c=_safe_number(temperature),
        power_watts=_safe_number(power),
        driver_version=str(driver) if driver is not None else global_driver,
        source="rocm-smi",
        metadata=data,
    )


def _discover_rocm() -> tuple[list[GpuDevice], list[str], bool]:
    if shutil.which("rocm-smi") is None:
        return [], [], False
    result = run_text(
        [
            "rocm-smi",
            "--showproductname",
            "--showmeminfo",
            "vram",
            "--showuse",
            "--showtemp",
            "--showpower",
            "--showdriverversion",
            "--json",
        ],
        timeout=20,
    )
    warnings: list[str] = []
    if not result.ok:
        result = run_text(
            [
                "rocm-smi",
                "--showproductname",
                "--showmeminfo",
                "vram",
                "--showuse",
                "--json",
            ],
            timeout=20,
        )
        if not result.ok:
            return [], ["rocm-smi inventory queries failed or timed out"], True
        warnings.append(
            "rocm-smi optional metrics unavailable; used compatibility inventory query"
        )
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return [], ["rocm-smi returned malformed JSON"], True
    if not isinstance(payload, Mapping):
        return [], ["rocm-smi JSON was not an object"], True

    global_driver_value = _lookup_metric(payload, "driver", "version")
    if global_driver_value is None:
        global_driver_value = next(
            (
                _lookup_metric(data, "driver", "version")
                for card, data in payload.items()
                if not str(card).lower().startswith("card") and isinstance(data, Mapping)
            ),
            None,
        )
    global_driver = str(global_driver_value) if global_driver_value is not None else None
    devices = [
        _rocm_device(str(card), data, global_driver)
        for card, data in payload.items()
        if str(card).lower().startswith("card") and isinstance(data, Mapping)
    ]
    if not devices:
        warnings.append("rocm-smi returned no card objects")
    return devices, warnings, True


def detect_backend(*, prefer: str | GpuBackend | None = None) -> GpuBackend:
    """Return the best detected backend as a stable enum."""

    preferred = prefer.value if isinstance(prefer, GpuBackend) else prefer
    result = detect(prefer=preferred)
    try:
        return GpuBackend(result.backend)
    except ValueError:
        return GpuBackend.UNKNOWN


def discover_gpus() -> GpuInventory:
    """Discover and normalize GPUs without raising for host-tool failures."""

    nvidia, warnings, nvidia_tool = _discover_nvidia()
    if nvidia:
        return GpuInventory(
            devices=tuple(nvidia),
            backend=GpuBackend.CUDA,
            source="nvidia-smi",
            warnings=tuple(warnings),
        )

    rocm, rocm_warnings, rocm_tool = _discover_rocm()
    warnings.extend(rocm_warnings)
    if rocm:
        return GpuInventory(
            devices=tuple(rocm),
            backend=GpuBackend.ROCM,
            source="rocm-smi",
            warnings=tuple(warnings),
        )

    backend = detect_backend()
    if backend == GpuBackend.MPS:
        return GpuInventory(
            devices=(
                GpuDevice(
                    backend=GpuBackend.MPS,
                    index=0,
                    name="Apple MPS",
                    source="platform",
                    metadata={"platform": platform.system(), "machine": platform.machine()},
                ),
            ),
            backend=GpuBackend.MPS,
            source="platform",
            warnings=tuple(warnings),
        )

    if backend in (GpuBackend.CUDA, GpuBackend.ROCM):
        tool = "nvidia-smi" if backend == GpuBackend.CUDA else "rocm-smi"
        if not (nvidia_tool if backend == GpuBackend.CUDA else rocm_tool):
            warnings.append(f"{backend.value} hardware detected but {tool} is unavailable")
    elif not warnings:
        warnings.append("No GPU detected; returning a valid CPU-only inventory")
    return GpuInventory(devices=(), backend=backend, source=None, warnings=tuple(warnings))


def check_gpu_health(*, inventory: GpuInventory | None = None) -> GpuHealth:
    """Return canonical health with distinct no-GPU and tooling states."""

    current = inventory if inventory is not None else discover_gpus()
    if not current.devices:
        tool_response_failed = any(
            ("nvidia-smi" in warning or "rocm-smi" in warning)
            and any(word in warning for word in ("failed", "malformed", "no GPU", "no card"))
            for warning in current.warnings
        )
        if current.backend in (
            GpuBackend.CUDA,
            GpuBackend.ROCM,
            GpuBackend.UNKNOWN,
        ) or tool_response_failed:
            reason = (
                "GPU backend detection was inconclusive"
                if current.backend == GpuBackend.UNKNOWN and not tool_response_failed
                else "GPU inventory tooling was unavailable or returned an invalid response"
            )
            return GpuHealth(
                healthy=False,
                status=GpuHealthStatus.TOOLING_UNAVAILABLE,
                reason=reason,
                warnings=current.warnings,
            )
        return GpuHealth(
            healthy=False,
            status=GpuHealthStatus.NO_GPU,
            reason="No GPU was detected",
            warnings=current.warnings,
        )

    try:
        report = _legacy_health_check()
    except Exception as exc:  # health is intentionally a non-throwing boundary
        return GpuHealth(
            healthy=False,
            status=GpuHealthStatus.UNHEALTHY,
            reason="GPU health checks could not complete",
            warnings=current.warnings,
            errors=(f"health check error: {type(exc).__name__}",),
            devices=current.devices,
        )

    failed = tuple(check.message for check in report.checks if check.status == "fail")
    warned = tuple(check.message for check in report.checks if check.status == "warn")
    if report.status == "fail":
        return GpuHealth(
            healthy=False,
            status=GpuHealthStatus.UNHEALTHY,
            reason="One or more GPU health checks failed",
            warnings=current.warnings + warned,
            errors=failed,
            devices=current.devices,
        )
    return GpuHealth(
        healthy=True,
        status=GpuHealthStatus.HEALTHY,
        reason="GPU inventory and health checks completed",
        warnings=current.warnings + warned,
        devices=current.devices,
    )


def _device_order(device: GpuDevice) -> tuple[Any, ...]:
    return (
        device.backend.value,
        device.index if device.index is not None else 2**31,
        device.uuid or "",
        device.name or "",
    )


def _rank(device: GpuDevice, preferred_backend: GpuBackend | None) -> tuple[Any, ...]:
    free = device.memory.free_mib if device.memory.free_mib is not None else -1
    total = device.memory.total_mib if device.memory.total_mib is not None else -1
    utilization = device.utilization_percent
    preferred = preferred_backend is None or device.backend == preferred_backend
    return (
        0 if preferred else 1,
        -free,
        -total,
        utilization if utilization is not None else 101,
        *_device_order(device),
    )


def select_gpu(
    policy: GpuSelectionPolicy | None = None,
    *,
    inventory: GpuInventory | None = None,
    health: GpuHealth | None = None,
) -> GpuSelectionResult:
    """Select a suitable GPU deterministically under an explicit policy."""

    selected_policy = policy or GpuSelectionPolicy()
    current = inventory if inventory is not None else discover_gpus()
    candidates = sorted(current.devices, key=_device_order)
    candidate_tuple = tuple(candidates)
    if not candidates:
        return GpuSelectionResult(
            selected=None,
            suitable=False,
            reason="No GPU candidates were discovered",
            candidates=candidate_tuple,
        )

    if selected_policy.require_healthy:
        health_result = health if health is not None else check_gpu_health(inventory=current)
        if not health_result.healthy:
            return GpuSelectionResult(
                selected=None,
                suitable=False,
                reason=f"Healthy GPU required: {health_result.reason}",
                candidates=candidate_tuple,
            )

    rejection_reasons: list[str] = []
    suitable: list[GpuDevice] = []
    for device in candidates:
        reasons: list[str] = []
        if selected_policy.minimum_total_mib is not None:
            if device.memory.total_mib is None:
                reasons.append("total VRAM is unavailable")
            elif device.memory.total_mib < selected_policy.minimum_total_mib:
                reasons.append("total VRAM is below the minimum")
        if selected_policy.minimum_free_mib is not None:
            if device.memory.free_mib is None:
                reasons.append("free VRAM is unavailable")
            elif device.memory.free_mib < selected_policy.minimum_free_mib:
                reasons.append("free VRAM is below the minimum")
        if selected_policy.maximum_utilization_percent is not None:
            if device.utilization_percent is None:
                reasons.append("utilization is unavailable")
            elif device.utilization_percent > selected_policy.maximum_utilization_percent:
                reasons.append("utilization exceeds the maximum")
        if reasons:
            rejection_reasons.extend(reasons)
        else:
            suitable.append(device)

    if not suitable:
        reason = (
            "; ".join(dict.fromkeys(rejection_reasons))
            if rejection_reasons
            else "No suitable GPU found"
        )
        return GpuSelectionResult(
            selected=None,
            suitable=False,
            reason=reason,
            candidates=candidate_tuple,
        )
    winner = sorted(
        suitable,
        key=lambda device: _rank(device, selected_policy.preferred_backend),
    )[0]
    used_fallback = (
        selected_policy.preferred_backend is not None
        and winner.backend != selected_policy.preferred_backend
    )
    reason_prefix = (
        "Preferred backend unavailable; selected fallback" if used_fallback else "Selected"
    )
    winner_label = winner.name or winner.uuid or winner.index
    return GpuSelectionResult(
        selected=winner,
        suitable=True,
        reason=f"{reason_prefix} deterministic best match: {winner_label}",
        candidates=candidate_tuple,
    )


def _torch_package_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def resolve_runtime(*, inventory: GpuInventory | None = None) -> RuntimeCapabilities:
    """Resolve hardware, tooling, and optional torch runtime capabilities."""

    current = inventory if inventory is not None else discover_gpus()
    warnings = list(current.warnings)
    installed = _torch_package_present()
    torch_mod: Any = None
    try:
        torch_mod = importlib.import_module("torch")
        installed = True
    except Exception as exc:  # broken binary wheels commonly raise OSError
        warnings.append(f"torch import failed: {type(exc).__name__}")

    torch_version = None
    cuda_version = None
    rocm_version = None
    cuda_usable = False
    rocm_usable = False
    mps_usable = False
    torch_usable = False
    if torch_mod is not None:
        try:
            torch_version = str(torch_mod.__version__)
            version_info = getattr(torch_mod, "version", None)
            cuda_version = getattr(version_info, "cuda", None)
            rocm_version = getattr(version_info, "hip", None)
            cuda_api_usable = bool(torch_mod.cuda.is_available())
            rocm_usable = cuda_api_usable and bool(rocm_version)
            cuda_usable = cuda_api_usable and not rocm_usable
            mps_api = getattr(getattr(torch_mod, "backends", None), "mps", None)
            mps_usable = bool(mps_api and mps_api.is_available())
            torch_usable = True
        except Exception as exc:
            warnings.append(f"torch runtime probe failed: {type(exc).__name__}")

    driver = next(
        (device.driver_version for device in current.devices if device.driver_version),
        None,
    )
    return RuntimeCapabilities(
        backend=current.backend,
        hardware_detected=current.backend in (
            GpuBackend.CUDA,
            GpuBackend.ROCM,
            GpuBackend.MPS,
        ),
        tooling_detected=current.source is not None,
        torch_installed=installed,
        torch_usable=torch_usable,
        cuda_usable=cuda_usable,
        rocm_usable=rocm_usable,
        mps_usable=mps_usable,
        driver_version=driver,
        torch_version=torch_version,
        cuda_version=str(cuda_version) if cuda_version is not None else None,
        rocm_version=str(rocm_version) if rocm_version is not None else None,
        warnings=tuple(warnings),
    )


__all__ = [
    "check_gpu_health",
    "detect_backend",
    "discover_gpus",
    "resolve_runtime",
    "select_gpu",
]
