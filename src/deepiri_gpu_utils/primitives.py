"""Typed, immutable GPU primitives with explicit JSON-friendly serialization.

Memory is represented in binary mebibytes (MiB).  GiB properties divide MiB
by 1024; they do not use decimal MB/GB units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class GpuBackend(StrEnum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"
    UNKNOWN = "unknown"


class GpuHealthStatus(StrEnum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    NO_GPU = "no_gpu"
    TOOLING_UNAVAILABLE = "tooling_unavailable"


def _freeze(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata floats must be finite")
        return value
    raise TypeError(f"metadata value is not JSON-compatible: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_thaw(item) for item in value]
    return value


def _backend(value: GpuBackend | str) -> GpuBackend:
    try:
        return value if isinstance(value, GpuBackend) else GpuBackend(value)
    except ValueError:
        return GpuBackend.UNKNOWN


def _validate_mib(name: str, value: int | None) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer MiB value or None")


@dataclass(frozen=True)
class GpuMemory:
    """GPU memory in MiB (2**20 bytes), with safe binary-GiB helpers."""

    total_mib: int | None = None
    free_mib: int | None = None
    used_mib: int | None = None

    def __post_init__(self) -> None:
        for name in ("total_mib", "free_mib", "used_mib"):
            _validate_mib(name, getattr(self, name))
        if self.total_mib is not None:
            if self.free_mib is not None and self.free_mib > self.total_mib:
                raise ValueError("free_mib cannot exceed total_mib")
            if self.used_mib is not None and self.used_mib > self.total_mib:
                raise ValueError("used_mib cannot exceed total_mib")

    @staticmethod
    def mib_to_gib(value: int | None) -> float | None:
        return None if value is None else value / 1024.0

    @staticmethod
    def gib_to_mib(value: float | int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("GiB value must be numeric or None")
        if not math.isfinite(value) or value < 0:
            raise ValueError("GiB value must be finite and non-negative")
        return round(float(value) * 1024)

    @property
    def total_gib(self) -> float | None:
        return self.mib_to_gib(self.total_mib)

    @property
    def free_gib(self) -> float | None:
        return self.mib_to_gib(self.free_mib)

    @property
    def used_gib(self) -> float | None:
        return self.mib_to_gib(self.used_mib)

    @property
    def calculated_used_mib(self) -> int | None:
        if self.used_mib is not None:
            return self.used_mib
        if self.total_mib is None or self.free_mib is None:
            return None
        return max(self.total_mib - self.free_mib, 0)

    def to_dict(self) -> dict[str, int | None]:
        return {
            "total_mib": self.total_mib,
            "free_mib": self.free_mib,
            "used_mib": self.used_mib,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GpuMemory:
        return cls(
            total_mib=data.get("total_mib"),
            free_mib=data.get("free_mib"),
            used_mib=data.get("used_mib"),
        )


@dataclass(frozen=True)
class GpuDevice:
    backend: GpuBackend
    index: int | None = None
    name: str | None = None
    uuid: str | None = None
    memory: GpuMemory = field(default_factory=GpuMemory)
    utilization_percent: float | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
    driver_version: str | None = None
    compute_capability: str | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend(self.backend))
        if self.index is not None and (
            not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0
        ):
            raise ValueError("index must be a non-negative integer or None")
        if self.utilization_percent is not None and not 0 <= self.utilization_percent <= 100:
            raise ValueError("utilization_percent must be between 0 and 100")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "memory": self.memory.to_dict(),
            "utilization_percent": self.utilization_percent,
            "temperature_c": self.temperature_c,
            "power_watts": self.power_watts,
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
            "source": self.source,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GpuDevice:
        memory = data.get("memory")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        return cls(
            backend=_backend(data.get("backend", "unknown")),
            index=data.get("index"),
            name=data.get("name"),
            uuid=data.get("uuid"),
            memory=GpuMemory.from_dict(memory) if isinstance(memory, Mapping) else GpuMemory(),
            utilization_percent=data.get("utilization_percent"),
            temperature_c=data.get("temperature_c"),
            power_watts=data.get("power_watts"),
            driver_version=data.get("driver_version"),
            compute_capability=data.get("compute_capability"),
            source=str(data.get("source", "unknown")),
            metadata=metadata,
        )


@dataclass(frozen=True)
class GpuInventory:
    devices: tuple[GpuDevice, ...] = ()
    backend: GpuBackend = GpuBackend.CPU
    source: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        object.__setattr__(self, "backend", _backend(self.backend))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def count(self) -> int:
        return len(self.devices)

    @property
    def has_gpu(self) -> bool:
        return bool(self.devices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "backend": self.backend.value,
            "source": self.source,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GpuInventory:
        devices = data.get("devices", ())
        return cls(
            devices=tuple(GpuDevice.from_dict(item) for item in devices),
            backend=_backend(data.get("backend", "unknown")),
            source=data.get("source"),
            warnings=tuple(data.get("warnings", ())),
        )


@dataclass(frozen=True)
class GpuHealth:
    healthy: bool
    status: GpuHealthStatus
    reason: str
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    devices: tuple[GpuDevice, ...] = ()

    def __post_init__(self) -> None:
        status = (
            self.status
            if isinstance(self.status, GpuHealthStatus)
            else GpuHealthStatus(self.status)
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "devices", tuple(self.devices))
        if self.healthy != (status == GpuHealthStatus.HEALTHY):
            raise ValueError("healthy must agree with status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "status": self.status.value,
            "reason": self.reason,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "devices": [device.to_dict() for device in self.devices],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GpuHealth:
        return cls(
            healthy=bool(data.get("healthy", False)),
            status=GpuHealthStatus(data.get("status", "unhealthy")),
            reason=str(data.get("reason", "")),
            warnings=tuple(data.get("warnings", ())),
            errors=tuple(data.get("errors", ())),
            devices=tuple(GpuDevice.from_dict(item) for item in data.get("devices", ())),
        )


@dataclass(frozen=True)
class GpuSelectionPolicy:
    preferred_backend: GpuBackend | None = None
    minimum_total_mib: int | None = None
    minimum_free_mib: int | None = None
    maximum_utilization_percent: float | None = None
    require_healthy: bool = False

    def __post_init__(self) -> None:
        if self.preferred_backend is not None:
            object.__setattr__(self, "preferred_backend", _backend(self.preferred_backend))
        _validate_mib("minimum_total_mib", self.minimum_total_mib)
        _validate_mib("minimum_free_mib", self.minimum_free_mib)
        if self.maximum_utilization_percent is not None and not (
            0 <= self.maximum_utilization_percent <= 100
        ):
            raise ValueError("maximum_utilization_percent must be between 0 and 100")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_backend": (
                self.preferred_backend.value if self.preferred_backend is not None else None
            ),
            "minimum_total_mib": self.minimum_total_mib,
            "minimum_free_mib": self.minimum_free_mib,
            "maximum_utilization_percent": self.maximum_utilization_percent,
            "require_healthy": self.require_healthy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GpuSelectionPolicy:
        preferred = data.get("preferred_backend")
        return cls(
            preferred_backend=_backend(preferred) if preferred is not None else None,
            minimum_total_mib=data.get("minimum_total_mib"),
            minimum_free_mib=data.get("minimum_free_mib"),
            maximum_utilization_percent=data.get("maximum_utilization_percent"),
            require_healthy=bool(data.get("require_healthy", False)),
        )


@dataclass(frozen=True)
class GpuSelectionResult:
    selected: GpuDevice | None
    suitable: bool
    reason: str
    candidates: tuple[GpuDevice, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.suitable != (self.selected is not None):
            raise ValueError("suitable must agree with whether a device was selected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "suitable": self.suitable,
            "reason": self.reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GpuSelectionResult:
        selected = data.get("selected")
        return cls(
            selected=GpuDevice.from_dict(selected) if isinstance(selected, Mapping) else None,
            suitable=bool(data.get("suitable", False)),
            reason=str(data.get("reason", "")),
            candidates=tuple(GpuDevice.from_dict(item) for item in data.get("candidates", ())),
        )


@dataclass(frozen=True)
class RuntimeCapabilities:
    backend: GpuBackend
    hardware_detected: bool
    tooling_detected: bool
    torch_installed: bool
    torch_usable: bool
    cuda_usable: bool
    rocm_usable: bool
    mps_usable: bool
    driver_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _backend(self.backend))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "hardware_detected": self.hardware_detected,
            "tooling_detected": self.tooling_detected,
            "torch_installed": self.torch_installed,
            "torch_usable": self.torch_usable,
            "cuda_usable": self.cuda_usable,
            "rocm_usable": self.rocm_usable,
            "mps_usable": self.mps_usable,
            "driver_version": self.driver_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "rocm_version": self.rocm_version,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeCapabilities:
        return cls(
            backend=_backend(data.get("backend", "unknown")),
            hardware_detected=bool(data.get("hardware_detected", False)),
            tooling_detected=bool(data.get("tooling_detected", False)),
            torch_installed=bool(data.get("torch_installed", False)),
            torch_usable=bool(data.get("torch_usable", False)),
            cuda_usable=bool(data.get("cuda_usable", False)),
            rocm_usable=bool(data.get("rocm_usable", False)),
            mps_usable=bool(data.get("mps_usable", False)),
            driver_version=data.get("driver_version"),
            torch_version=data.get("torch_version"),
            cuda_version=data.get("cuda_version"),
            rocm_version=data.get("rocm_version"),
            warnings=tuple(data.get("warnings", ())),
        )


__all__ = [
    "GpuBackend",
    "GpuDevice",
    "GpuHealth",
    "GpuHealthStatus",
    "GpuInventory",
    "GpuMemory",
    "GpuSelectionPolicy",
    "GpuSelectionResult",
    "RuntimeCapabilities",
]
