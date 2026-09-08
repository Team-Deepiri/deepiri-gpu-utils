"""deepiri-gpu-utils package.

This package centralizes GPU/device detection and setup helpers for the Deepiri
ecosystem.
"""

from ._version import __version__
from .canonical import (
    check_gpu_health,
    detect_backend,
    discover_gpus,
    resolve_runtime,
    select_gpu,
)
from .health import health_check
from .inventory import GPUInfo, GPUInventoryResult, GPUSelection, choose_suitable_gpu, gpu_inventory
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
from .torch_device import DeviceDecision, resolve_torch_device

__all__ = [
    "DeviceDecision",
    "GPUInfo",
    "GPUInventoryResult",
    "GPUSelection",
    "GpuBackend",
    "GpuDevice",
    "GpuHealth",
    "GpuHealthStatus",
    "GpuInventory",
    "GpuMemory",
    "GpuSelectionPolicy",
    "GpuSelectionResult",
    "RuntimeCapabilities",
    "__version__",
    "check_gpu_health",
    "choose_suitable_gpu",
    "detect_backend",
    "discover_gpus",
    "gpu_inventory",
    "health_check",
    "resolve_runtime",
    "resolve_torch_device",
    "select_gpu",
]
