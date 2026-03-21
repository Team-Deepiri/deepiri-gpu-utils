from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DevicePolicy = Literal["auto", "cuda", "mps", "cpu", "rocm"]


@dataclass(frozen=True)
class DeviceDecision:
    device: str
    notes: list[str] = field(default_factory=list)


def resolve_torch_device(policy: DevicePolicy = "auto") -> DeviceDecision:
    """Resolve best torch device (stub).

    Later branches will optionally use `torch` when installed.
    """

    _ = policy
    return DeviceDecision(device="cpu", notes=["resolve_torch_device stub"])

