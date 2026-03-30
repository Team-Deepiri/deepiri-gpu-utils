from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .detect import detect

if TYPE_CHECKING:
    pass

DevicePolicy = Literal["auto", "cuda", "mps", "cpu", "rocm"]


@dataclass(frozen=True)
class DeviceDecision:
    device: str
    notes: list[str] = field(default_factory=list)
    torch_available: bool = False


def resolve_torch_device(policy: DevicePolicy = "auto") -> DeviceDecision:
    """Pick a torch device string; uses ``torch`` when the optional extra is installed."""

    notes: list[str] = []

    if policy == "cpu":
        try:
            import torch as torch_mod
        except ImportError:
            return DeviceDecision(
                device="cpu",
                notes=["policy=cpu; torch not installed"],
                torch_available=False,
            )
        _ = torch_mod.__version__
        return DeviceDecision(device="cpu", notes=["policy=cpu"], torch_available=True)

    try:
        import torch
    except ImportError:
        d = detect()
        guess = "cpu"
        if policy in ("cuda", "rocm") and d.backend in ("cuda", "rocm"):
            guess = "cuda"
        elif policy == "mps" and d.backend == "mps":
            guess = "mps"
        elif policy == "auto":
            if d.backend == "cuda":
                guess = "cuda"
            elif d.backend == "mps":
                guess = "mps"
            elif d.backend == "rocm":
                guess = "cuda"
        notes.append(
            "torch not installed; heuristic from detect() and policy. "
            "Install: pip install 'deepiri-gpu-utils[torch]'"
        )
        return DeviceDecision(device=guess, notes=notes, torch_available=False)

    d = detect()
    notes.append(f"detect backend={d.backend}")

    if policy == "cpu":
        return DeviceDecision(device="cpu", notes=notes + ["policy=cpu"], torch_available=True)

    if policy == "cuda":
        if torch.cuda.is_available():
            return DeviceDecision(
                device="cuda",
                notes=notes + ["policy=cuda, torch.cuda.is_available()=True"],
                torch_available=True,
            )
        return DeviceDecision(
            device="cpu",
            notes=notes + ["policy=cuda but CUDA not available in torch; falling back to cpu"],
            torch_available=True,
        )

    if policy == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceDecision(device="mps", notes=notes + ["policy=mps"], torch_available=True)
        return DeviceDecision(
            device="cpu",
            notes=notes + ["policy=mps but MPS not available; cpu"],
            torch_available=True,
        )

    if policy == "rocm":
        if torch.cuda.is_available():
            return DeviceDecision(
                device="cuda",
                notes=notes + ["policy=rocm mapped to torch cuda device"],
                torch_available=True,
            )
        return DeviceDecision(
            device="cpu",
            notes=notes + ["policy=rocm but no torch CUDA; cpu"],
            torch_available=True,
        )

    # auto
    mps_mod = getattr(torch.backends, "mps", None)
    if d.backend == "mps" and mps_mod and torch.backends.mps.is_available():
        return DeviceDecision(device="mps", notes=notes + ["auto: MPS"], torch_available=True)
    if d.backend in ("cuda", "rocm") and torch.cuda.is_available():
        return DeviceDecision(
            device="cuda",
            notes=notes + ["auto: CUDA/ROCm via torch.cuda"],
            torch_available=True,
        )
    return DeviceDecision(device="cpu", notes=notes + ["auto: cpu fallback"], torch_available=True)
