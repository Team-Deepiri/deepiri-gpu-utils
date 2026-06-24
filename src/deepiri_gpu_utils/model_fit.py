"""Check whether a specific Ollama model fits the current hardware tier.

Read-only: reuses :mod:`ollama` tier logic and :func:`detect.detect` without
pulling models or touching Docker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .detect import DetectResult, detect
from .ollama import ModelFit, categorize_model, recommend_models, setup_tier
from .system_info import system_ram_gb

FitCategory = Literal["recommended", "usable", "marginal", "no"]


@dataclass(frozen=True)
class ModelFitResult:
    """Outcome of a single-model hardware fit check."""

    model: str
    fit: FitCategory
    setup_tier: str
    system_ram_gb: int
    effective_vram_gb: int
    default_model: str
    suitable: bool
    reason: str
    notes: list[str] = field(default_factory=list)


def _effective_vram_gb(d: DetectResult, ram_gb: int) -> int:
    if d.backend == "mps":
        return ram_gb
    if d.backend != "cuda":
        return 0
    nv = d.details.get("nvidia")
    if isinstance(nv, dict) and "memory_gb" in nv:
        return int(nv["memory_gb"])
    return 0


def model_fit_check(model_name: str, *, backend_hint: str | None = None) -> ModelFitResult:
    """Report whether ``model_name`` fits the detected hardware tier."""

    model = model_name.strip()
    d = detect()
    ram = system_ram_gb()
    vram = _effective_vram_gb(d, ram)

    hint = (backend_hint or "").lower().strip()
    notes: list[str] = []
    if hint in ("cpu", "cpu-only"):
        vram = 0
        notes.append("backend_hint forces CPU-tier VRAM=0 for model sizing.")
    elif hint in ("mps", "apple", "metal"):
        vram = ram
        notes.append("backend_hint uses unified memory estimate for Apple-style sizing.")

    tier = setup_tier(ram, vram)
    fit: ModelFit = categorize_model(model, ram, vram)
    rec = recommend_models(backend_hint=backend_hint)

    suitable = fit in ("recommended", "usable")
    if fit == "recommended":
        reason = f"{model!r} is recommended for setup_tier={tier}."
    elif fit == "usable":
        reason = f"{model!r} is usable on this host (setup_tier={tier})."
    elif fit == "marginal":
        reason = f"{model!r} is marginal; expect slow or unstable inference."
    else:
        reason = f"{model!r} is not suitable for this host (setup_tier={tier})."

    return ModelFitResult(
        model=model,
        fit=fit,
        setup_tier=tier,
        system_ram_gb=ram,
        effective_vram_gb=vram,
        default_model=rec.default_model,
        suitable=suitable,
        reason=reason,
        notes=notes,
    )
