"""Hardware memory helpers shared by higher-level Deepiri libraries."""

from __future__ import annotations

from .detect import DetectResult


def effective_vram_gb(detect_result: DetectResult, ram_gb: int) -> int:
    """Estimate effective VRAM for model sizing on the active backend."""

    if detect_result.backend == "mps":
        return ram_gb
    if detect_result.backend != "cuda":
        return 0
    nvidia = detect_result.details.get("nvidia")
    if isinstance(nvidia, dict) and "memory_gb" in nvidia:
        return int(nvidia["memory_gb"])
    return 0


def apply_backend_hint_vram(
    vram_gb: int,
    ram_gb: int,
    backend_hint: str | None,
) -> tuple[int, list[str]]:
    """Adjust VRAM estimate for optional backend hints (cpu/mps)."""

    hint = (backend_hint or "").lower().strip()
    notes: list[str] = []
    if hint in ("cpu", "cpu-only"):
        notes.append("backend_hint forces CPU-tier VRAM=0 for model sizing.")
        return 0, notes
    if hint in ("mps", "apple", "metal"):
        notes.append("backend_hint uses unified memory estimate for Apple-style sizing.")
        return ram_gb, notes
    return vram_gb, notes
