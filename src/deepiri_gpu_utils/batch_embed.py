"""GPU-aware batch sizing for embedding and inference workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deepiri_gpu_utils.detect import detect
from deepiri_gpu_utils.hardware import effective_vram_gb
from deepiri_gpu_utils.torch_device import resolve_torch_device


@dataclass(frozen=True)
class BatchEmbedPolicy:
    """Recommended batch size and device for text embedding."""

    device: str
    batch_size: int
    max_seq_length: int
    notes: list[str]


def recommend_embed_batch(
    *,
    policy: Literal["auto", "cuda", "mps", "cpu"] = "auto",
    dim: int = 384,
    min_batch: int = 8,
    max_batch: int = 128,
    ram_gb: int = 16,
) -> BatchEmbedPolicy:
    """Derive batch size from GPU memory when available."""
    decision = resolve_torch_device(policy)
    notes = list(decision.notes)
    batch_size = min_batch
    max_seq = 512

    d = detect()
    vram_gb = effective_vram_gb(d, ram_gb=ram_gb)

    if decision.device.startswith("cuda") and vram_gb:
        # ~2MB per (batch * seq * dim) float32 rough heuristic for MiniLM-class
        usable = max(1.0, float(vram_gb) * 0.4)
        est = int(usable * 1024 / max(1, dim // 64))
        batch_size = max(min_batch, min(max_batch, est))
        notes.append(f"vram_gb={vram_gb} heuristic batch={batch_size}")
    elif decision.device == "mps":
        batch_size = max(min_batch, min(64, max_batch))
        notes.append("mps conservative batch")
    else:
        batch_size = min_batch
        notes.append("cpu min batch")

    return BatchEmbedPolicy(
        device=decision.device,
        batch_size=batch_size,
        max_seq_length=max_seq,
        notes=notes,
    )
