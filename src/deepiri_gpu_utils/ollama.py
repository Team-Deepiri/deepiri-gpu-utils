from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OllamaRecommendation:
    default_model: str
    recommended_models: list[str]
    notes: list[str]
    category: Literal["unknown"] = "unknown"


def recommend_models(*, backend_hint: str | None = None) -> OllamaRecommendation:
    """Hardware-aware Ollama model recommendation (stub)."""

    _ = backend_hint
    return OllamaRecommendation(
        default_model="mistral:7b",
        recommended_models=["mistral:7b"],
        notes=["ollama recommend stub; implement hardware filtering in later branches"],
    )

