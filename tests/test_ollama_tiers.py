from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from deepiri_gpu_utils.ollama import categorize_model, recommend_models, setup_tier  # noqa: E402


class TestOllamaTiers(unittest.TestCase):
    def test_setup_tier_premium(self) -> None:
        self.assertEqual(setup_tier(32, 16), "setup5")

    def test_setup_tier_minimal(self) -> None:
        self.assertEqual(setup_tier(8, 0), "minimal")

    def test_mistral_7b_high_end(self) -> None:
        self.assertEqual(categorize_model("mistral:7b", 32, 16), "recommended")

    def test_mistral_7b_low_ram(self) -> None:
        self.assertEqual(categorize_model("mistral:7b", 4, 0), "no")

    def test_70b_requires_vram(self) -> None:
        self.assertEqual(categorize_model("llama3.1:70b", 64, 16), "no")
        self.assertEqual(categorize_model("llama3.1:70b", 64, 48), "marginal")

    def test_recommend_models_runs(self) -> None:
        rec = recommend_models()
        self.assertTrue(rec.default_model)
        self.assertEqual(rec.category, "hardware_tiered")


if __name__ == "__main__":
    unittest.main()
