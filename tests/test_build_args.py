from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from deepiri_gpu_utils.build_args import (  # noqa: E402
    build_args_from_detection,
    infer_device_type_from_base_image,
)
from deepiri_gpu_utils.detect import DetectResult  # noqa: E402


class TestBuildArgs(unittest.TestCase):
    def test_infer_from_base_image(self) -> None:
        self.assertEqual(
            infer_device_type_from_base_image("pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"),
            "gpu",
        )
        self.assertEqual(infer_device_type_from_base_image("python:3.11-slim"), "cpu")
        self.assertEqual(
            infer_device_type_from_base_image("someorg/pytorch-mps-macos:dev"),
            "mpsos",
        )

    def test_explicit_gpu(self) -> None:
        b = build_args_from_detection(device_type="gpu")
        self.assertEqual(b.device_type, "gpu")
        self.assertIn("cuda", b.base_image)
        self.assertEqual(b.build_args["DEVICE_TYPE"], "gpu")

    def test_explicit_mpsos(self) -> None:
        b = build_args_from_detection(device_type="mpsos")
        self.assertEqual(b.device_type, "mpsos")
        self.assertEqual(b.build_args["BASE_IMAGE"], "python:3.11-slim")

    def test_auto_cuda_high_vram(self) -> None:
        dr = DetectResult(
            backend="cuda",
            details={
                "nvidia": {
                    "meets_min_vram": True,
                    "name": "RTX 4090",
                    "memory_gb": 24.0,
                }
            },
        )
        b = build_args_from_detection(device_type="auto", detect_result=dr)
        self.assertEqual(b.device_type, "gpu")
        self.assertIn("pytorch", b.base_image)

    def test_auto_cuda_low_vram(self) -> None:
        dr = DetectResult(
            backend="cuda",
            details={"nvidia": {"meets_min_vram": False, "memory_gb": 2.0}},
        )
        b = build_args_from_detection(device_type="auto", detect_result=dr)
        self.assertEqual(b.device_type, "cpu")
        self.assertTrue(any("VRAM below" in w for w in b.warnings))

    def test_auto_mps_sets_mpsos(self) -> None:
        dr = DetectResult(backend="mps", details={})
        b = build_args_from_detection(device_type="auto", detect_result=dr)
        self.assertEqual(b.device_type, "mpsos")
        self.assertEqual(b.build_args["DEVICE_TYPE"], "mpsos")


if __name__ == "__main__":
    unittest.main()
