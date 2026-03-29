from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import deepiri_gpu_utils.detect as det  # noqa: E402


class TestDetect(unittest.TestCase):
    def test_prefer_invalid_emits_warning(self) -> None:
        r = det.detect(prefer="not-a-backend")
        self.assertIn("Unknown prefer", r.warnings[0])

    @patch.object(det, "query_nvidia_smi", return_value=None)
    @patch.object(det.system_info, "lspci_nvidia_present", return_value=False)
    @patch.object(det, "query_rocm_smi", return_value=None)
    @patch("deepiri_gpu_utils.detect.platform.system", return_value="Linux")
    @patch("deepiri_gpu_utils.detect.platform.machine", return_value="x86_64")
    def test_cpu_when_no_gpu(self, *_mocks: object) -> None:
        r = det.detect()
        self.assertEqual(r.backend, "cpu")
        self.assertGreaterEqual(r.confidence, 0.5)

    @patch.object(det, "query_nvidia_smi", return_value=None)
    @patch.object(det.system_info, "lspci_nvidia_present", return_value=True)
    @patch.object(det, "query_rocm_smi", return_value=None)
    @patch.object(det.system_info, "is_wsl", return_value=False)
    @patch("deepiri_gpu_utils.detect.platform.system", return_value="Linux")
    def test_cuda_lspci_when_smi_missing(self, *_m: object) -> None:
        r = det.detect()
        self.assertEqual(r.backend, "cuda")
        self.assertTrue(r.details.get("nvidia_drivers_missing"))

    @patch.object(det, "query_nvidia_smi")
    @patch.object(det, "query_rocm_smi", return_value=None)
    @patch("deepiri_gpu_utils.detect.platform.system", return_value="Linux")
    def test_cuda_when_nvidia_ok(self, _mock_system: object, _rocm: object, nv: object) -> None:
        nv.return_value = {
            "driver_version": "550.0",
            "memory_mib": 8192,
            "memory_gb": 8.0,
            "name": "NVIDIA GeForce RTX 3080",
            "meets_min_vram": True,
            "blackwell_family": False,
        }
        r = det.detect()
        self.assertEqual(r.backend, "cuda")
        self.assertEqual(r.details["nvidia"]["name"], "NVIDIA GeForce RTX 3080")

    @patch.object(det, "query_nvidia_smi", return_value=None)
    @patch.object(det, "query_rocm_smi", return_value=None)
    @patch("deepiri_gpu_utils.detect.platform.system", return_value="Darwin")
    @patch("deepiri_gpu_utils.detect.platform.machine", return_value="arm64")
    def test_mps_on_darwin(self, *_m: object) -> None:
        r = det.detect()
        self.assertEqual(r.backend, "mps")

    def test_parse_nvidia_line(self) -> None:
        out = det._parse_nvidia_csv_line("550.54, 8192, NVIDIA GeForce RTX 3080")
        assert out is not None
        self.assertEqual(out["memory_gb"], 8.0)
        self.assertFalse(out["blackwell_family"])

    def test_blackwell_flag(self) -> None:
        out = det._parse_nvidia_csv_line("560.0, 16384, NVIDIA GeForce RTX 5090")
        assert out is not None
        self.assertTrue(out["blackwell_family"])


if __name__ == "__main__":
    unittest.main()
