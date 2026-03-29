from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from deepiri_gpu_utils.doctor import doctor  # noqa: E402
from deepiri_gpu_utils.setup import setup_device, setup_device_mac  # noqa: E402
from deepiri_gpu_utils.torch_device import resolve_torch_device  # noqa: E402


class TestDoctor(unittest.TestCase):
    @patch("deepiri_gpu_utils.doctor.detect")
    def test_doctor_has_findings(self, mock_det: object) -> None:
        from deepiri_gpu_utils.detect import DetectResult

        mock_det.return_value = DetectResult(backend="cpu", confidence=0.8, details={})
        rep = doctor()
        self.assertIn("platform", rep.findings)
        self.assertIn("system_ram_gb", rep.findings)


class TestSetup(unittest.TestCase):
    def test_setup_nvidia_runbook_nonempty(self) -> None:
        p = setup_device("nvidia", dry_run=True)
        self.assertGreater(len(p.runbook), 3)
        self.assertTrue(any("NVIDIA" in line for line in p.runbook))

    def test_setup_mac(self) -> None:
        p = setup_device_mac(dry_run=True)
        self.assertEqual(p.device, "apple")


class TestTorchDevice(unittest.TestCase):
    def test_resolve_cpu_policy(self) -> None:
        d = resolve_torch_device("cpu")
        self.assertEqual(d.device, "cpu")


if __name__ == "__main__":
    unittest.main()
