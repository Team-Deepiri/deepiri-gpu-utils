from __future__ import annotations

import io
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from deepiri_gpu_utils.cli import build_parser, main  # noqa: E402


class TestCliStub(unittest.TestCase):
    def test_parser_accepts_commands(self) -> None:
        parser = build_parser()

        parser.parse_args(["detect", "--json"])
        parser.parse_args(["doctor", "--json"])
        parser.parse_args(["setup", "--device", "auto", "--dry-run"])
        parser.parse_args(["build-args", "--json"])
        parser.parse_args(["build-args", "--device-type", "cpu", "--json"])
        parser.parse_args(["build-args", "--base-image-only"])
        parser.parse_args(["validate", "--json"])
        parser.parse_args(["torch-device", "--json"])
        parser.parse_args(["torch-device", "--policy", "cpu", "--json"])

        # nested subcommand
        parser.parse_args(["ollama", "recommend", "--json"])

    def test_main_detect_returns_zero(self) -> None:
        rc = main(["detect", "--json"])
        self.assertEqual(rc, 0)

    def test_main_build_args_base_image_only_one_line(self) -> None:
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            rc = main(["build-args", "--device-type", "cpu", "--base-image-only"])
        self.assertEqual(rc, 0)
        line = buf.getvalue().strip().splitlines()
        self.assertEqual(len(line), 1)
        self.assertIn("slim", line[0])

    def test_main_doctor_and_validate_json_do_not_crash(self) -> None:
        rc1 = main(["doctor", "--json"])
        self.assertEqual(rc1, 0)

        rc2 = main(["validate", "--json"])
        self.assertEqual(rc2, 0)

        rc3 = main(["torch-device", "--policy", "cpu", "--json"])
        self.assertEqual(rc3, 0)


if __name__ == "__main__":
    unittest.main()
