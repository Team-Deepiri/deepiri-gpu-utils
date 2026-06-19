"""Regression tests for read-only host probes in ``system_info``.

All filesystem reads (``/proc/version``, ``/proc/meminfo``) and subprocess
calls (``sysctl``, ``lspci``, ``dmidecode``) are mocked so behavior is
deterministic and never depends on the host being WSL, Linux, macOS, or root.
"""

from __future__ import annotations

import subprocess
from unittest.mock import mock_open

from conftest import FakeProc, run_router, which_map

import deepiri_gpu_utils.system_info as si


def _raise_oserror(*args, **kwargs):
    raise OSError("mocked")


# --- is_wsl ------------------------------------------------------------------


def test_is_wsl_true_from_release(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "release", lambda: "5.10-microsoft-standard-WSL2")
    assert si.is_wsl() is True


def test_is_wsl_true_from_proc_version(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "release", lambda: "5.15.0-generic")
    monkeypatch.setattr("builtins.open", mock_open(read_data="Linux version ... Microsoft ..."))
    assert si.is_wsl() is True


def test_is_wsl_false(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "release", lambda: "5.15.0-generic")
    monkeypatch.setattr("builtins.open", _raise_oserror)
    assert si.is_wsl() is False


# --- system_ram_gb -----------------------------------------------------------


def test_system_ram_gb_linux(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    meminfo = "MemTotal:       33554432 kB\nMemFree:          100 kB\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=meminfo))
    assert si.system_ram_gb() == 32


def test_system_ram_gb_linux_read_error_returns_zero(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr("builtins.open", _raise_oserror)
    assert si.system_ram_gb() == 0


def test_system_ram_gb_darwin(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        subprocess, "run", run_router({"sysctl": FakeProc(returncode=0, stdout="17179869184\n")})
    )
    assert si.system_ram_gb() == 16


def test_system_ram_gb_other_platform_returns_zero(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Windows")
    assert si.system_ram_gb() == 0


# --- lspci_nvidia_present ----------------------------------------------------


def test_lspci_non_linux_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Darwin")
    assert si.lspci_nvidia_present() is None


def test_lspci_missing_binary_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr(si.shutil, "which", which_map({}))
    assert si.lspci_nvidia_present() is None


def test_lspci_detects_nvidia(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr(si.shutil, "which", which_map({"lspci": "/usr/bin/lspci"}))
    line = "01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [RTX 3090]\n"
    monkeypatch.setattr(subprocess, "run", run_router({"lspci": FakeProc(0, line)}))
    assert si.lspci_nvidia_present() is True


def test_lspci_no_nvidia_returns_false(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr(si.shutil, "which", which_map({"lspci": "/usr/bin/lspci"}))
    line = "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics\n"
    monkeypatch.setattr(subprocess, "run", run_router({"lspci": FakeProc(0, line)}))
    assert si.lspci_nvidia_present() is False


def test_lspci_timeout_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr(si.shutil, "which", which_map({"lspci": "/usr/bin/lspci"}))
    monkeypatch.setattr(
        subprocess,
        "run",
        run_router({"lspci": subprocess.TimeoutExpired(cmd="lspci", timeout=10)}),
    )
    assert si.lspci_nvidia_present() is None


# --- docker_cli_available ----------------------------------------------------


def test_docker_cli_available_true(monkeypatch) -> None:
    monkeypatch.setattr(si.shutil, "which", which_map({"docker": "/usr/bin/docker"}))
    assert si.docker_cli_available() is True


def test_docker_cli_available_false(monkeypatch) -> None:
    monkeypatch.setattr(si.shutil, "which", which_map({}))
    assert si.docker_cli_available() is False


# --- nvidia_container_toolkit_hint -------------------------------------------


def test_toolkit_hint_present_on_path(monkeypatch) -> None:
    monkeypatch.setattr(si.shutil, "which", which_map({"nvidia-ctk": "/usr/bin/nvidia-ctk"}))
    monkeypatch.setattr(si.os.path, "isfile", lambda p: False)
    hint = si.nvidia_container_toolkit_hint()
    assert hint["nvidia_ctk_on_path"] is True
    assert "nvidia_container_binary" not in hint


def test_toolkit_hint_binary_on_disk(monkeypatch) -> None:
    monkeypatch.setattr(si.shutil, "which", which_map({}))
    monkeypatch.setattr(
        si.os.path, "isfile", lambda p: p == "/usr/bin/nvidia-container-runtime"
    )
    hint = si.nvidia_container_toolkit_hint()
    assert hint["nvidia_ctk_on_path"] is False
    assert hint["nvidia_container_binary"] == "/usr/bin/nvidia-container-runtime"


# --- dmidecode_inventory -----------------------------------------------------


def test_dmidecode_not_linux(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Darwin")
    out = si.dmidecode_inventory()
    assert out["available"] is False


def test_dmidecode_requires_root(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr(si.shutil, "which", which_map({"dmidecode": "/usr/sbin/dmidecode"}))
    monkeypatch.setattr(si.os, "geteuid", lambda: 1000)
    out = si.dmidecode_inventory()
    assert out["available"] is False
    assert "root" in out["reason"]


def test_dmidecode_success_as_root(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr(si.shutil, "which", which_map({"dmidecode": "/usr/sbin/dmidecode"}))
    monkeypatch.setattr(si.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        subprocess, "run", run_router({"dmidecode": FakeProc(returncode=0, stdout="ACME\n")})
    )
    out = si.dmidecode_inventory()
    assert out["available"] is True
    assert out["system_manufacturer"] == "ACME"


def test_dmidecode_no_data_as_root(monkeypatch) -> None:
    monkeypatch.setattr(si.platform, "system", lambda: "Linux")
    monkeypatch.setattr(si.shutil, "which", which_map({"dmidecode": "/usr/sbin/dmidecode"}))
    monkeypatch.setattr(si.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        subprocess, "run", run_router({"dmidecode": FakeProc(returncode=1, stdout="")})
    )
    out = si.dmidecode_inventory()
    assert out["available"] is False
