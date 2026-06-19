"""Shared test helpers for deepiri-gpu-utils baseline regression suite.

These helpers exist to lock down *current* behavior in a CPU-only / no-GPU
environment. They never require real GPU hardware: every external probe
(``nvidia-smi``, ``rocm-smi``, ``lspci``, ``dmidecode``, Docker, ``/proc`` reads,
and ``torch`` imports) is faked here so tests are deterministic on any host.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import types
from dataclasses import dataclass

import pytest

# Make ``src/`` importable even if the package is not installed editable.
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class FakeProc:
    """Minimal stand-in for :class:`subprocess.CompletedProcess`."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def which_map(mapping: dict[str, str]):
    """Return a fake ``shutil.which`` resolving command name -> path (or None)."""

    def _which(cmd, *args, **kwargs):
        return mapping.get(cmd)

    return _which


def run_router(routes: dict[str, object]):
    """Return a fake ``subprocess.run`` dispatching on the executable name.

    ``routes`` maps a command name (argv[0]) to either a :class:`FakeProc` to
    return, or an exception instance/class to raise (e.g. ``OSError`` or
    ``subprocess.TimeoutExpired``). Unknown commands return a non-zero result.
    """

    def _run(cmd, *args, **kwargs):
        name = cmd[0] if isinstance(cmd, (list, tuple)) else cmd
        resp = routes.get(name)
        if resp is None:
            return FakeProc(returncode=1, stdout="", stderr="")
        if isinstance(resp, BaseException):
            raise resp
        if isinstance(resp, type) and issubclass(resp, BaseException):
            raise resp(name)
        return resp

    return _run


def make_fake_torch(*, cuda: bool = False, mps: bool = False, version: str = "2.0.0+fake"):
    """Build a fake ``torch`` module object suitable for ``sys.modules`` injection."""

    mod = types.ModuleType("torch")
    mod.__version__ = version
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return mod


@contextlib.contextmanager
def torch_absent():
    """Force ``import torch`` to raise :class:`ImportError` within the block."""

    sentinel = object()
    saved = sys.modules.get("torch", sentinel)
    sys.modules["torch"] = None  # a None entry makes ``import torch`` raise ImportError
    try:
        yield
    finally:
        if saved is sentinel:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


@contextlib.contextmanager
def torch_present(*, cuda: bool = False, mps: bool = False):
    """Inject a fake ``torch`` module so ``import torch`` succeeds within the block."""

    fake = make_fake_torch(cuda=cuda, mps=mps)
    sentinel = object()
    saved = sys.modules.get("torch", sentinel)
    sys.modules["torch"] = fake
    try:
        yield fake
    finally:
        if saved is sentinel:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


@pytest.fixture
def force_cpu(monkeypatch):
    """Force :func:`detect` to resolve to CPU regardless of host hardware."""

    import deepiri_gpu_utils.detect as det

    monkeypatch.setattr(det, "query_nvidia_smi", lambda: None)
    monkeypatch.setattr(det, "query_rocm_smi", lambda: None)
    monkeypatch.setattr(det.system_info, "lspci_nvidia_present", lambda: None)
    monkeypatch.setattr(det.system_info, "is_wsl", lambda: False)
    monkeypatch.setattr(det.platform, "system", lambda: "Linux")
    monkeypatch.setattr(det.platform, "machine", lambda: "x86_64")
    return "cpu"
