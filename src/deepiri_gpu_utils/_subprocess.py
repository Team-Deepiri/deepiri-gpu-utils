"""Internal safe runner for external commands.

Centralizes the success / nonzero-exit / timeout / missing-executable handling
that detection and host-probe code share. This module is private: it is not
exported from the package ``__init__`` and is not part of the public API, so it
may change without notice.

It deliberately does not broaden behavior beyond what the callers already did:
processes are run with captured text output and never raise for the common
failure modes (timeout, OS error / missing executable). Callers keep their own
``shutil.which`` guards and result parsing.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RunResult:
    """Normalized outcome of an external command.

    ``ok`` is True only when the process actually executed and exited ``0``.
    Timeouts and OS errors (including a missing executable) are reported with
    ``ok=False`` instead of raising.
    """

    ok: bool
    returncode: int | None
    stdout: str
    stderr: str


def run_text(cmd: Sequence[str], *, timeout: float) -> RunResult:
    """Run ``cmd`` capturing text output; never raise for common failure modes.

    Mirrors the previous inline ``subprocess.run(..., capture_output=True,
    text=True, check=False)`` calls plus their ``except (OSError,
    TimeoutExpired)`` handling, returning a :class:`RunResult` instead.
    """

    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return RunResult(ok=False, returncode=None, stdout="", stderr="")

    return RunResult(
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
