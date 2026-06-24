"""Shell-friendly export lines for Docker / compose build workflows.

Read-only: emits ``export KEY=value`` lines suitable for ``eval`` in bash.
Does not execute commands or mutate the current process environment.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from .build_args import BuildArgs, build_args_from_detection
from .detect import DetectResult


@dataclass(frozen=True)
class ShellExport:
    """Shell ``export`` lines and the underlying build args."""

    lines: list[str] = field(default_factory=list)
    build_args: BuildArgs | None = None


def _quote(value: str) -> str:
    return shlex.quote(value)


def build_args_shell_export(
    *,
    device_type: str = "auto",
    detect_result: DetectResult | None = None,
    prefix: str = "",
) -> ShellExport:
    """Return ``export VAR=value`` lines for Cyrex-style docker build args.

    ``prefix`` is prepended to each variable name (e.g. ``CYREX_`` ->
    ``export CYREX_BASE_IMAGE=...``).
    """

    ba = build_args_from_detection(device_type=device_type, detect_result=detect_result)
    lines = [f"export {prefix}{key}={_quote(val)}" for key, val in ba.build_args.items()]
    return ShellExport(lines=lines, build_args=ba)
