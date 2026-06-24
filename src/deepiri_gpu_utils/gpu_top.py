"""Read-only GPU process listing via nvidia-smi."""

from __future__ import annotations

import csv
import io
import shutil
from dataclasses import dataclass, field

from ._subprocess import run_text


@dataclass(frozen=True)
class GPUProcess:
    """One GPU compute process reported by nvidia-smi."""

    gpu_index: int | None
    pid: int | None
    process_name: str | None
    used_memory_mib: int | None


@dataclass(frozen=True)
class GPUTopResult:
    """GPU process table plus warnings."""

    processes: list[GPUProcess] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str | None = None


def _safe_int(value: str) -> int | None:
    v = value.strip()
    if not v or v.startswith("["):
        return None
    try:
        return int(v)
    except ValueError:
        return None


def gpu_top() -> GPUTopResult:
    """List GPU compute processes (NVIDIA only; empty on other backends)."""

    if not shutil.which("nvidia-smi"):
        return GPUTopResult(
            warnings=["nvidia-smi not available; no GPU process listing."],
            source=None,
        )

    res = run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,gpu_bus_id,gpu_serial,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    if not res.ok or not res.stdout.strip():
        # Fallback query without uuid noise
        res = run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_index,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=15,
        )
    if not res.ok:
        return GPUTopResult(warnings=["nvidia-smi process query failed."], source="nvidia-smi")

    processes: list[GPUProcess] = []
    for raw_line in res.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        reader = csv.reader(io.StringIO(line))
        try:
            row = next(reader)
        except StopIteration:
            continue
        cells = [c.strip() for c in row]
        if len(cells) >= 4 and cells[0].isdigit():
            processes.append(
                GPUProcess(
                    gpu_index=_safe_int(cells[0]),
                    pid=_safe_int(cells[1]),
                    process_name=cells[2] or None,
                    used_memory_mib=_safe_int(cells[3]),
                )
            )
        elif len(cells) >= 2:
            processes.append(
                GPUProcess(
                    gpu_index=None,
                    pid=_safe_int(cells[-3]) if len(cells) >= 3 else _safe_int(cells[0]),
                    process_name=cells[-2] if len(cells) >= 2 else None,
                    used_memory_mib=_safe_int(cells[-1]),
                )
            )

    return GPUTopResult(processes=processes, source="nvidia-smi")
