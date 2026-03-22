from __future__ import annotations

import csv
import io
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Literal

Backend = Literal["cuda", "rocm", "mps", "cpu", "unknown"]

_MIN_GPU_MEMORY_GB = 4
_BLACKWELL_RE = re.compile(r"(RTX 5080|RTX 5090|Blackwell)", re.IGNORECASE)


@dataclass(frozen=True)
class DetectResult:
    backend: Backend
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _coerce_prefer(prefer: str | Backend | None) -> tuple[Backend | None, list[str]]:
    """Normalize CLI / API prefer hint; return warnings for invalid values."""

    if prefer is None:
        return None, []
    if isinstance(prefer, str):
        p = prefer.lower().strip()
        if not p:
            return None, []
        mapping: dict[str, Backend] = {
            "cuda": "cuda",
            "nvidia": "cuda",
            "rocm": "rocm",
            "amd": "rocm",
            "mps": "mps",
            "apple": "mps",
            "metal": "mps",
            "cpu": "cpu",
        }
        if p not in mapping:
            return None, [f"Unknown prefer value {prefer!r}; ignoring."]
        return mapping[p], []
    return prefer, []


def _prefer_mismatch_warnings(want: Backend | None, got: Backend) -> list[str]:
    if want is None or want == got:
        return []
    return [f"Preferred backend was {want!r} but detected {got!r}."]


def _parse_nvidia_csv_line(line: str) -> dict[str, Any] | None:
    reader = csv.reader(io.StringIO(line.strip()))
    try:
        row = next(reader)
    except StopIteration:
        return None
    if len(row) < 3:
        return None
    driver_version = row[0].strip()
    try:
        memory_mib = int(float(row[1].strip()))
    except ValueError:
        return None
    gpu_name = ",".join(x.strip() for x in row[2:]).strip() or "unknown"
    memory_gb = max(memory_mib / 1024.0, 0.0)
    return {
        "driver_version": driver_version,
        "memory_mib": memory_mib,
        "memory_gb": round(memory_gb, 2),
        "name": gpu_name,
        "meets_min_vram": memory_gb >= _MIN_GPU_MEMORY_GB,
        "blackwell_family": bool(_BLACKWELL_RE.search(gpu_name)),
    }


def query_nvidia_smi() -> dict[str, Any] | None:
    """Run `nvidia-smi` and parse the first GPU row, or return None."""

    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return None
    parsed = _parse_nvidia_csv_line(lines[0])
    return parsed


def query_rocm_smi() -> dict[str, Any] | None:
    """Best-effort ROCm presence (no NVIDIA)."""

    if not shutil.which("rocm-smi"):
        return None
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    return {"raw": text.strip()[:2000]}


def detect(*, prefer: str | Backend | None = None) -> DetectResult:
    """Detect best available accelerator backend (NVIDIA, ROCm, MPS, or CPU).

    Parity targets: ``diri-cyrex/scripts/utils/detect_gpu.sh`` (NVIDIA + VRAM),
    ``team_dev_environments/ai-team/start.sh`` (cuda vs mps vs other).
    """

    pref, w_pref = _coerce_prefer(prefer)
    warnings = list(w_pref)

    nv = query_nvidia_smi()
    if nv is not None:
        warnings.extend(_prefer_mismatch_warnings(pref, "cuda"))
        details: dict[str, Any] = {
            "nvidia": nv,
            "min_gpu_memory_gb": _MIN_GPU_MEMORY_GB,
            "platform": platform.system(),
        }
        return DetectResult(
            backend="cuda",
            confidence=0.92,
            details=details,
            warnings=warnings,
        )

    rocm = query_rocm_smi()
    if rocm is not None:
        warnings.extend(_prefer_mismatch_warnings(pref, "rocm"))
        return DetectResult(
            backend="rocm",
            confidence=0.78,
            details={"rocm": rocm, "platform": platform.system()},
            warnings=warnings,
        )

    if platform.system() == "Darwin":
        warnings.extend(_prefer_mismatch_warnings(pref, "mps"))
        return DetectResult(
            backend="mps",
            confidence=0.88,
            details={"platform": "Darwin", "machine": platform.machine()},
            warnings=warnings,
        )

    warnings.extend(_prefer_mismatch_warnings(pref, "cpu"))
    return DetectResult(
        backend="cpu",
        confidence=0.82,
        details={"platform": platform.system(), "machine": platform.machine()},
        warnings=warnings,
    )
