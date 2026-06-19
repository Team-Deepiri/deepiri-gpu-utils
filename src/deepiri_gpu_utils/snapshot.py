"""Hardware snapshot capture and diff for CI regression tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .summary import hardware_summary


@dataclass(frozen=True)
class SnapshotDiff:
    """Field-level differences between two snapshots."""

    changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    added: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def capture_snapshot() -> dict[str, Any]:
    """Capture the current hardware summary as a JSON-serializable dict."""

    snap = hardware_summary()
    return {
        "schema": "deepiri-gpu-snapshot/v1",
        "summary": {
            "detect": _jsonable(snap.detect),
            "doctor": _jsonable(snap.doctor),
            "inventory": _jsonable(snap.inventory),
            "build_args": _jsonable(snap.build_args),
            "ollama": _jsonable(snap.ollama),
            "torch_device": _jsonable(snap.torch_device),
            "system_ram_gb": snap.system_ram_gb,
            "gpu_count": snap.gpu_count,
            "total_vram_gb": snap.total_vram_gb,
            "notes": snap.notes,
        },
    }


def save_snapshot(path: str | Path) -> Path:
    """Write a snapshot JSON file; return the resolved path."""

    target = Path(path)
    target.write_text(json.dumps(capture_snapshot(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a snapshot JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _flatten_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", payload)
    flat: dict[str, Any] = {}
    if isinstance(summary, dict):
        for key, value in summary.items():
            if key in ("detect", "doctor", "inventory", "build_args", "ollama", "torch_device"):
                if isinstance(value, dict) and "backend" in value:
                    flat[f"{key}.backend"] = value.get("backend")
                if key == "detect" and isinstance(value, dict):
                    flat["detect.confidence"] = value.get("confidence")
                if key == "doctor" and isinstance(value, dict):
                    flat["doctor.status"] = value.get("status")
                if key == "ollama" and isinstance(value, dict):
                    flat["ollama.setup_tier"] = value.get("setup_tier")
                    flat["ollama.default_model"] = value.get("default_model")
                if key == "torch_device" and isinstance(value, dict):
                    flat["torch_device.device"] = value.get("device")
            else:
                flat[key] = value
    return flat


def diff_snapshots(left: dict[str, Any], right: dict[str, Any]) -> SnapshotDiff:
    """Diff two snapshot payloads on stable summary fields."""

    a = _flatten_summary(left)
    b = _flatten_summary(right)
    keys = set(a) | set(b)
    changed: dict[str, tuple[Any, Any]] = {}
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    for key in sorted(keys):
        if key not in a:
            added[key] = b[key]
        elif key not in b:
            removed[key] = a[key]
        elif a[key] != b[key]:
            changed[key] = (a[key], b[key])
    return SnapshotDiff(changed=changed, added=added, removed=removed)


def render_diff_text(diff: SnapshotDiff) -> str:
    """Human-readable snapshot diff."""

    lines = ["snapshot diff:"]
    if not diff.changed and not diff.added and not diff.removed:
        lines.append("  (no differences)")
        return "\n".join(lines)
    for key, (old, new) in diff.changed.items():
        lines.append(f"  ~ {key}: {old!r} -> {new!r}")
    for key, value in diff.added.items():
        lines.append(f"  + {key}: {value!r}")
    for key, value in diff.removed.items():
        lines.append(f"  - {key}: {value!r}")
    return "\n".join(lines)
