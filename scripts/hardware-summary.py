#!/usr/bin/env python3
"""Aggregate hardware snapshot via the library API (read-only)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from deepiri_gpu_utils.summary import hardware_summary


def _jsonable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main() -> int:
    snap = hardware_summary()
    payload = {
        "backend": snap.detect.backend,
        "doctor_status": snap.doctor.status,
        "gpu_count": snap.gpu_count,
        "total_vram_gb": snap.total_vram_gb,
        "system_ram_gb": snap.system_ram_gb,
        "torch_device": snap.torch_device.device,
        "base_image": snap.build_args.base_image,
        "device_type": snap.build_args.device_type,
        "notes": snap.notes,
    }
    if "--full" in sys.argv:
        payload = _jsonable(snap)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
