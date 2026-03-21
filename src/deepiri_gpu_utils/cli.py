from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from .build_args import build_args_from_detection
from .detect import detect
from .doctor import doctor
from .ollama import recommend_models
from .setup import DeviceArg, setup_device, setup_device_mac


def _to_jsonable(obj: Any) -> Any:
    """Convert nested dataclasses/objects into JSON-serializable primitives."""

    # Handle dataclasses (including nested ones) explicitly.
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # Fall back to object dicts for non-dataclass objects.
    if hasattr(obj, "__dict__"):
        return {str(k): _to_jsonable(v) for k, v in obj.__dict__.items()}

    return obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepiri-gpu", description="deepiri-gpu-utils CLI")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_detect = subparsers.add_parser("detect", help="Detect best available backend")
    p_detect.add_argument(
        "--prefer",
        default=None,
        help="Optional backend hint (cuda/rocm/mps/cpu)",
    )
    p_detect.add_argument("--json", action="store_true", help="Emit JSON")

    p_doctor = subparsers.add_parser("doctor", help="Run readiness checks")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON")

    p_setup = subparsers.add_parser("setup", help="Set up prerequisites for a device")
    p_setup.add_argument(
        "--device",
        default="auto",
        choices=["auto", "nvidia", "amd", "apple", "cpu"],
        help="Target device profile",
    )
    p_setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute privileged commands",
    )
    p_setup.add_argument("--yes", action="store_true", help="Skip confirmations (where applicable)")

    p_build_args = subparsers.add_parser("build-args", help="Emit docker build args for detection")
    p_build_args.add_argument(
        "--device-type",
        default="auto",
        choices=["auto", "gpu", "cpu", "mpsos"],
        help="Override detection (default: auto from detect)",
    )
    p_build_args.add_argument("--json", action="store_true", help="Emit JSON")

    p_validate = subparsers.add_parser("validate", help="Aggregate validate checks (skeleton)")
    p_validate.add_argument("--json", action="store_true", help="Emit JSON")

    p_ollama = subparsers.add_parser("ollama", help="Ollama related helpers")
    ollama_sub = p_ollama.add_subparsers(dest="ollama_cmd", required=True)
    p_rec = ollama_sub.add_parser("recommend", help="Recommend Ollama model(s)")
    p_rec.add_argument("--backend-hint", default=None, help="Optional backend hint")
    p_rec.add_argument("--json", action="store_true", help="Emit JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "detect":
        result = detect(prefer=args.prefer)
        if args.json:
            print(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
        else:
            print(f"Detected backend: {result.backend}")
            if result.warnings:
                for w in result.warnings:
                    print(f"Warning: {w}")
        return 0

    if args.cmd == "doctor":
        report = doctor()
        if args.json:
            print(json.dumps(_to_jsonable(report), indent=2, sort_keys=True))
        else:
            print(f"Doctor status: {report.status}")
            if report.runbook:
                print("\nRunbook:")
                for line in report.runbook:
                    print(f"- {line}")
        return 0

    if args.cmd == "setup":
        # In the skeleton, we just return a runbook.
        device_arg: DeviceArg = args.device  # type: ignore[assignment]
        # This bootstrap branch never executes privileged commands.
        dry_run = True

        if device_arg == "apple":
            plan = setup_device_mac(dry_run=dry_run)
        else:
            plan = setup_device(device=device_arg, dry_run=dry_run)

        if plan.runbook:
            print("\n".join(plan.runbook))
        return 0

    if args.cmd == "build-args":
        out = build_args_from_detection(device_type=args.device_type)
        if args.json:
            print(json.dumps(_to_jsonable(out), indent=2, sort_keys=True))
        else:
            for key, val in out.build_args.items():
                print(f"{key}={val}")
            if out.warnings:
                for w in out.warnings:
                    print(f"Warning: {w}")
        return 0

    if args.cmd == "validate":
        d = detect()
        rep = doctor()
        payload = {"detect": _to_jsonable(d), "doctor": _to_jsonable(rep)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Validation complete.")
        return 0

    if args.cmd == "ollama":
        rec = recommend_models(backend_hint=args.backend_hint)
        if args.json:
            print(json.dumps(_to_jsonable(rec), indent=2, sort_keys=True))
        else:
            print(f"Default model: {rec.default_model}")
            if rec.recommended_models:
                print("Recommended:", ", ".join(rec.recommended_models))
        return 0

    parser.error("Unknown command")
    return 2

