from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from .build_args import build_args_from_detection
from .compose_gpu import compose_gpu_config
from .detect import detect
from .doctor import doctor
from .export_env import build_args_shell_export
from .install_check import install_readiness, install_readiness_all
from .inventory import choose_suitable_gpu, gpu_inventory
from .model_fit import model_fit_check
from .ollama import recommend_models
from .profiles import all_backend_profiles, backend_profile
from .setup import DeviceArg, setup_device, setup_device_mac
from .summary import hardware_summary
from .torch_device import resolve_torch_device


def _to_jsonable(obj: Any) -> Any:
    """Convert nested dataclasses/objects into JSON-serializable primitives."""

    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

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

    p_doctor = subparsers.add_parser("doctor", help="Run readiness checks (Docker, DMI hints, WSL)")
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON")

    p_setup = subparsers.add_parser("setup", help="Print setup runbook (does not run sudo)")
    p_setup.add_argument(
        "--device",
        default="auto",
        choices=["auto", "nvidia", "amd", "apple", "cpu"],
        help="Target device profile",
    )
    p_setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Mark plan as dry-run (default unless --yes)",
    )
    p_setup.add_argument(
        "--yes",
        action="store_true",
        help="Mark plan as confirmed (still prints runbook only; no privileged execution)",
    )

    p_build_args = subparsers.add_parser("build-args", help="Emit docker build args for detection")
    p_build_args.add_argument(
        "--device-type",
        default="auto",
        choices=["auto", "gpu", "cpu", "mpsos"],
        help="Override detection (default: auto from detect)",
    )
    p_build_args.add_argument("--json", action="store_true", help="Emit JSON")
    p_build_args.add_argument(
        "--base-image-only",
        action="store_true",
        help=(
            "Print only BASE_IMAGE (one line) for shells: "
            "BASE_IMAGE=$(deepiri-gpu build-args --base-image-only)"
        ),
    )

    p_validate = subparsers.add_parser(
        "validate",
        help="Aggregate detect, doctor, build-args, ollama, torch-device",
    )
    p_validate.add_argument("--json", action="store_true", help="Emit JSON")

    p_ollama = subparsers.add_parser("ollama", help="Ollama related helpers")
    ollama_sub = p_ollama.add_subparsers(dest="ollama_cmd", required=True)
    p_rec = ollama_sub.add_parser("recommend", help="Recommend Ollama model(s) by hardware tier")
    p_rec.add_argument("--backend-hint", default=None, help="Optional backend hint (cpu/mps/cuda)")
    p_rec.add_argument("--json", action="store_true", help="Emit JSON")

    p_torch = subparsers.add_parser(
        "torch-device",
        help="Resolve torch device (optional [torch] extra)",
    )
    p_torch.add_argument(
        "--policy",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu", "rocm"],
        help="Device selection policy",
    )
    p_torch.add_argument("--json", action="store_true", help="Emit JSON")

    p_inventory = subparsers.add_parser(
        "inventory",
        help="List detected GPUs (read-only); optional VRAM suitability check",
    )
    p_inventory.add_argument(
        "--min-memory-gb",
        type=float,
        default=None,
        help="If set, include a suitability selection for this minimum VRAM (GB)",
    )
    p_inventory.add_argument(
        "--backend",
        default=None,
        help="Optional backend filter for the suitability check (cuda/rocm/mps)",
    )
    p_inventory.add_argument("--json", action="store_true", help="Emit JSON")

    p_summary = subparsers.add_parser(
        "summary",
        help="Aggregate detect, doctor, inventory, build-args, ollama, torch-device",
    )
    p_summary.add_argument("--json", action="store_true", help="Emit JSON")

    p_export = subparsers.add_parser(
        "export-env",
        help="Print shell export lines for docker build args (read-only)",
    )
    p_export.add_argument(
        "--device-type",
        default="auto",
        choices=["auto", "gpu", "cpu", "mpsos"],
        help="Override detection (default: auto from detect)",
    )
    p_export.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for exported variable names (e.g. CYREX_)",
    )

    p_model_fit = subparsers.add_parser(
        "model-fit",
        help="Check whether a specific Ollama model fits this hardware",
    )
    p_model_fit.add_argument("model", help="Ollama model id (e.g. mistral:7b)")
    p_model_fit.add_argument("--backend-hint", default=None, help="Optional backend hint")
    p_model_fit.add_argument("--json", action="store_true", help="Emit JSON")

    p_install = subparsers.add_parser(
        "install-check",
        help="Check driver/tooling install readiness for NVIDIA, AMD, Apple, or CPU",
    )
    p_install.add_argument(
        "--device",
        default="auto",
        choices=["auto", "nvidia", "amd", "apple", "cpu"],
        help="Target device profile (default: auto-detect)",
    )
    p_install.add_argument(
        "--all",
        action="store_true",
        help="Report readiness for every backend profile (cuda, rocm, mps, cpu)",
    )
    p_install.add_argument("--json", action="store_true", help="Emit JSON")

    p_profile = subparsers.add_parser(
        "profile",
        help="Show canonical install/docker profile for a GPU backend",
    )
    p_profile.add_argument(
        "--backend",
        default=None,
        choices=["cuda", "rocm", "mps", "cpu"],
        help="Backend profile to show (default: detected backend)",
    )
    p_profile.add_argument(
        "--all",
        action="store_true",
        help="List all backend profiles",
    )
    p_profile.add_argument("--json", action="store_true", help="Emit JSON")

    p_compose = subparsers.add_parser(
        "compose-gpu",
        help="Emit Docker Compose GPU device/deploy hints for the active backend",
    )
    p_compose.add_argument(
        "--backend",
        default=None,
        choices=["cuda", "rocm", "mps", "cpu"],
        help="Override backend (default: detect)",
    )
    p_compose.add_argument("--json", action="store_true", help="Emit JSON")

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
        device_arg: DeviceArg = args.device  # type: ignore[assignment]
        dry_run = args.dry_run or not args.yes

        if device_arg == "apple":
            plan = setup_device_mac(dry_run=dry_run)
        else:
            plan = setup_device(device=device_arg, dry_run=dry_run)

        if plan.runbook:
            print("\n".join(plan.runbook))
        return 0

    if args.cmd == "build-args":
        out = build_args_from_detection(device_type=args.device_type)
        if args.base_image_only:
            print(out.base_image)
            return 0
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
        ba = build_args_from_detection(device_type="auto")
        ollama_rec = recommend_models()
        torch_dec = resolve_torch_device("auto")
        payload = {
            "detect": _to_jsonable(d),
            "doctor": _to_jsonable(rep),
            "build_args": _to_jsonable(ba),
            "ollama": _to_jsonable(ollama_rec),
            "torch_device": _to_jsonable(torch_dec),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"validate: detect={d.backend} doctor={rep.status} torch={torch_dec.device}")
        return 0

    if args.cmd == "ollama":
        rec = recommend_models(backend_hint=args.backend_hint)
        if args.json:
            print(json.dumps(_to_jsonable(rec), indent=2, sort_keys=True))
        else:
            print(f"Default model: {rec.default_model} (setup_tier={rec.setup_tier})")
            if rec.recommended_models:
                print("Recommended:", ", ".join(rec.recommended_models[:8]))
            if rec.usable_models:
                print("Usable:", ", ".join(rec.usable_models[:8]))
        return 0

    if args.cmd == "torch-device":
        td = resolve_torch_device(args.policy)  # type: ignore[arg-type]
        if args.json:
            print(json.dumps(_to_jsonable(td), indent=2, sort_keys=True))
        else:
            print(f"torch device: {td.device} (torch_installed={td.torch_available})")
            for n in td.notes:
                print(f"  note: {n}")
        return 0

    if args.cmd == "inventory":
        inv = gpu_inventory()
        selection = None
        if args.min_memory_gb is not None:
            selection = choose_suitable_gpu(args.min_memory_gb, backend=args.backend)
        if args.json:
            payload = _to_jsonable(inv)
            if selection is not None:
                payload["selection"] = _to_jsonable(selection)
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            if inv.gpus:
                for g in inv.gpus:
                    idx = g.index if g.index is not None else "-"
                    mem = f"{g.memory_gb}GB" if g.memory_gb is not None else "VRAM=?"
                    print(f"[{idx}] {g.backend} {g.name or 'unknown'} ({mem}) via {g.source}")
            else:
                print("No GPUs detected.")
            for w in inv.warnings:
                print(f"Warning: {w}")
            if selection is not None:
                print(
                    f"Suitable for {args.min_memory_gb}GB: "
                    f"{selection.suitable} - {selection.reason}"
                )
        return 0

    if args.cmd == "summary":
        snap = hardware_summary()
        if args.json:
            payload = {
                "detect": _to_jsonable(snap.detect),
                "doctor": _to_jsonable(snap.doctor),
                "inventory": _to_jsonable(snap.inventory),
                "build_args": _to_jsonable(snap.build_args),
                "ollama": _to_jsonable(snap.ollama),
                "torch_device": _to_jsonable(snap.torch_device),
                "system_ram_gb": snap.system_ram_gb,
                "gpu_count": snap.gpu_count,
                "total_vram_gb": snap.total_vram_gb,
                "notes": snap.notes,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"summary: backend={snap.detect.backend} doctor={snap.doctor.status} "
                f"gpus={snap.gpu_count} torch={snap.torch_device.device}"
            )
            for note in snap.notes:
                print(f"  note: {note}")
        return 0

    if args.cmd == "export-env":
        export = build_args_shell_export(device_type=args.device_type, prefix=args.prefix)
        print("\n".join(export.lines))
        return 0

    if args.cmd == "model-fit":
        result = model_fit_check(args.model, backend_hint=args.backend_hint)
        if args.json:
            print(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
        else:
            print(f"model-fit: {result.model} -> {result.fit} (suitable={result.suitable})")
            print(f"  {result.reason}")
            for note in result.notes:
                print(f"  note: {note}")
        return 0

    if args.cmd == "install-check":
        if args.all:
            results = install_readiness_all()
            if args.json:
                print(json.dumps(_to_jsonable(results), indent=2, sort_keys=True))
            else:
                for item in results:
                    status = "ready" if item.ready else "missing"
                    print(f"{item.backend}: {status} (device={item.device})")
                    if item.missing_required:
                        print(f"  missing: {', '.join(item.missing_required)}")
            return 0

        result = install_readiness(device=args.device)
        if args.json:
            print(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
        else:
            status = "ready" if result.ready else "not ready"
            print(f"install-check: {result.profile_label} -> {status}")
            if result.missing_required:
                print(f"  missing required: {', '.join(result.missing_required)}")
            if result.drivers_missing:
                print("  drivers/tooling missing for PCI-visible GPU")
            for step in result.install_steps[:3]:
                print(f"  - {step}")
        return 0

    if args.cmd == "profile":
        if args.all:
            profiles = all_backend_profiles()
            if args.json:
                print(json.dumps(_to_jsonable(profiles), indent=2, sort_keys=True))
            else:
                for profile in profiles:
                    print(f"{profile.backend}: {profile.label}")
            return 0

        backend = args.backend
        if backend is None:
            d = detect()
            backend = d.backend if d.backend in ("cuda", "rocm", "mps", "cpu") else "cpu"
        profile = backend_profile(backend)
        if args.json:
            print(json.dumps(_to_jsonable(profile), indent=2, sort_keys=True))
        else:
            print(f"{profile.label} ({profile.backend})")
            print("Install:")
            for step in profile.install_steps:
                print(f"  - {step}")
        return 0

    if args.cmd == "compose-gpu":
        cfg = compose_gpu_config(backend=args.backend)
        if args.json:
            print(json.dumps(_to_jsonable(cfg), indent=2, sort_keys=True))
        else:
            print(f"compose-gpu: backend={cfg.backend}")
            if cfg.deploy_devices:
                print(f"  deploy.devices: {cfg.deploy_devices}")
            if cfg.run_gpu_args:
                print(f"  docker run args: {' '.join(cfg.run_gpu_args)}")
            for note in cfg.notes:
                print(f"  note: {note}")
        return 0

    parser.error("Unknown command")
    return 2
