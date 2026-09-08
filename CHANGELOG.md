# Changelog

## 0.3.0 (Unreleased)

- Add immutable canonical GPU primitives with explicit serialization.
- Add `detect_backend()`, `discover_gpus()`, `select_gpu()`,
  `check_gpu_health()`, and `resolve_runtime()` while preserving existing APIs.
- Normalize CUDA and ROCm inventory metrics and represent MPS/CPU states.
- Keep torch optional and degrade safely for absent or broken installations.
- Make POSIX shell integration tests and Linux `dmidecode` root checks portable
  on Windows.

## 0.1.0 (Unreleased)

- Initial skeleton for `deepiri-gpu-utils`
- CLI: `detect`, `doctor`, `setup`, `build-args`, `validate`, `ollama recommend`, `torch-device`
- **Phase 1:** `detect()` + `build_args_from_detection()` aligned with Cyrex `detect_gpu.sh` /
  `Dockerfile`; `build-args --device-type`
- **Phase 2:** `doctor()` with RAM/Docker/NVIDIA toolkit/DMI (`dmidecode` when root), WSL notes;
  `setup` / `setup_device` / `setup_device_mac` runbooks (read-only, no privileged execution)
- **Phase 3:** `ollama.recommend_models` — `setup_tier` + `categorize_model` from
  `check-ollama-models.sh`; `validate` includes ollama summary
- **Phase 4:** `resolve_torch_device()` with optional `[torch]` extra
- **Phase 5:** Linux **lspci** NVIDIA path when `nvidia-smi` missing; WSL/ROCm warnings in `detect`
- **Phase 6:** `scripts/` compose fragment + README; top-level README matrix
