# deepiri-gpu-utils

## What this is

`deepiri-gpu-utils` is the **Deepiri Hybrid LLM Build & Dev Toolkit**.

**Implemented (v0.1 work in progress):** `detect()` probes NVIDIA (`nvidia-smi`), ROCm
(`rocm-smi`), Apple Silicon (`Darwin` → MPS), then CPU; `build-args` emits Cyrex-aligned
`DEVICE_TYPE`, `BASE_IMAGE`, and `BUILD_TYPE` (see `diri-cyrex` `detect_gpu.sh` + `Dockerfile`).
`doctor` / `validate` use live detection. `setup` remains a dry-run-oriented stub until Phase 2.

## CLI

Run:

```bash
deepiri-gpu --help
deepiri-gpu detect --json
deepiri-gpu doctor --json
deepiri-gpu setup --device auto --dry-run
deepiri-gpu build-args --json
deepiri-gpu build-args --device-type gpu   # override auto-detection
deepiri-gpu validate --json
deepiri-gpu ollama recommend --json
```

