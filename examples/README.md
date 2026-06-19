# Examples

Runnable helpers that compose `deepiri-gpu-utils` for compose builds, CI, and
local preflight. All are read-only — they never mutate GPU state or the
environment.

## Compose: GPU build args

Cyrex-style services expect `BASE_IMAGE` and `DEVICE_TYPE` from detection. From the repo root:

```bash
eval "$(deepiri-gpu build-args | sed -n 's/^\\(.*\\)=\\(.*\\)$/export \\1=\\2/p')"
docker compose -f docker-compose.dev.yml build cyrex
```

Or use the dedicated export command / shell helpers:

```bash
source examples/export-build-env.sh
echo "$BASE_IMAGE" "$DEVICE_TYPE"

# optional prefix for multi-service compose
PREFIX=CYREX_ source examples/export-build-env.sh
```

Or pass explicitly:

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime \
  --build-arg DEVICE_TYPE=gpu \
  --build-arg BUILD_TYPE=prebuilt \
  -f diri-cyrex/Dockerfile .
```

See `cyrex-gpu.fragment.yml` and `compose-env.fragment.sh` for minimal patterns.

## CI / compose preflight

```bash
examples/ci-preflight.sh
examples/ci-preflight.sh --strict   # fail when detect≠cpu but inventory is empty
```

## GPU install guides (all backends)

NVIDIA (CUDA), AMD (ROCm), Apple (MPS), and CPU-only each have a canonical
profile with install steps, verify commands, and compose/docker hints:

```bash
deepiri-gpu profile --all --json
deepiri-gpu install-check --all --json
examples/gpu-install-guide.sh
examples/gpu-install-guide.sh amd
```

## Library API scripts

```bash
python examples/hardware-summary.py
python examples/hardware-summary.py --full
python examples/check-model-fit.py mistral:7b --json
examples/generate-gpu-report.sh
deepiri-gpu visualize
deepiri-gpu model-matrix
examples/run-stress-test.sh probes 2
examples/ci-health-gate.sh
examples/save-snapshot.sh baseline.json
```

## Ollama on Linux / WSL (NVIDIA Container Toolkit)

See `deepiri-platform/scripts/docs/README-GPU-SETUP.md` for one-time Docker GPU setup.
