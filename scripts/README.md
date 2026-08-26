# Scripts

Helper scripts that compose `deepiri-gpu-utils` for compose builds, CI, and
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
source scripts/export-build-env.sh
echo "$BASE_IMAGE" "$DEVICE_TYPE"

# optional prefix for multi-service compose
PREFIX=CYREX_ source scripts/export-build-env.sh
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
scripts/ci-preflight.sh
scripts/ci-preflight.sh --strict   # fail when detect≠cpu but inventory is empty
```

### Health gate and hardware baselines

`health` has intentional tri-state exits: `0=ok`, `1=warn`, and `2=fail`.
The CI wrapper always writes the JSON report. Warnings are visible but do not
block by default, which keeps ordinary CPU-only GitHub runners viable; hard
failures still block. GPU-required jobs can set
`DEEPIRI_GPU_FAIL_ON_WARNING=true`.

```bash
scripts/ci-health-gate.sh gpu-health.json
DEEPIRI_GPU_FAIL_ON_WARNING=true scripts/ci-health-gate.sh gpu-health.json
```

Upload `gpu-health.json` with the CI provider's artifact mechanism using an
`always()` condition so a failed gate does not discard diagnostics.

Hardware snapshots describe the runner, so do not create or commit them from
ordinary pull-request runners. Snapshot capture is a no-op until a real
baseline/release/nightly job explicitly enables it:

```bash
DEEPIRI_GPU_SNAPSHOT_ENABLED=true \
  DEEPIRI_GPU_SNAPSHOT_PATH=gpu-snapshot.json \
  scripts/save-snapshot.sh
```

Diffing runs only when `DEEPIRI_GPU_SNAPSHOT_BASELINE` names an existing
baseline file. The JSON diff defaults to `gpu-snapshot-diff.json` and can be
changed with `DEEPIRI_GPU_SNAPSHOT_DIFF_REPORT`.

```bash
DEEPIRI_GPU_SNAPSHOT_ENABLED=true \
  DEEPIRI_GPU_SNAPSHOT_BASELINE=approved-baseline.json \
  scripts/save-snapshot.sh candidate.json
```

## GPU install guides (all backends)

NVIDIA (CUDA), AMD (ROCm), Apple (MPS), and CPU-only each have a canonical
profile with install steps, verify commands, and compose/docker hints:

```bash
deepiri-gpu profile --all --json
deepiri-gpu install-check --all --json
scripts/gpu-install-guide.sh
scripts/gpu-install-guide.sh amd
```

## Library API scripts

```bash
python scripts/hardware-summary.py
python scripts/hardware-summary.py --full
python scripts/check-model-fit.py mistral:7b --json
scripts/generate-gpu-report.sh
deepiri-gpu visualize
deepiri-gpu model-matrix
scripts/run-stress-test.sh probes 2
scripts/ci-health-gate.sh
scripts/save-snapshot.sh baseline.json
```

## Ollama on Linux / WSL (NVIDIA Container Toolkit)

See `deepiri-platform/scripts/docs/README-GPU-SETUP.md` for one-time Docker GPU setup.
