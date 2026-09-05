# deepiri-gpu-utils

## What this is

`deepiri-gpu-utils` is the **Deepiri Hybrid LLM Build & Dev Toolkit** — detection, Docker build
args, readiness checks, setup runbooks, Ollama tiering, and optional PyTorch device resolution.

It is also Deepiri's reusable GPU primitive library. Generic host concerns live
here: backend detection, normalized inventory and memory, health, deterministic
selection, container hints, hardware readiness, snapshots, serialization, and
optional torch runtime resolution. Application concepts such as rooms,
scheduler leases, training rounds, transport, databases, and remote workflow
orchestration remain outside this package.

## Installation and supported backends

```bash
pip install deepiri-gpu-utils

# Optional; torch is never imported at package import time or required for
# non-torch features.
pip install 'deepiri-gpu-utils[torch]'
```

The canonical backend enum covers NVIDIA CUDA, AMD ROCm, Apple MPS, CPU, and an
explicit unknown state. NVIDIA metrics require a working `nvidia-smi`. Detailed
AMD metrics require a `rocm-smi` version with JSON output. MPS is represented as
a real accelerator backend, but macOS does not expose the same per-device
telemetry. Missing tools, malformed output, failed processes, and CPU-only hosts
produce typed results with warnings instead of uncaught probe exceptions.

## Canonical Python API

```python
from deepiri_gpu_utils import (
    GpuBackend,
    GpuSelectionPolicy,
    check_gpu_health,
    detect_backend,
    discover_gpus,
    resolve_runtime,
    select_gpu,
)

backend = detect_backend()
inventory = discover_gpus()
selection = select_gpu(
    GpuSelectionPolicy(
        preferred_backend=GpuBackend.CUDA,
        minimum_total_mib=12 * 1024,
        minimum_free_mib=8 * 1024,
        maximum_utilization_percent=80,
        require_healthy=True,
    ),
    inventory=inventory,
)
health = check_gpu_health(inventory=inventory)
runtime = resolve_runtime(inventory=inventory)

payload = inventory.to_dict()
restored = type(inventory).from_dict(payload)
```

`GpuMemory` uses binary MiB and GiB: one GiB is exactly 1024 MiB. Canonical
primitives are frozen dataclasses; devices and warnings use tuples, and nested
backend metadata is frozen. `to_dict()` emits stable enum strings and ordinary
JSON-friendly containers, while `from_dict()` round-trips optional fields and
metadata. Backend metadata is deliberately best-effort: consumers should rely
on normalized fields where present and treat vendor-specific keys as optional.

`preferred_backend` is a preference: selection falls back to another backend
when necessary. Minimum-memory and maximum-utilization requirements are strict;
a device is rejected when the metric needed to prove a requirement is missing.
Health distinguishes `healthy`, `unhealthy`, `no_gpu`, and
`tooling_unavailable`. Runtime capabilities report torch installation and
usability separately.

The pre-0.3 APIs and CLI remain supported, including `detect()`,
`gpu_inventory()`, `choose_suitable_gpu()`, `health_check()`,
`resolve_torch_device()`, `GPUInfo`, `GPUInventoryResult`, and `GPUSelection`.
The canonical APIs are additive and reuse the existing detection, subprocess,
and health foundations.

### Implemented

| Area | Behavior |
|------|----------|
| **`detect`** | `nvidia-smi`; Linux **lspci** fallback when drivers missing; ROCm; Darwin → MPS; CPU; WSL hints |
| **`build-args`** | Cyrex-style `BASE_IMAGE`, `DEVICE_TYPE`, `BUILD_TYPE` (`detect_gpu.sh` + `Dockerfile`) |
| **`doctor`** | RAM, Docker CLI, NVIDIA Container Toolkit hints, **DMI** via `dmidecode` when root, WSL notes; status `ok` / `warn` |
| **`setup`** | Printable runbooks for **nvidia / amd / apple / cpu / auto** (no `sudo` execution from Python) |
| **`ollama recommend`** | Hardware tiers + `categorize_model` ported from `check-ollama-models.sh` (logic only; no Docker UI) |
| **`torch-device`** | Uses **`torch`** when `[torch]` extra installed; otherwise heuristics from `detect()` |
| **`validate`** | JSON bundle: detect + doctor + build-args + ollama + torch-device |
| **`inventory`** | Read-only multi-GPU inventory + optional VRAM suitability check |
| **`summary`** | Aggregate snapshot: detect + doctor + inventory + build-args + ollama + torch-device |
| **`export-env`** | Shell `export` lines for docker build args (compose/CI friendly) |
| **`model-fit`** | Check whether a specific Ollama model fits detected hardware tier |
| **`install-check`** | Driver/tooling readiness for NVIDIA, AMD/ROCm, Apple/MPS, or CPU |
| **`profile`** | Canonical install + docker profile per backend (`cuda` / `rocm` / `mps` / `cpu`) |
| **`compose-gpu`** | Docker Compose GPU device/deploy hints for downstream repos |
| **`visualize`** | ASCII terminal dashboard or self-contained HTML hardware report |
| **`model-matrix`** | Curated Ollama model fit matrix for the detected host |
| **`workload`** | Heuristic memory fit estimate for a model + context size |
| **`stress`** | Bounded GPU/CPU stress test with read-only VRAM/util telemetry |
| **`health`** | CI health gate (exit 0=ok, 1=warn, 2=fail) |
| **`env-hints`** | Recommended runtime env vars per backend |
| **`capacity`** | Estimate concurrent model instances that fit in memory |
| **`top`** | List NVIDIA GPU compute processes |
| **`snapshot`** | Save/diff hardware snapshot JSON for CI regression |

Optional extra: `pip install 'deepiri-gpu-utils[torch]'`.

## CLI

```bash
deepiri-gpu --help
deepiri-gpu detect --json
deepiri-gpu doctor --json
deepiri-gpu setup --device auto              # default: dry-run style; add --yes to mark confirmed
deepiri-gpu build-args --json
deepiri-gpu build-args --device-type gpu
deepiri-gpu validate --json
deepiri-gpu ollama recommend --json
deepiri-gpu torch-device --policy auto --json
deepiri-gpu inventory --json
deepiri-gpu inventory --min-memory-gb 8 --json
deepiri-gpu summary --json
eval "$(deepiri-gpu export-env)"
deepiri-gpu model-fit mistral:7b --json
deepiri-gpu install-check --json
deepiri-gpu install-check --device amd --json
deepiri-gpu install-check --all --json
deepiri-gpu profile --all --json
deepiri-gpu compose-gpu --json
deepiri-gpu visualize
deepiri-gpu visualize --html gpu-report.html
deepiri-gpu model-matrix --json
deepiri-gpu workload mistral:7b --json
deepiri-gpu stress --duration 5 --json
deepiri-gpu stress --mode probes --duration 2 --json
deepiri-gpu health --json
deepiri-gpu env-hints --json
deepiri-gpu capacity mistral:7b --json
deepiri-gpu top --json
deepiri-gpu snapshot save baseline.json
deepiri-gpu snapshot diff baseline.json current.json
```

## Development

Requires Python >= 3.11. No runtime dependencies; `torch` is an optional extra.

```bash
# Create / activate a virtual environment (example)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install the package with dev tooling (pytest + ruff)
pip install -e ".[dev]"

# Optionally include the PyTorch extra for torch-device tests
pip install -e ".[dev,torch]"

# Run the test suite (CPU-only; no GPU required)
pytest

# Lint
ruff check .

# Byte-compile check
python -m compileall src tests
```

The test suite mocks all external probes (`nvidia-smi`, `rocm-smi`, `lspci`,
`dmidecode`, `sysctl`, Docker, and `torch`), so it runs deterministically on a
CPU-only machine without GPU hardware.
