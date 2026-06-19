# deepiri-gpu-utils

## What this is

`deepiri-gpu-utils` is the **Deepiri Hybrid LLM Build & Dev Toolkit** — detection, Docker build
args, readiness checks, setup runbooks, Ollama tiering, and optional PyTorch device resolution.

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
