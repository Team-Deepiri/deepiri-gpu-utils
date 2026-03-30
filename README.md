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