# Examples

## Compose: GPU build args

Cyrex-style services expect `BASE_IMAGE` and `DEVICE_TYPE` from detection. From the repo root:

```bash
eval "$(deepiri-gpu build-args | sed -n 's/^\\(.*\\)=\\(.*\\)$/export \\1=\\2/p')"
docker compose -f docker-compose.dev.yml build cyrex
```

Or pass explicitly:

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime \
  --build-arg DEVICE_TYPE=gpu \
  --build-arg BUILD_TYPE=prebuilt \
  -f diri-cyrex/Dockerfile .
```

See `cyrex-gpu.fragment.yml` for a minimal `build.args` pattern.

## Ollama on Linux / WSL (NVIDIA Container Toolkit)

See `deepiri-platform/scripts/docs/README-GPU-SETUP.md` for one-time Docker GPU setup.
