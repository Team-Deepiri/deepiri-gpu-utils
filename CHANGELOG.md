# Changelog

## 0.1.0 (Unreleased)

- Initial skeleton for `deepiri-gpu-utils`
- CLI stub: `detect`, `doctor`, `setup`, `build-args`, `validate`, `ollama recommend`
- Package module stubs and minimal tests
- **Phase 1 (partial):** real `detect()` (NVIDIA via `nvidia-smi`, ROCm via `rocm-smi`, MPS on
  Darwin, CPU fallback) and `build_args_from_detection()` aligned with Cyrex `detect_gpu.sh` /
  `Dockerfile` (`BASE_IMAGE`, `DEVICE_TYPE`, `BUILD_TYPE=prebuilt`); `doctor` uses live detection;
  `build-args --device-type`; unit tests with mocks

