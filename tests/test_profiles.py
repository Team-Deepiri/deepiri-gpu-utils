"""Regression tests for :mod:`deepiri_gpu_utils.profiles`."""

from __future__ import annotations

from deepiri_gpu_utils.profiles import all_backend_profiles, backend_profile


def test_all_backend_profiles_cover_every_type() -> None:
    backends = {p.backend for p in all_backend_profiles()}
    assert backends == {"cuda", "rocm", "mps", "cpu"}


def test_cuda_profile_has_install_and_compose_hints() -> None:
    profile = backend_profile("cuda")
    assert profile.required_tools == ["nvidia-smi"]
    assert profile.compose_deploy_devices == ["nvidia"]
    assert profile.docker_run_gpu_args == ["--gpus", "all"]
    assert profile.install_steps


def test_rocm_profile_has_device_mappings() -> None:
    profile = backend_profile("rocm")
    assert profile.required_tools == ["rocm-smi"]
    assert "/dev/kfd" in profile.docker_run_gpu_args
