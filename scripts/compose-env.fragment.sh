#!/usr/bin/env bash
# Minimal fragment for Docker Compose build scripts.
#
#   source scripts/compose-env.fragment.sh
#   docker compose build --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg DEVICE_TYPE="$DEVICE_TYPE" cyrex

set -euo pipefail
eval "$(deepiri-gpu export-env)"
