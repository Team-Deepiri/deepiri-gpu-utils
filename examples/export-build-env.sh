#!/usr/bin/env bash
# Source Cyrex-style docker build args from detection (read-only).
#
# Usage:
#   source examples/export-build-env.sh
#   echo "$BASE_IMAGE" "$DEVICE_TYPE"
#
# Or with a variable prefix for multi-service compose:
#   PREFIX=CYREX_ source examples/export-build-env.sh

set -euo pipefail

PREFIX="${PREFIX:-}"
eval "$(deepiri-gpu export-env --device-type auto --prefix "$PREFIX")"
