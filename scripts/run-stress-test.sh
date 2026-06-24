#!/usr/bin/env bash
# Quick bounded stress test with JSON output.
#
# Usage:
#   scripts/run-stress-test.sh
#   scripts/run-stress-test.sh probes 2
#   scripts/run-stress-test.sh compute 10 cuda

set -euo pipefail

MODE="${1:-probes}"
DURATION="${2:-2}"
BACKEND="${3:-auto}"

deepiri-gpu stress --mode "$MODE" --duration "$DURATION" --backend "$BACKEND" --json
