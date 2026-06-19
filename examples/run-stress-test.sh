#!/usr/bin/env bash
# Quick bounded stress test with JSON output.
#
# Usage:
#   examples/run-stress-test.sh
#   examples/run-stress-test.sh probes 2
#   examples/run-stress-test.sh compute 10 cuda

set -euo pipefail

MODE="${1:-probes}"
DURATION="${2:-2}"
BACKEND="${3:-auto}"

deepiri-gpu stress --mode "$MODE" --duration "$DURATION" --backend "$BACKEND" --json
