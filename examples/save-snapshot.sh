#!/usr/bin/env bash
# Save a hardware snapshot for later diff in CI.
#
# Usage:
#   examples/save-snapshot.sh baseline.json

set -euo pipefail

OUT="${1:-gpu-snapshot.json}"
deepiri-gpu snapshot save "$OUT" --json
