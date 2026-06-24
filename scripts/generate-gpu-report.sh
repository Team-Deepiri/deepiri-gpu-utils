#!/usr/bin/env bash
# Generate an HTML GPU hardware report and print the path.
#
# Usage:
#   scripts/generate-gpu-report.sh
#   scripts/generate-gpu-report.sh /tmp/my-gpu-report.html

set -euo pipefail

OUT="${1:-gpu-report.html}"
deepiri-gpu visualize --html "$OUT"
echo "Open: file://$(realpath "$OUT")"
