#!/usr/bin/env bash
# Generate an HTML GPU hardware report and print the path.
#
# Usage:
#   examples/generate-gpu-report.sh
#   examples/generate-gpu-report.sh /tmp/my-gpu-report.html

set -euo pipefail

OUT="${1:-gpu-report.html}"
deepiri-gpu visualize --html "$OUT"
echo "Open: file://$(realpath "$OUT")"
