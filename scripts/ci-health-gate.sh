#!/usr/bin/env bash
# Preserve health JSON while mapping the CLI's tri-state result to CI policy.
#
# Usage:
#   scripts/ci-health-gate.sh [report.json]
#   DEEPIRI_GPU_FAIL_ON_WARNING=true scripts/ci-health-gate.sh

set -euo pipefail

REPORT_PATH="${1:-${DEEPIRI_GPU_HEALTH_REPORT:-gpu-health.json}}"
FAIL_ON_WARNING="${DEEPIRI_GPU_FAIL_ON_WARNING:-false}"

REPORT_DIR="$(dirname "$REPORT_PATH")"

if ! mkdir -p "$REPORT_DIR"; then
  echo "Unable to create GPU health report directory: $REPORT_DIR" >&2
  exit 2
fi

if ! : >"$REPORT_PATH"; then
  echo "Unable to write GPU health report: $REPORT_PATH" >&2
  exit 2
fi

set +e
deepiri-gpu health --json >"$REPORT_PATH"
health_exit=$?
set -e

case "$health_exit" in
  0)
    echo "GPU health gate passed; report: $REPORT_PATH"
    exit 0
    ;;
  1)
    if [[ "${GITHUB_ACTIONS:-false}" == "true" ]]; then
      echo "::warning::deepiri-gpu health reported warnings; report: $REPORT_PATH"
    else
      echo "GPU health gate warning; report: $REPORT_PATH" >&2
    fi
    case "$FAIL_ON_WARNING" in
      1|true|TRUE|True|yes|YES|Yes) exit 1 ;;
      *) exit 0 ;;
    esac
    ;;
  2)
    echo "GPU health gate failed; report: $REPORT_PATH" >&2
    exit 2
    ;;
  *)
    echo "deepiri-gpu health returned unexpected exit code $health_exit; report: $REPORT_PATH" >&2
    exit "$health_exit"
    ;;
esac
