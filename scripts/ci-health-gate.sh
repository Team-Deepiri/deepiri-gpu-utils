#!/usr/bin/env bash
# CI health gate: exit 0=ok, 1=warn, 2=fail
#
# Usage:
#   scripts/ci-health-gate.sh

set -euo pipefail

deepiri-gpu health --json
exit $?
