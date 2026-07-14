#!/usr/bin/env bash
# Save/diff hardware state only for explicitly selected baseline builds.

set -euo pipefail

ENABLED="${DEEPIRI_GPU_SNAPSHOT_ENABLED:-false}"
OUT="${DEEPIRI_GPU_SNAPSHOT_PATH:-${1:-gpu-snapshot.json}}"
BASELINE="${DEEPIRI_GPU_SNAPSHOT_BASELINE:-}"
DIFF_REPORT="${DEEPIRI_GPU_SNAPSHOT_DIFF_REPORT:-gpu-snapshot-diff.json}"

case "$ENABLED" in
  1|true|TRUE|True|yes|YES|Yes) ;;
  *)
    echo "GPU snapshot save skipped (set DEEPIRI_GPU_SNAPSHOT_ENABLED=true on a baseline build)"
    exit 0
    ;;
esac

deepiri-gpu snapshot save "$OUT" --json
echo "GPU snapshot saved: $OUT"

if [[ -z "$BASELINE" ]]; then
  echo "GPU snapshot diff skipped (no baseline provided)"
  exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "GPU snapshot baseline does not exist: $BASELINE" >&2
  exit 2
fi

deepiri-gpu snapshot diff "$BASELINE" "$OUT" --json >"$DIFF_REPORT"
echo "GPU snapshot diff saved: $DIFF_REPORT"
