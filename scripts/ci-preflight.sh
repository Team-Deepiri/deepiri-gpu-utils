#!/usr/bin/env bash
# CI / compose preflight: validate JSON contracts and exit non-zero on doctor warn.
#
# Usage:
#   scripts/ci-preflight.sh
#   scripts/ci-preflight.sh --strict   # also fail when no GPUs are inventoried

set -euo pipefail

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

echo "==> deepiri-gpu validate --json"
deepiri-gpu validate --json >/dev/null

echo "==> deepiri-gpu summary --json"
SUMMARY_JSON="$(deepiri-gpu summary --json)"
python3 - <<'PY' "$SUMMARY_JSON" "$STRICT"
import json
import sys

payload = json.loads(sys.argv[1])
strict = int(sys.argv[2])

required = {
    "detect", "doctor", "inventory", "build_args", "ollama",
    "torch_device", "system_ram_gb", "gpu_count", "total_vram_gb", "notes",
}
missing = required - set(payload)
if missing:
    raise SystemExit(f"summary missing keys: {sorted(missing)}")

status = payload["doctor"]["status"]
if status == "warn":
    print("doctor status=warn (see runbook in summary JSON)")
    raise SystemExit(2)

if strict and payload["gpu_count"] == 0 and payload["detect"]["backend"] != "cpu":
    print("strict: detect backend is not cpu but inventory gpu_count=0")
    raise SystemExit(3)

print("preflight ok")
PY

echo "==> deepiri-gpu inventory --json"
deepiri-gpu inventory --json >/dev/null

echo "All preflight checks passed."
