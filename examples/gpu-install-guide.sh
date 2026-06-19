#!/usr/bin/env bash
# Print install + verify steps for the detected (or requested) GPU backend.
#
# Usage:
#   examples/gpu-install-guide.sh
#   examples/gpu-install-guide.sh amd
#   examples/gpu-install-guide.sh --all

set -euo pipefail

DEVICE="${1:-auto}"
if [[ "$DEVICE" == "--all" ]]; then
  deepiri-gpu install-check --all
  exit 0
fi

deepiri-gpu install-check --device "$DEVICE" --json | python3 -c '
import json
import sys

item = json.load(sys.stdin)
print(f"# {item[\"profile_label\"]} install guide (ready={item[\"ready\"]})")
missing = item.get("missing_required") or []
if missing:
    print(f"# missing: {\", \".join(missing)}")
print()
print("## Install steps")
for step in item.get("install_steps", []):
    print(f"- {step}")
print()
print("## Verify")
for cmd in item.get("verify_commands", []):
    print(f"  {cmd}")
'
