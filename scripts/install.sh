#!/usr/bin/env bash
# Install deepiri-gpu-utils via curl:
#   curl -fsSL https://raw.githubusercontent.com/Team-Deepiri/deepiri-gpu-utils/main/scripts/install.sh | bash
set -euo pipefail

REPO="Team-Deepiri/deepiri-gpu-utils"
REPO_URL="https://github.com/${REPO}.git"
BRANCH="${DEEPIRI_GPU_UTILS_BRANCH:-main}"
KEEP_DIR="${DEEPIRI_GPU_UTILS_KEEP_DIR:-0}"
WITH_TORCH="${DEEPIRI_GPU_UTILS_TORCH:-0}"

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Clone (when needed) and pip-install deepiri-gpu-utils into ~/.local/bin.

Options:
  -h, --help     Show this help
  --dry-run      Print actions without installing
  --with-torch   Install optional [torch] extra

Environment:
  DEEPIRI_GPU_UTILS_SRC         Existing checkout
  DEEPIRI_GPU_UTILS_BRANCH      Git branch (default: main)
  DEEPIRI_GPU_UTILS_KEEP_DIR    Keep clone directory when set to 1
  DEEPIRI_GPU_UTILS_TORCH       Set to 1 to install [torch] extra

Requires: git, python3 (>=3.11)
Verify:   deepiri-gpu detect --json
EOF
}

log() { printf '==> %s\n' "$*"; }

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --with-torch) WITH_TORCH=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for cmd in git python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: $cmd is required." >&2; exit 1; }
done

ROOT=""
CLEANUP=""
LOCAL_BIN="${HOME}/.local/bin"

if [[ -n "${DEEPIRI_GPU_UTILS_SRC:-}" && -f "${DEEPIRI_GPU_UTILS_SRC}/pyproject.toml" ]]; then
  ROOT="${DEEPIRI_GPU_UTILS_SRC}"
elif [[ -n "${BASH_SOURCE[0]:-}" ]] && [[ "${BASH_SOURCE[0]}" != bash ]] && [[ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pyproject.toml" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  ROOT="$(mktemp -d)"
  [[ "$KEEP_DIR" != "1" ]] && CLEANUP="$ROOT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would clone ${REPO_URL} to ${ROOT}"
    log "Would pip install -e . into venv and link deepiri-gpu to ${LOCAL_BIN}"
    exit 0
  fi
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Would pip install from ${ROOT}"
  exit 0
fi

trap '[[ -n "$CLEANUP" ]] && rm -rf "$CLEANUP"' EXIT
cd "$ROOT"

VENV="${ROOT}/.venv"
log "Creating venv at ${VENV}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel -q

EXTRAS=""
[[ "$WITH_TORCH" == "1" ]] && EXTRAS="[torch]"
log "Installing deepiri-gpu-utils${EXTRAS}"
"$VENV/bin/pip" install -e ".${EXTRAS}" -q

mkdir -p "$LOCAL_BIN"
ln -sf "$VENV/bin/deepiri-gpu" "$LOCAL_BIN/deepiri-gpu"

export PATH="${LOCAL_BIN}:${PATH}"
log "Installed deepiri-gpu"
deepiri-gpu --help >/dev/null
echo ""
echo "Verify: deepiri-gpu detect --json"
if [[ ":$PATH:" != *":${LOCAL_BIN}:"* ]]; then
  echo "Add to your shell profile: export PATH=\"${LOCAL_BIN}:\$PATH\""
fi
