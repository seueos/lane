#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_venv
ensure_dirs

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_onnx.sh [--name lane_detector] [--batch 1] [--providers CPUExecutionProvider]

Defaults to validating:
  artifacts/onnx/<name>.onnx
EOF
}

NAME="lane_detector"
BATCH="1"
PROVIDERS="CPUExecutionProvider"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --providers) PROVIDERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

ONNX_PATH="${ONNX_DIR}/${NAME}.onnx"
if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ONNX artifact not found: ${ONNX_PATH}" >&2
  echo "Run: scripts/export_onnx.sh --src /path/to/model.onnx --name ${NAME}" >&2
  exit 2
fi

"${VENV_PY}" "${ROOT_DIR}/scripts/validate_onnx.py" \
  --onnx "${ONNX_PATH}" \
  --batch "${BATCH}" \
  --providers "${PROVIDERS}"
