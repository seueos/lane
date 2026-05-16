#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_venv
ensure_dirs

usage() {
  cat <<'EOF'
Usage:
  scripts/export_onnx.sh --src /path/to/model.onnx [--name lane_detector]

Writes:
  artifacts/onnx/<name>.onnx
  artifacts/onnx/<name>.json   (sha256, size, etc)
EOF
}

SRC=""
NAME="lane_detector"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${SRC}" ]]; then
  echo "--src is required" >&2
  usage
  exit 2
fi

"${VENV_PY}" "${ROOT_DIR}/scripts/export_onnx_artifact.py" \
  --src "${SRC}" \
  --name "${NAME}" \
  --out-dir "${ONNX_DIR}"
