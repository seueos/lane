#!/usr/bin/env bash
# Copy pinned ONNX artifacts to Jetson (server → device).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_dirs

ENV_FILE="${ENV_FILE:-deploy/jetson.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy deploy/jetson.env.example and set JETSON_HOST." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

: "${JETSON_HOST:?Set JETSON_HOST in $ENV_FILE}"
REMOTE_DIR="${JETSON_REMOTE_DIR:-~/lane_tracking}"

if [[ ! -d "${ONNX_DIR}" ]] || [[ -z "$(ls -A "${ONNX_DIR}" 2>/dev/null || true)" ]]; then
  echo "No files in ${ONNX_DIR}. Run scripts/export_onnx.sh on the server first." >&2
  exit 2
fi

echo "Syncing ${ONNX_DIR}/ → ${JETSON_HOST}:${REMOTE_DIR}/artifacts/onnx/"
rsync -avz "${ONNX_DIR}/" "${JETSON_HOST}:${REMOTE_DIR}/artifacts/onnx/"

echo "Done. On Jetson:"
echo "  bash scripts/trt_build.sh --name lane_detector --fp16 --shapes 'input:1x3x360x640'"
echo "  bash scripts/trt_bench.sh --name lane_detector"
