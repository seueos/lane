#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_dirs

usage() {
  cat <<'EOF'
Jetson-only (recommended): build a TensorRT engine via trtexec.

Prereqs:
  - Run this on Jetson Orin Nano with JetPack installed.
  - Ensure 'trtexec' is on PATH (typically /usr/src/tensorrt/bin/trtexec).

Usage:
  scripts/trt_build.sh [--name lane_detector] [--fp16] [--max-workspace-mib 1024] [--shapes "input:1x3x360x640"]

Reads:
  artifacts/onnx/<name>.onnx
Writes:
  artifacts/trt/<name>.engine
  artifacts/trt/<name>.trtexec.log

Lane detector default (640x360 NCHW): pass --shapes with your graph input name, e.g.:
  --shapes "input:1x3x360x640"
EOF
}

NAME="lane_detector"
FP16="0"
MAX_WORKSPACE_MIB="1024"
SHAPES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --fp16) FP16="1"; shift 1 ;;
    --max-workspace-mib) MAX_WORKSPACE_MIB="$2"; shift 2 ;;
    --shapes) SHAPES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

ONNX_PATH="${ONNX_DIR}/${NAME}.onnx"
ENGINE_PATH="${TRT_DIR}/${NAME}.engine"
LOG_PATH="${TRT_DIR}/${NAME}.trtexec.log"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ONNX artifact not found: ${ONNX_PATH}" >&2
  echo "Copy/export ONNX into artifacts/onnx first (export_onnx.sh on server, then rsync artifacts/onnx/)." >&2
  exit 2
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found on PATH. On Jetson it is often at /usr/src/tensorrt/bin/trtexec" >&2
  exit 2
fi

CMD=(trtexec --onnx="${ONNX_PATH}" --saveEngine="${ENGINE_PATH}" --workspace="${MAX_WORKSPACE_MIB}")
if [[ "${FP16}" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ -n "${SHAPES}" ]]; then
  CMD+=(--shapes="${SHAPES}")
fi

echo "Running: ${CMD[*]}"
("${CMD[@]}" 2>&1 | tee "${LOG_PATH}")
echo "Wrote: ${ENGINE_PATH}"
echo "Wrote: ${LOG_PATH}"
