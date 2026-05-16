#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_dirs

usage() {
  cat <<'EOF'
Jetson-only: benchmark a TensorRT engine via trtexec.

Prereqs:
  - Run on Jetson with JetPack installed.
  - 'trtexec' available on PATH.

Usage:
  scripts/trt_bench.sh [--name lane_detector] [--iters 100] [--duration 0]

Notes:
  - If artifacts/trt/<name>.engine is missing, run scripts/trt_build.sh first.
EOF
}

NAME="lane_detector"
ITERS="100"
DURATION="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

ENGINE_PATH="${TRT_DIR}/${NAME}.engine"
LOG_PATH="${TRT_DIR}/${NAME}.bench.trtexec.log"

if [[ ! -f "${ENGINE_PATH}" ]]; then
  echo "Engine not found: ${ENGINE_PATH}" >&2
  echo "Run: scripts/trt_build.sh --name ${NAME}" >&2
  exit 2
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found on PATH. On Jetson it is often at /usr/src/tensorrt/bin/trtexec" >&2
  exit 2
fi

CMD=(trtexec --loadEngine="${ENGINE_PATH}" --iterations="${ITERS}")
if [[ "${DURATION}" != "0" ]]; then
  CMD+=(--duration="${DURATION}")
fi

echo "Running: ${CMD[*]}"
("${CMD[@]}" 2>&1 | tee "${LOG_PATH}")
echo "Wrote: ${LOG_PATH}"
