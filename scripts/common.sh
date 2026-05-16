#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
ONNX_DIR="${ARTIFACTS_DIR}/onnx"
TRT_DIR="${ARTIFACTS_DIR}/trt"

VENV_PY="${VENV_PY:-${ROOT_DIR}/.venv/bin/python}"

ensure_venv() {
  if [[ ! -x "${VENV_PY}" ]]; then
    echo "venv not found at ${VENV_PY}. Create/activate .venv first:" >&2
    echo "  python -m venv .venv && source .venv/bin/activate" >&2
    echo "  pip install -r requirements.server.txt" >&2
    exit 1
  fi
}

ensure_dirs() {
  mkdir -p "${ONNX_DIR}" "${TRT_DIR}"
}
