#!/usr/bin/env bash
# Run ON the Jetson (or via: ssh jetson 'bash -s' < scripts/setup_jetson_device.sh)
# Uses system OpenCV (GStreamer) + jetson-ai-lab onnxruntime-gpu wheels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f deploy/jetson.env ]]; then
  # shellcheck source=/dev/null
  source deploy/jetson.env
fi

ONNX_INDEX_URL="${ONNX_INDEX_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
JETSON_PYTHON="${JETSON_PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "Warning: /etc/nv_tegra_release not found — is this a Jetson?" >&2
fi

echo "Installing system packages (sudo)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-pip python3-opencv \
  gstreamer1.0-tools gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-libav \
  v4l-utils

"$JETSON_PYTHON" -m venv "$VENV_DIR" --system-site-packages
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel
python -m pip install -r requirements/jetson.txt
python -m pip install onnxruntime-gpu --index-url "$ONNX_INDEX_URL"

python scripts/check_env.py
echo ""
echo "Jetson venv ready. Example CSI run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py csi --csi --process-width ${JETSON_PROCESS_WIDTH:-640} --print-fps"
