#!/usr/bin/env bash
# Create a desktop venv that behaves like Jetson app code paths (JETSON_SIM=1).
# ONNX: CPU onnxruntime — same API as onnxruntime-gpu on the device.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv-jetson-sim}"

if ! command -v "$PYTHON" >/dev/null; then
  echo "Python not found: $PYTHON" >&2
  exit 1
fi

"$PYTHON" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel
python -m pip install -r requirements/dev-x86.txt

mkdir -p deploy
if [[ ! -f deploy/jetson.env ]]; then
  cp deploy/jetson.env.example deploy/jetson.env
  echo "Created deploy/jetson.env from example — edit JETSON_HOST before sync."
fi

cat > "$VENV_DIR/bin/activate.jetson-sim" <<'EOF'
# Source after: source .venv-jetson-sim/bin/activate
export JETSON_SIM=1
export JETSON_PROCESS_WIDTH=640
EOF

echo ""
echo "Dev Jetson-sim venv ready: $VENV_DIR"
echo "  source $VENV_DIR/bin/activate"
echo "  source $VENV_DIR/bin/activate.jetson-sim   # sets JETSON_SIM=1"
echo "  python scripts/check_env.py"
echo "  python main.py training_video/your.mp4 --process-width 640 --no-display -o out.mp4"
