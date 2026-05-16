#!/usr/bin/env bash
# Push project to Jetson and run device setup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-deploy/jetson.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy deploy/jetson.env.example and set JETSON_HOST." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

: "${JETSON_HOST:?Set JETSON_HOST in $ENV_FILE}"
REMOTE_DIR="${JETSON_REMOTE_DIR:-~/lane_tracking}"

RSYNC_EXCLUDES=(
  --exclude '.venv/'
  --exclude '.venv-jetson-sim/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.git/'
  --exclude 'training_video/'
  --exclude '*.mp4'
  --exclude 'deploy/jetson.env'
)

echo "Syncing to ${JETSON_HOST}:${REMOTE_DIR}"
rsync -avz --delete "${RSYNC_EXCLUDES[@]}" "$ROOT/" "${JETSON_HOST}:${REMOTE_DIR}/"

echo "Running setup on device..."
ssh "$JETSON_HOST" "cd ${REMOTE_DIR} && cp -n deploy/jetson.env.example deploy/jetson.env 2>/dev/null || true; bash scripts/setup_jetson_device.sh"

echo "Done. SSH in and run:"
echo "  ssh ${JETSON_HOST}"
echo "  cd ${REMOTE_DIR} && source .venv/bin/activate"
echo "  python main.py csi --csi --process-width ${JETSON_PROCESS_WIDTH:-640} --print-fps"
