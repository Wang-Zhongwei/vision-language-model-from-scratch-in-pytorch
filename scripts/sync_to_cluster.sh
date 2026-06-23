#!/usr/bin/env bash
# Push code (not data/checkpoints) to the H100 box over SSH.
#   REMOTE=user@h100-box REMOTE_DIR=~/vlm ./scripts/sync_to_cluster.sh
set -euo pipefail

REMOTE="${REMOTE:?set REMOTE=user@host}"
REMOTE_DIR="${REMOTE_DIR:-~/vlm}"

rsync -avz --progress \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'checkpoints' \
  --exclude 'data' \
  --exclude '*.pt' \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo "Synced to ${REMOTE}:${REMOTE_DIR}"
echo "Next:  ssh ${REMOTE}  then  cd ${REMOTE_DIR} && pip install -r requirements-train.txt && ./scripts/launch.sh configs/stage1_align.yaml"
