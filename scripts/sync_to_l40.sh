#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-yudanni}"
REMOTE_HOST="${REMOTE_HOST:-172.16.43.21}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
REMOTE_DIR="${REMOTE_DIR:-~/ssd/da_final_project}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing ${LOCAL_DIR} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -avz \
  --exclude ".DS_Store" \
  --exclude "__pycache__" \
  --exclude "outputs" \
  --exclude "data" \
  -e "ssh -p ${REMOTE_PORT}" \
  "${LOCAL_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Sync complete."
