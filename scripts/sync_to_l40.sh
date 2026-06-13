#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-yudanni}"
REMOTE_HOST="${REMOTE_HOST:-172.16.43.21}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
REMOTE_DIR="${REMOTE_DIR:-~/ssd/da_final_project}"
SSH_KEY="${SSH_KEY:-/Users/danielyu/.ssh/id_ed25519_l40}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing ${LOCAL_DIR} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
ssh -i "${SSH_KEY}" -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -avz \
  --exclude ".DS_Store" \
  --exclude ".git" \
  --exclude ".local-tools" \
  --exclude "__pycache__" \
  --exclude "downloads" \
  --exclude "outputs" \
  --exclude "data" \
  -e "ssh -i ${SSH_KEY} -p ${REMOTE_PORT}" \
  "${LOCAL_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Sync complete."
