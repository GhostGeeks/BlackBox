#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="/opt/blackbox"
SERVICE_NAME="blackbox-oled"
VENV_DIR="$INSTALL_DIR/.venv"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: Run with sudo:"
  echo "  sudo bash scripts/update.sh"
  exit 1
fi

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  echo "ERROR: $INSTALL_DIR is not a git repo. Run install.sh first."
  exit 1
fi

echo "==> Stopping service..."
systemctl stop "$SERVICE_NAME" || true

echo "==> Updating repo..."
git -C "$INSTALL_DIR" pull --ff-only

echo "==> Updating Python deps..."
if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
  "$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
fi

echo "==> Restarting service..."
systemctl restart "$SERVICE_NAME"

echo "==> Status:"
systemctl --no-pager --full status "$SERVICE_NAME" || true

echo "==> Recent logs:"
journalctl -u "$SERVICE_NAME" -n 80 --no-pager || true
