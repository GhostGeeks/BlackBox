#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
REPO_URL_DEFAULT="https://github.com/GhostGeeks/BlackBox.git"
INSTALL_DIR="/opt/blackbox"
SERVICE_NAME="blackbox-oled"
APP_PATH="$INSTALL_DIR/OLED/app.py"
VENV_DIR="$INSTALL_DIR/.venv"

REPO_URL="${1:-$REPO_URL_DEFAULT}"

# Pick a runtime user:
# 1) ghostgeeks01 if it exists
# 2) pi if it exists
# 3) sudo invoker if present
APP_USER=""
if id -u ghostgeeks01 >/dev/null 2>&1; then
  APP_USER="ghostgeeks01"
elif id -u pi >/dev/null 2>&1; then
  APP_USER="pi"
elif [[ -n "${SUDO_USER:-}" ]] && id -u "$SUDO_USER" >/dev/null 2>&1; then
  APP_USER="$SUDO_USER"
else
  echo "ERROR: Could not determine an app user (ghostgeeks01/pi/SUDO_USER)."
  exit 1
fi

echo "==> Using app user: $APP_USER"
echo "==> Repo: $REPO_URL"
echo "==> Install dir: $INSTALL_DIR"

# Must be run with sudo/root
if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: Run this script with sudo:"
  echo "  sudo bash scripts/install.sh"
  exit 1
fi

# =========================
# OS PACKAGES
# =========================
echo "==> Installing OS packages..."
apt update
apt install -y \
  git \
  python3 python3-venv python3-pip \
  i2c-tools \
  alsa-utils pulseaudio-utils \
  bluez \
  libatlas-base-dev \
  libjpeg-dev zlib1g-dev \
  libfreetype6-dev \
  fonts-dejavu-core \
  raspi-config \
  swig \
  build-essential \
  python3-dev \
  liblgio-dev \
  liblgpio1 

# =========================
# DEVICE ID (persistent)
# =========================
echo "==> Creating persistent device identity..."
mkdir -p /etc/blackbox

DEVICE_FILE="/etc/blackbox/device.json"
if [[ ! -f "$DEVICE_FILE" ]]; then
  # Generate a stable ID once
  UUID="$(cat /proc/sys/kernel/random/uuid | tr -d '-' | cut -c1-12)"
  DEVICE_ID="bbx-$UUID"
  CREATED_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  cat > "$DEVICE_FILE" <<EOF
{
  "device_id": "$DEVICE_ID",
  "created_utc": "$CREATED_UTC"
}
EOF
  chmod 0644 "$DEVICE_FILE"
  echo "    Created device_id=$DEVICE_ID"
else
  echo "    Device identity already exists: $DEVICE_FILE"
fi

# Optional: set hostname from device id if hostname is still default-ish
# (Keeps your fleet unique on the network)
CUR_HOST="$(hostname)"
if [[ "$CUR_HOST" == "raspberrypi" || "$CUR_HOST" == "pi" ]]; then
  SHORT_ID="$(python3 - <<'PY'
import json
p="/etc/blackbox/device.json"
d=json.load(open(p,"r"))
did=d.get("device_id","bbx-unknown")
print(did.split("-")[-1][:6])
PY
)"
  NEW_HOST="blackbox-$SHORT_ID"
  echo "==> Setting hostname to $NEW_HOST"
  hostnamectl set-hostname "$NEW_HOST" || true
fi

# =========================
# ENABLE I2C
# =========================
echo "==> Enabling I2C..."
if command -v raspi-config >/dev/null 2>&1; then
  # Non-interactive enable I2C (idempotent)
  raspi-config nonint do_i2c 0 || true
fi

# Also ensure dtparam=i2c_arm=on is present (covers edge cases)
BOOT_CFG=""
if [[ -f /boot/firmware/config.txt ]]; then
  BOOT_CFG="/boot/firmware/config.txt"
elif [[ -f /boot/config.txt ]]; then
  BOOT_CFG="/boot/config.txt"
fi

if [[ -n "$BOOT_CFG" ]]; then
  if ! grep -qE '^\s*dtparam=i2c_arm=on' "$BOOT_CFG"; then
    echo "dtparam=i2c_arm=on" >> "$BOOT_CFG"
  fi
fi

# Ensure user is in required groups
echo "==> Ensuring $APP_USER is in groups (i2c, gpio, audio, bluetooth)..."
usermod -aG i2c,gpio,audio,bluetooth "$APP_USER" || true

# =========================
# CLONE / UPDATE REPO
# =========================
echo "==> Installing repo into $INSTALL_DIR..."
if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "    Repo exists; updating..."
  git -C "$INSTALL_DIR" fetch --all --prune
  git -C "$INSTALL_DIR" reset --hard origin/HEAD
else
  rm -rf "$INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

# Ensure ownership is sane
chown -R "$APP_USER:$APP_USER" "$INSTALL_DIR"

# =========================
# PYTHON VENV + DEPS
# =========================
echo "==> Creating/updating venv..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools

# If you have requirements.txt at repo root, install it.
# If not, install the known core deps your code uses.
if [[ -f "$INSTALL_DIR/requirements.txt" ]]; then
  echo "==> Installing requirements.txt..."
else
  echo "==> No requirements.txt found; installing core deps..."
  "$VENV_DIR/bin/pip" install \
    luma.oled luma.core \
    pillow \
    gpiozero \
    lgpio
fi

# =========================
# SUDOERS: allow reboot/poweroff without password
# (your app.py calls: sudo -n systemctl reboot/poweroff)
# =========================
echo "==> Installing sudoers rule for reboot/poweroff..."
SUDOERS_FILE="/etc/sudoers.d/${SERVICE_NAME}"
cat > "$SUDOERS_FILE" <<EOF
# Allow $APP_USER to reboot/poweroff without a password (for OLED menu)
$APP_USER ALL=NOPASSWD: /bin/systemctl reboot, /bin/systemctl poweroff
EOF
chmod 0440 "$SUDOERS_FILE"

# =========================
# SYSTEMD SERVICE
# =========================
echo "==> Installing systemd service: $SERVICE_NAME..."
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=BlackBox OLED Device (OLED/app.py)
After=network.target bluetooth.target
Wants=bluetooth.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONUNBUFFERED=1
Environment=GPIOZERO_PIN_FACTORY=lgpio
Environment=BLACKBOX_ROOT=$INSTALL_DIR
ExecStart=$VENV_DIR/bin/python $APP_PATH
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "==> Service status:"
systemctl --no-pager --full status "$SERVICE_NAME" || true

echo "==> Recent logs:"
journalctl -u "$SERVICE_NAME" -n 80 --no-pager || true

echo "==> Done."
echo "    If you changed /boot config for I2C, a reboot may be required:"
echo "      sudo reboot"
