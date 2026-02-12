#!/usr/bin/env bash
set -euo pipefail

REPO_URL_DEFAULT="https://github.com/GhostGeeks/BlackBox.git"
INSTALL_DIR="/opt/blackbox"
SERVICE_NAME="blackbox-oled"
APP_PATH="$INSTALL_DIR/OLED/app.py"
VENV_DIR="$INSTALL_DIR/.venv"

REPO_URL="${1:-$REPO_URL_DEFAULT}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run with sudo."
  exit 1
fi

# Detect runtime user
if id -u ghostgeeks01 >/dev/null 2>&1; then
  APP_USER="ghostgeeks01"
elif id -u pi >/dev/null 2>&1; then
  APP_USER="pi"
else
  APP_USER="${SUDO_USER:-root}"
fi

echo "==> Using app user: $APP_USER"

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
  libjpeg-dev zlib1g-dev \
  libfreetype6-dev \
  fonts-dejavu-core \
  raspi-config \
  swig \
  build-essential \
  python3-dev \
  liblgpio-dev \
  liblgpio1

# =========================
# ENABLE I2C
# =========================
raspi-config nonint do_i2c 0 || true

# =========================
# CLONE / UPDATE REPO
# =========================
if [[ -d "$INSTALL_DIR/.git" ]]; then
  git -C "$INSTALL_DIR" pull --ff-only
else
  rm -rf "$INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

chown -R "$APP_USER:$APP_USER" "$INSTALL_DIR"

# =========================
# PYTHON VENV + DEPS
# =========================
echo "==> Creating venv..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools

echo "==> Installing Python requirements..."
"$VENV_DIR/bin/python" -m pip install -r "$INSTALL_DIR/requirements.txt"

# =========================
# SYSTEMD SERVICE
# =========================
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=BlackBox OLED Device
After=network.target bluetooth.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$INSTALL_DIR
Environment=PYTHONUNBUFFERED=1
Environment=GPIOZERO_PIN_FACTORY=lgpio
ExecStart=$VENV_DIR/bin/python $APP_PATH
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo "==> Verifying venv interpreter..."
test -x "$VENV_DIR/bin/python"

echo "==> Verifying imports..."
"$VENV_DIR/bin/python" -c "import gpiozero, lgpio; import luma.core, luma.oled; import PIL; print('imports OK')"

echo "==> Install complete."
