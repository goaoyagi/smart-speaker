#!/usr/bin/env bash
#
# Install the systemd units that start SearXNG (Docker) and the smart speaker
# (src/main.py) automatically when the Raspberry Pi boots.
#
# Usage (on the Raspberry Pi, from the repository root):
#   sudo ./deploy/install.sh
#
# Environment variable overrides:
#   RUN_USER            user that runs src.main            (default: $SUDO_USER)
#   PYTHON_BIN          python interpreter                 (default: <repo>/venv/bin/python3 if present)
#   SEARXNG_PORT        host port for SearXNG              (default: 8080)
#   SEARXNG_IMAGE       container image                    (default: searxng/searxng)
#   SEARXNG_CONFIG_DIR  settings.yml location on the host  (default: /etc/searxng)
#   CONTAINER_NAME      container name                     (default: searxng)
#   START_SERVICES      start the units after install      (default: true)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy"
SYSTEMD_DIR="/etc/systemd/system"

RUN_USER="${RUN_USER:-${SUDO_USER:-}}"
SEARXNG_PORT="${SEARXNG_PORT:-8080}"
SEARXNG_IMAGE="${SEARXNG_IMAGE:-searxng/searxng}"
SEARXNG_CONFIG_DIR="${SEARXNG_CONFIG_DIR:-/etc/searxng}"
CONTAINER_NAME="${CONTAINER_NAME:-searxng}"
START_SERVICES="${START_SERVICES:-true}"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

[ "$(id -u)" -eq 0 ] || die "root privileges are required. Run: sudo ./deploy/install.sh"
[ -n "${RUN_USER}" ] || die "RUN_USER could not be detected. Run: sudo RUN_USER=pi ./deploy/install.sh"
id "${RUN_USER}" >/dev/null 2>&1 || die "user '${RUN_USER}' does not exist"
command -v docker >/dev/null 2>&1 || die "docker is not installed. Install it first: curl -fsSL https://get.docker.com | sh"
command -v systemctl >/dev/null 2>&1 || die "systemd is required"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "${PROJECT_DIR}/venv/bin/python3" ]; then
        PYTHON_BIN="${PROJECT_DIR}/venv/bin/python3"
    else
        PYTHON_BIN="$(command -v python3 || true)"
    fi
fi
[ -x "${PYTHON_BIN}" ] || die "python interpreter not found. Set PYTHON_BIN explicitly."

# Only reference groups that exist, otherwise systemd refuses to start the unit.
SUPPLEMENTARY_GROUPS=""
for group in audio video gpio spi i2c; do
    if getent group "${group}" >/dev/null 2>&1; then
        SUPPLEMENTARY_GROUPS="${SUPPLEMENTARY_GROUPS}${SUPPLEMENTARY_GROUPS:+ }${group}"
    fi
done
[ -n "${SUPPLEMENTARY_GROUPS}" ] || SUPPLEMENTARY_GROUPS="${RUN_USER}"

if [ ! -f "${PROJECT_DIR}/.env" ]; then
    info "no .env found; src/config.py defaults will be used"
fi

info "SearXNG settings: ${SEARXNG_CONFIG_DIR}/settings.yml"
install -d -m 755 "${SEARXNG_CONFIG_DIR}"
if [ -f "${SEARXNG_CONFIG_DIR}/settings.yml" ]; then
    info "settings.yml already exists; keeping it (delete it to regenerate)"
else
    secret_key="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    sed "s|__SEARXNG_SECRET_KEY__|${secret_key}|g" \
        "${DEPLOY_DIR}/searxng/settings.yml" > "${SEARXNG_CONFIG_DIR}/settings.yml"
    chmod 644 "${SEARXNG_CONFIG_DIR}/settings.yml"
fi

info "installing ${SYSTEMD_DIR}/searxng.service"
sed -e "s|__CONTAINER_NAME__|${CONTAINER_NAME}|g" \
    -e "s|__SEARXNG_PORT__|${SEARXNG_PORT}|g" \
    -e "s|__SEARXNG_IMAGE__|${SEARXNG_IMAGE}|g" \
    -e "s|__SEARXNG_CONFIG_DIR__|${SEARXNG_CONFIG_DIR}|g" \
    "${DEPLOY_DIR}/searxng.service" > "${SYSTEMD_DIR}/searxng.service"

info "installing ${SYSTEMD_DIR}/smart-speaker.service (user=${RUN_USER}, python=${PYTHON_BIN})"
sed -e "s|__RUN_USER__|${RUN_USER}|g" \
    -e "s|__SUPPLEMENTARY_GROUPS__|${SUPPLEMENTARY_GROUPS}|g" \
    -e "s|__PROJECT_DIR__|${PROJECT_DIR}|g" \
    -e "s|__PYTHON_BIN__|${PYTHON_BIN}|g" \
    "${DEPLOY_DIR}/smart-speaker.service" > "${SYSTEMD_DIR}/smart-speaker.service"

chmod 644 "${SYSTEMD_DIR}/searxng.service" "${SYSTEMD_DIR}/smart-speaker.service"

info "reloading systemd"
systemctl daemon-reload
systemctl enable docker.service >/dev/null 2>&1 || true
systemctl enable searxng.service smart-speaker.service

if [ "${START_SERVICES}" = "true" ]; then
    info "starting services"
    systemctl restart searxng.service
    systemctl restart smart-speaker.service
fi

cat <<EOF

Installed. Both services now start automatically on boot.

  status : sudo systemctl status smart-speaker.service searxng.service
  logs   : journalctl -u smart-speaker.service -f
  stop   : sudo systemctl stop smart-speaker.service
  start  : sudo systemctl start smart-speaker.service
  remove : sudo ./deploy/uninstall.sh
EOF
