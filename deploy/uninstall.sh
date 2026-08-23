#!/usr/bin/env bash
#
# Remove the systemd units installed by deploy/install.sh.
#
# Usage (on the Raspberry Pi, from the repository root):
#   sudo ./deploy/uninstall.sh
#
# The SearXNG settings under SEARXNG_CONFIG_DIR (default /etc/searxng) are kept
# unless REMOVE_CONFIG=true is set.

set -euo pipefail

SYSTEMD_DIR="/etc/systemd/system"
SEARXNG_CONFIG_DIR="${SEARXNG_CONFIG_DIR:-/etc/searxng}"
REMOVE_CONFIG="${REMOVE_CONFIG:-false}"

info() { echo "==> $*"; }

[ "$(id -u)" -eq 0 ] || { echo "ERROR: root privileges are required" >&2; exit 1; }

for unit in smart-speaker.service searxng.service; do
    info "disabling ${unit}"
    systemctl disable --now "${unit}" >/dev/null 2>&1 || true
    rm -f "${SYSTEMD_DIR}/${unit}"
done

systemctl daemon-reload

if [ "${REMOVE_CONFIG}" = "true" ]; then
    info "removing ${SEARXNG_CONFIG_DIR}"
    rm -rf "${SEARXNG_CONFIG_DIR}"
fi

info "done"
