#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
DISPLAY_WIDTH="${DISPLAY_WIDTH:-1440}"
DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-900}"
DISPLAY_DEPTH="${DISPLAY_DEPTH:-24}"
DISPLAY_NUMBER="${DISPLAY#:}"

mkdir -p data/cache data/models data/exports logs
rm -f "/tmp/.X${DISPLAY_NUMBER}-lock" "/tmp/.X11-unix/X${DISPLAY_NUMBER}"

Xvfb "${DISPLAY}" \
  -screen 0 "${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}x${DISPLAY_DEPTH}" \
  -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &

sleep 1
fluxbox >/tmp/fluxbox.log 2>&1 &

if [[ -n "${VNC_PASSWORD:-}" ]]; then
  x11vnc -storepasswd "${VNC_PASSWORD}" /tmp/artinsight-vnc.pass >/dev/null
  x11vnc -display "${DISPLAY}" -forever -shared -localhost \
    -rfbport 5900 -rfbauth /tmp/artinsight-vnc.pass >/tmp/x11vnc.log 2>&1 &
else
  x11vnc -display "${DISPLAY}" -forever -shared -localhost \
    -rfbport 5900 -nopw >/tmp/x11vnc.log 2>&1 &
fi

websockify --web=/usr/share/novnc/ 6080 localhost:5900 >/tmp/novnc.log 2>&1 &

exec python main.py
