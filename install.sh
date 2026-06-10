#!/usr/bin/env bash
#
# install.sh — symlink highllama into your PATH (+ optional systemd user service)
#
#   ./install.sh              # link highllama into ~/.local/bin
#   ./install.sh --systemd    # also install a systemd user unit (Linux)
#   ./install.sh --uninstall  # remove the link and the user unit (if installed)
#   BIN_DIR=~/bin ./install.sh   # link somewhere else
#
set -euo pipefail

BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"   # absolute, works from any cwd

script="$SRC_DIR/highllama"
target="$BIN_DIR/highllama"

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}"
UNIT_FILE="$CONFIG_DIR/systemd/user/highllama.service"
ENV_FILE="$CONFIG_DIR/highllama/highllama.env"

have_user_systemd() {
  [ "$(uname -s)" = "Linux" ] && command -v systemctl >/dev/null 2>&1 \
    && systemctl --user show-environment >/dev/null 2>&1
}

if [ "${1:-}" = "--uninstall" ] || [ "${1:-}" = "-u" ]; then
  if [ -L "$target" ]; then
    rm "$target"
    echo ">> removed $target"
  else
    echo ">> skip $target (not a symlink, or absent)"
  fi
  if [ -f "$UNIT_FILE" ]; then
    if have_user_systemd; then
      systemctl --user disable --now highllama.service 2>/dev/null || true
    fi
    rm "$UNIT_FILE"
    have_user_systemd && systemctl --user daemon-reload || true
    echo ">> removed $UNIT_FILE"
    echo ">> kept $ENV_FILE (your config) — remove it manually if unwanted"
  fi
  exit 0
fi

if [ ! -f "$script" ]; then
  echo "!! missing $script" >&2
  exit 1
fi
chmod +x "$script"

mkdir -p "$BIN_DIR"
# refuse to clobber a real file the user put there; relinking ours is fine
if [ -e "$target" ] && [ ! -L "$target" ]; then
  echo "!! $target exists and is not a symlink — remove it first" >&2
  exit 1
fi
ln -sf "$script" "$target"
echo ">> $target -> $script"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo ">> NOTE: $BIN_DIR is not in PATH — add: export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

# ---- optional systemd user service (Linux only, opt-in via --systemd) -------
if [ "${1:-}" != "--systemd" ]; then
  echo ">> optional: ./install.sh --systemd  adds a systemd user service (journal logs)"
elif have_user_systemd; then
  # env file holds the server config (highllama reads these as env vars).
  # MODEL must be set — the interactive picker can't run under systemd.
  if [ ! -f "$ENV_FILE" ]; then
    mkdir -p "$(dirname "$ENV_FILE")"
    cat > "$ENV_FILE" <<'EOF'
# highllama service config — every var here overrides a highllama default.
# MODEL is required (no tty for the interactive picker under systemd):
#   a local path, a substring matching a local GGUF, or an HF repo:quant tag.
#MODEL=qwen3-coder
#CONTEXT=64k
#PORT=8089
#HOST=0.0.0.0
#KVTYPE=q4_0
#NCMOE=
#BACKEND=
EOF
    echo ">> wrote $ENV_FILE (edit it: set MODEL= before starting the service)"
  fi

  mkdir -p "$(dirname "$UNIT_FILE")"
  cat > "$UNIT_FILE" <<EOF
[Unit]
Description=highllama llama.cpp server
After=network.target

[Service]
Type=exec
EnvironmentFile=$ENV_FILE
ExecStart=$script
Restart=on-failure
RestartSec=5
TimeoutStopSec=15

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  echo ">> installed $UNIT_FILE"
  echo ">>   start:   systemctl --user start highllama"
  echo ">>   on boot: systemctl --user enable highllama   (+ loginctl enable-linger $USER)"
  echo ">>   logs:    journalctl --user -u highllama -f"
else
  echo "!! --systemd: no systemd user session detected — skipped the user service" >&2
fi

echo ">> done. try: highllama -h"
