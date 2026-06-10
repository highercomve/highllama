#!/usr/bin/env bash
#
# install.sh — symlink highllama into your PATH
#
#   ./install.sh              # link highllama into ~/.local/bin
#   ./install.sh --uninstall  # remove the link
#   BIN_DIR=~/bin ./install.sh   # link somewhere else
#
set -euo pipefail

BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"   # absolute, works from any cwd

script="$SRC_DIR/highllama"
target="$BIN_DIR/highllama"

if [ "${1:-}" = "--uninstall" ] || [ "${1:-}" = "-u" ]; then
  if [ -L "$target" ]; then
    rm "$target"
    echo ">> removed $target"
  else
    echo ">> skip $target (not a symlink, or absent)"
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

echo ">> done. try: highllama -h"
