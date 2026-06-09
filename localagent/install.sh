#!/usr/bin/env bash
#
# install.sh — symlink the localagent CLIs into your PATH
#
#   ./install.sh              # link localagent + claude-local into ~/.local/bin
#   ./install.sh --uninstall  # remove the links
#   BIN_DIR=~/bin ./install.sh   # link somewhere else
#
set -euo pipefail

BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"   # absolute, works from any cwd

# link name -> script in this repo
LINKS="localagent:localagent.sh claude-local:claude-local.sh"

if [ "${1:-}" = "--uninstall" ] || [ "${1:-}" = "-u" ]; then
  for pair in $LINKS; do
    link="$BIN_DIR/${pair%%:*}"
    if [ -L "$link" ]; then
      rm "$link"
      echo ">> removed $link"
    else
      echo ">> skip $link (not a symlink, or absent)"
    fi
  done
  exit 0
fi

mkdir -p "$BIN_DIR"
for pair in $LINKS; do
  name="${pair%%:*}"; script="$SRC_DIR/${pair#*:}"
  if [ ! -f "$script" ]; then
    echo "!! missing $script" >&2
    exit 1
  fi
  chmod +x "$script"
  target="$BIN_DIR/$name"
  # refuse to clobber a real file the user put there; relinking ours is fine
  if [ -e "$target" ] && [ ! -L "$target" ]; then
    echo "!! $target exists and is not a symlink — remove it first" >&2
    exit 1
  fi
  ln -sf "$script" "$target"
  echo ">> $target -> $script"
done

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo ">> NOTE: $BIN_DIR is not in PATH — add: export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

echo ">> done. try: localagent --help"
