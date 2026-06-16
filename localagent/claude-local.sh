#!/usr/bin/env bash
# claude-local — backward-compatible launcher for Claude Code gateway mode.
#
# This is now a thin wrapper around agent-local. Use agent-local directly for
# other agents (codex, pi, etc.).
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(cd "$(dirname "$SELF")" && pwd)"

exec "$HERE/agent-local.sh" claude "$@"
