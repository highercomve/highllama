#!/usr/bin/env bash
# claude-local — start a Claude Code session in GATEWAY mode.
#
# In this mode the session's traffic goes through the local router proxy (:8090):
#   - the MAIN session model (Opus) is relayed transparently to api.anthropic.com
#   - any subagent with `model: local-llama` is served by your local llama-server
#
# Your normal `claude` is completely unaffected — this is the opt-in launcher.
#
# Usage: exactly like `claude`, e.g.
#   claude-local                       # interactive Opus session, local-llama agent available
#   claude-local --model local-llama -p "2+2"   # run the MAIN loop on the local model
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(cd "$(dirname "$SELF")" && pwd)"
PORT="${LLAMA_PROXY_PORT:-8090}"

# make sure the router proxy is up (also warns if llama-server is down)
"$HERE/localagent.sh" proxy start >&2 || true

export ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT"
export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
# NB: auth is whatever you already use. Subscription OAuth is forwarded to Anthropic
# by the proxy for Opus traffic. If Opus passthrough ever 401s, export ANTHROPIC_API_KEY.

exec claude "$@"
