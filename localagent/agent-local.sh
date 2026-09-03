#!/usr/bin/env bash
# agent-local — run any agent CLI with the local proxy as its LLM gateway.
#
# The proxy exposes both Anthropic and OpenAI-compatible APIs on :8090 and routes
# local-model traffic to your llama-server. This launcher sets the environment so
# the wrapped agent talks to the proxy instead of the cloud.
#
# Usage:
#   agent-local claude                        # interactive Claude Code session
#   agent-local claude --model local-llama -p "2+2"
#   agent-local codex                         # OpenAI Codex CLI
#   agent-local pi                            # pi coding agent
#
# The first argument is the agent binary to exec; remaining args are passed through.
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(cd "$(dirname "$SELF")" && pwd)"
PORT="${LLAMA_PROXY_PORT:-8090}"
HOST="${LLAMA_PROXY_HOST:-127.0.0.1}"
PROXY_URL="http://$HOST:$PORT"

# Is the proxy host this machine? Only then does starting the proxy locally make sense.
is_local_host() {
  case "$HOST" in
    127.*|localhost|::1) return 0 ;;
  esac
  hostname -I 2>/dev/null | tr ' ' '\n' | grep -qxF "$HOST"
}

if is_local_host; then
  # make sure the proxy is up (also warns if llama-server is down)
  "$HERE/localagent.sh" proxy start >&2 || true
else
  echo "agent-local: proxy host $HOST is remote; not starting a local proxy" >&2
fi

AGENT="${1:-}"
shift || true

if [[ -z "$AGENT" ]]; then
  echo "usage: agent-local <agent-binary> [args...]" >&2
  echo "examples: agent-local claude | agent-local codex | agent-local pi" >&2
  exit 1
fi

# Common OpenAI-compatible env. Most agents honor these.
export OPENAI_BASE_URL="${PROXY_URL}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

# Agent-specific extras.
case "$AGENT" in
  claude|claude-code)
    export ANTHROPIC_BASE_URL="$PROXY_URL"
    export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
    # NB: auth is whatever you already use. Subscription OAuth is forwarded to Anthropic
    # by the proxy for Opus traffic. If Opus passthrough ever 401s, export ANTHROPIC_API_KEY.
    ;;
  codex)
    # Codex uses OPENAI_BASE_URL / OPENAI_API_KEY by default.
    ;;
  pi)
    # pi loads providers from extensions. If you want pi to use the proxy, point its
    # llamacpp extension config at ${PROXY_URL}/v1 instead of llama-server directly.
    echo "hint: configure ~/.pi/agent/llamacpp.json with {\"url\":\"${PROXY_URL}/v1\"}" >&2
    ;;
esac

exec "$AGENT" "$@"
