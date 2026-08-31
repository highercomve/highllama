#!/usr/bin/env python3
"""
proxy.py — zero-dependency translation / logging proxy.

Exposes two APIs on the same port:

1. Anthropic Messages API (POST /v1/messages, POST /v1/messages/count_tokens)
   ROUTES by model name:
     - a "local" model  -> translate Anthropic <-> OpenAI and serve from llama-server
     - anything else    -> transparently pass through to api.anthropic.com

   This lets a native Claude Code subagent run on the local model while the main
   session stays on a cloud model: point the session at this proxy (ANTHROPIC_BASE_URL)
   and give the subagent `model: local-<x>`. Cloud traffic is relayed untouched.

2. OpenAI-compatible API (POST /v1/chat/completions, GET /v1/models)
   For opencode, Codex, pi, or any other OpenAI-native client that should still be
   logged by the proxy. These requests are forwarded verbatim to llama-server and
   recorded in the dataset DB.

The `localagent` CLI uses the Anthropic path; passthrough simply never fires for
local-only use:

    ANTHROPIC_BASE_URL=http://127.0.0.1:8090  ANTHROPIC_API_KEY=local  claude --bare -p "..."

OpenAI clients can point here too:

    OPENAI_BASE_URL=http://127.0.0.1:8090/v1  OPENAI_API_KEY=local  <client>

Stdlib only. Python 3.10+.

Env:
  LLAMA_PROXY_PORT          listen port              (default 8090)
  LLAMA_PROXY_HOST          listen host              (default 127.0.0.1)
  LLAMA_BASE                llama-server base url     (default http://127.0.0.1:8089)
  LLAMA_MODEL               force upstream model id   (default: auto-detect from /v1/models)
  LOCAL_MODEL_ALIAS         extra name that routes local (default "local-llama"; any
                            model whose name starts with "local" also routes local)
  ANTHROPIC_PASSTHROUGH_BASE upstream for non-local models (default https://api.anthropic.com)
  LLAMA_DISABLE_THINKING    "1" forces enable_thinking=false (default: enabled)
  LLAMA_PROXY_LOG           append a debug log here   (default: none; stderr only)
"""

import datetime
import http.client
import io
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

def load_dotenv():
    # Load .env file from the same directory as proxy.py
    here = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(here, ".env")
    if os.path.isfile(dotenv_path):
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        os.environ.setdefault(k, v)
        except Exception as e:
            print(f"[proxy] failed to load .env: {e}", file=sys.stderr)

load_dotenv()

PORT = int(os.environ.get("LLAMA_PROXY_PORT", "8090"))
HOST = os.environ.get("LLAMA_PROXY_HOST", "127.0.0.1")
LLAMA_BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8089").rstrip("/")
FORCED_MODEL = os.environ.get("LLAMA_MODEL")
LOG_PATH = os.environ.get("LLAMA_PROXY_LOG")
# Local reasoning models (gemma-4, gpt-oss, etc.) can use thinking when the
# template supports it. We default to enabling it; set LLAMA_DISABLE_THINKING=1
# to force it off (useful for weak models that burn tokens thinking).
DISABLE_THINKING = os.environ.get("LLAMA_DISABLE_THINKING", "0") != "0"


def _inject_thinking(body):
    """Ensure local-model requests use the template's thinking setting.

    When not disabled, default enable_thinking to True so models like gemma-4
    actually reason. If the client already sent chat_template_kwargs we only
    fill in a missing enable_thinking, preserving explicit overrides.
    """
    if DISABLE_THINKING:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        return
    kwargs = body.setdefault("chat_template_kwargs", {})
    kwargs.setdefault("enable_thinking", True)

LOCAL_ALIAS = os.environ.get("LOCAL_MODEL_ALIAS", "local-llama")
ANTHROPIC_UP = os.environ.get(
    "ANTHROPIC_PASSTHROUGH_BASE", "https://api.anthropic.com"
).rstrip("/")
OPENAI_UP = os.environ.get(
    "OPENAI_PASSTHROUGH_BASE", "https://api.openai.com"
).rstrip("/")
OPENCODE_UP = os.environ.get(
    "OPENCODE_PASSTHROUGH_BASE", "https://opencode.ai/zen/go"
).rstrip("/")
OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")

# Models OpenCode Go serves ONLY over the OpenAI Responses API (/v1/responses).
# Sending them to /v1/chat/completions yields an opaque upstream 500, so the
# proxy rejects that early with an actionable message instead.
OPENCODE_RESPONSES_ONLY = {
    m.strip() for m in os.environ.get(
        "OPENCODE_RESPONSES_ONLY", "gpt-5.6-luna,grok-4.5,grok-4.6"
    ).split(",") if m.strip()
}

OPENCODE_GO_PROVIDER = {  # model id -> wire protocol
    "deepseek-v4-pro": "openai", "deepseek-v4-flash": "openai",
    "deepseek-v4-flash-vision-exp": "openai",
    "glm-5.3": "openai", "glm-5.3-flash": "openai",
    "glm-5.2": "openai", "glm-5.1": "openai", "glm-5": "openai",
    "gpt-5.6-luna": "openai",  # responses-only, see OPENCODE_RESPONSES_ONLY
    "grok-4.6": "openai", "grok-4.5": "openai",  # responses-only
    "hy4-preview": "openai", "hy3": "openai", "hy3-preview": "openai",
    "kimi-k3": "openai", "kimi-k2.7-code": "openai",
    "kimi-k2.6": "openai", "kimi-k2.5": "openai",
    "longcat-2.0": "openai",
    "mimo-v2.5": "openai", "mimo-v2.5-pro": "openai",
    "mimo-v2-pro": "openai", "mimo-v2-omni": "openai",
    "muse-spark-1.2-contributor": "openai",
    "minimax-m3": "anthropic", "minimax-m2.7": "anthropic", "minimax-m2.5": "anthropic",
    "qwen3.8-max": "anthropic", "qwen3.8-flash": "anthropic",
    "qwen3.7-max": "anthropic", "qwen3.7-plus": "anthropic",
    "qwen3.6-plus": "anthropic", "qwen3.5-plus": "anthropic",
}

_up = urlparse(LLAMA_BASE)
UP_HOST = _up.hostname
UP_PORT = _up.port or (443 if _up.scheme == "https" else 80)
UP_HTTPS = _up.scheme == "https"

_an = urlparse(ANTHROPIC_UP)
AN_HOST = _an.hostname
AN_PORT = _an.port or (443 if _an.scheme == "https" else 80)
AN_HTTPS = _an.scheme == "https"

_oa = urlparse(OPENAI_UP)
OA_HOST = _oa.hostname
OA_PORT = _oa.port or (443 if _oa.scheme == "https" else 80)
OA_HTTPS = _oa.scheme == "https"

_oc = urlparse(OPENCODE_UP)
OC_HOST = _oc.hostname
OC_PORT = _oc.port or (443 if _oc.scheme == "https" else 80)
OC_HTTPS = _oc.scheme == "https"

# hop-by-hop headers we must not forward when relaying to Anthropic
_DROP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "accept-encoding",
    "keep-alive",
    "proxy-connection",
}


def is_local_model(name):
    """Which requests get served by llama-server vs. passed through to the cloud."""
    if not name:
        return False
    n = name.lower()
    return (
        name == MODEL
        or name == FORCED_MODEL
        or name == LOCAL_ALIAS
        or n.startswith("local")
    )


def openai_conn():
    if OA_HTTPS:
        return http.client.HTTPSConnection(OA_HOST, OA_PORT, timeout=600)
    return http.client.HTTPConnection(OA_HOST, OA_PORT, timeout=600)


def opencode_conn():
    if OC_HTTPS:
        return http.client.HTTPSConnection(OC_HOST, OC_PORT, timeout=600)
    return http.client.HTTPConnection(OC_HOST, OC_PORT, timeout=600)


def get_opencode_model_and_protocol(model_name):
    if not model_name:
        return None, None
    core_name = model_name
    is_opencode = False
    if model_name.lower().startswith("opencode-go/"):
        core_name = model_name[12:]
        is_opencode = True
    elif model_name.lower().startswith("opencode-"):
        core_name = model_name[9:]
        is_opencode = True
        
    if not is_opencode:
        if model_name in OPENCODE_GO_PROVIDER:
            return model_name, OPENCODE_GO_PROVIDER[model_name]
        return None, None

    protocol = OPENCODE_GO_PROVIDER.get(core_name)
    if not protocol:
        # Fallback heuristics for new models
        cn_lower = core_name.lower()
        if "qwen" in cn_lower or "minimax" in cn_lower:
            protocol = "anthropic"
        else:
            protocol = "openai"
            
    return core_name, protocol


def _is_openai_format(body):
    """Heuristic: OpenAI chat bodies have messages whose content is typically a string
    (or array of {type:text/image_url}) and system is inside messages[]. Anthropic bodies
    have a top-level `system` and content as list of {type:text/tool_use/tool_result}."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0]
    if not isinstance(first, dict):
        return False
    content = first.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list) and content:
        # Anthropic content blocks always have a `type` key; OpenAI content parts have type too.
        # Distinguish by role names: OpenAI allows 'system' inside messages; Anthropic uses
        # 'user' | 'assistant' only at the message level.
        if first.get("role") == "system":
            return True
        # If content parts are OpenAI-style {type:'text',text:..} or {type:'image_url',..}
        # it is still an OpenAI request.
        return True
    return False


def _openai_messages_flat(messages):
    """Flatten OpenAI messages to role/content strings for dataset logging."""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        reasoning = m.get("reasoning_content") or m.get("reasoning")
        
        flat_content = ""
        if isinstance(content, str):
            flat_content = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type")
                    if t == "text":
                        parts.append(part.get("text", ""))
                    elif t == "image_url":
                        parts.append("[image]")
                    elif t == "thinking":
                        parts.append(f"\n<thinking>\n{part.get('thinking', '')}\n</thinking>\n")
                    elif t == "redacted_thinking":
                        parts.append(f"\n<redacted_thinking>\n{part.get('data', '')}\n</redacted_thinking>\n")
            flat_content = "".join(parts)
        else:
            flat_content = json.dumps(content) if content is not None else ""
            
        if reasoning:
            flat_content = f"\n<thinking>\n{reasoning}\n</thinking>\n" + flat_content
            
        out.append({"role": role, "content": flat_content})
    return out


def _build_dataset_item_openai(body, response_content):
    """Build a dataset item from an OpenAI-format request/response."""
    try:
        req_model = body.get("model", MODEL)
        messages = list(body.get("messages", []))
        tools = body.get("tools")
        system = ""
        for m in messages:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                system = m["content"]
                break

        assistant_message = {"role": "assistant", "content": response_content}
        messages.append(assistant_message)

        messages_flat = _openai_messages_flat(messages)

        conversations = []
        for m in messages:
            role = m.get("role")
            if role == "user":
                from_val = "human"
            elif role == "assistant":
                from_val = "gpt"
            elif role == "tool":
                from_val = "function"
            else:
                from_val = role
            flat = _openai_messages_flat([m])[0]
            conversations.append({"from": from_val, "value": flat["content"]})

        has_tools = False
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") in ("tool_call", "tool_calls", "tool_result"):
                        has_tools = True
                        break
            if m.get("tool_calls") or m.get("tool_call_id"):
                has_tools = True

        item = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": req_model,
            "system": system,
            "messages": messages,
            "messages_flat": messages_flat,
            "conversations": conversations,
            "has_tool_calls": has_tools,
        }
        if tools:
            item["tools"] = tools
        return item
    except Exception as e:
        log("Error building OpenAI dataset item:", e)
        return None


def anthropic_conn():
    if AN_HTTPS:
        return http.client.HTTPSConnection(AN_HOST, AN_PORT, timeout=600)
    return http.client.HTTPConnection(AN_HOST, AN_PORT, timeout=600)


def log(*a):
    msg = "[proxy %s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(msg, file=sys.stderr, flush=True)
    if LOG_PATH:
        try:
            with open(LOG_PATH, "a") as f:
                f.write(msg + "\n")
        except OSError:
            pass


def upstream_conn():
    if UP_HTTPS:
        return http.client.HTTPSConnection(UP_HOST, UP_PORT, timeout=600)
    return http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=600)


def detect_model():
    if FORCED_MODEL:
        return FORCED_MODEL
    try:
        c = upstream_conn()
        c.request("GET", "/v1/models")
        r = c.getresponse()
        data = json.loads(r.read())
        c.close()
        items = data.get("data") or data.get("models") or []
        # Prefer a non-embedding chat model; fall back to the first item.
        for item in items:
            mid = (item.get("id") or item.get("name") or "").lower()
            if "embedding" not in mid:
                return item.get("id") or item.get("name")
        if items:
            return items[0].get("id") or items[0].get("name")
    except Exception as e:  # noqa
        log("model detect failed:", e)
    return "local-model"


MODEL = detect_model()
log("upstream", LLAMA_BASE, "model", MODEL, "listening", "%s:%d" % (HOST, PORT))


# ---------------------------------------------------------------------------
# request translation: Anthropic /v1/messages body -> OpenAI chat body
# ---------------------------------------------------------------------------
def _text_from_content(content):
    """Flatten an Anthropic content value (str | list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif isinstance(b, str):
            parts.append(b)
    return "".join(parts)


def anthropic_to_openai(body):
    msgs = []

    # system: string OR list of {type:text,text} blocks (may carry cache_control)
    system = body.get("system")
    if system:
        sys_text = system if isinstance(system, str) else _text_from_content(system)
        if sys_text:
            msgs.append({"role": "system", "content": sys_text})

    for m in body.get("messages", []):
        role = m.get("role")
        content = m.get("content")

        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for b in content or []:
                t = b.get("type")
                if t == "text":
                    text_parts.append(b.get("text", ""))
                elif t == "tool_use":
                    tool_calls.append(
                        {
                            "id": b.get("id"),
                            "type": "function",
                            "function": {
                                "name": b.get("name"),
                                "arguments": json.dumps(b.get("input", {})),
                            },
                        }
                    )
                # thinking / redacted_thinking blocks are dropped
            am = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                am["tool_calls"] = tool_calls
            msgs.append(am)
            continue

        # user (or tool) message: may contain text + tool_result blocks
        leftover_text = []
        for b in content or []:
            t = b.get("type")
            if t == "tool_result":
                result = b.get("content")
                if isinstance(result, list):
                    result = _text_from_content(result)
                elif not isinstance(result, str):
                    result = json.dumps(result)
                if b.get("is_error"):
                    result = "[tool error] " + (result or "")
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id"),
                        "content": result or "",
                    }
                )
            elif t == "text":
                leftover_text.append(b.get("text", ""))
            elif t == "image":
                leftover_text.append("[image omitted]")
        if leftover_text:
            msgs.append({"role": "user", "content": "".join(leftover_text)})

    out = {
        "model": MODEL,
        "messages": msgs,
        "stream": bool(body.get("stream")),
    }
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    _inject_thinking(out)

    # tools
    tools = body.get("tools")
    if tools:
        otools = []
        for t in tools:
            if "input_schema" in t or "name" in t:  # anthropic-style tool
                otools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get(
                                "input_schema", {"type": "object", "properties": {}}
                            ),
                        },
                    }
                )
        if otools:
            out["tools"] = otools
            tc = body.get("tool_choice")
            if isinstance(tc, dict):
                ty = tc.get("type")
                if ty == "auto":
                    out["tool_choice"] = "auto"
                elif ty == "any":
                    out["tool_choice"] = "required"
                elif ty == "tool" and tc.get("name"):
                    out["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tc["name"]},
                    }
    return out


FINISH_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
    None: "end_turn",
}


def new_msg_id():
    return "msg_" + uuid.uuid4().hex[:24]


# ---------------------------------------------------------------------------
# tool-call SALVAGE: some local models (Qwen3-Coder especially) emit tool calls
# as TEXT instead of structured tool_calls when llama.cpp's parser misses them,
# e.g.  <function=Read><parameter=file_path>\n/x\n</parameter></function>
# or the Hermes JSON style  <tool_call>{"name":..,"arguments":{..}}</tool_call>.
# We detect those in the content and convert them to real tool_use blocks.
# ---------------------------------------------------------------------------
_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
_TOOLCALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# a partial marker possibly straddling the end of the buffered text
_PARTIAL_MARKER = re.compile(
    r"<(?:function|tool_call|parameter)\b[^>]*$|<[/a-z_]*$", re.IGNORECASE
)


def _coerce(v):
    """Best-effort: turn a parameter string into a JSON scalar/obj when it clearly is
    one (numbers, bools, arrays, objects), otherwise keep it as a plain string."""
    s = v.strip()
    if s and (
        s[0] in "{["
        or s in ("true", "false", "null")
        or re.fullmatch(r"-?\d+(\.\d+)?", s)
    ):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            pass
    return s


def salvage_tool_calls(text):
    """Return (clean_text, [tool_use_dict, ...]) extracting any leaked tool-call syntax.
    If nothing is found, the tool list is empty and clean_text == text."""
    if not text or ("<function=" not in text and "<tool_call>" not in text):
        return text, []
    calls = []

    def _json_sub(m):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            return m.group(0)
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not name:
            return m.group(0)
        calls.append(
            {
                "type": "tool_use",
                "id": "toolu_" + uuid.uuid4().hex[:24],
                "name": name,
                "input": args or {},
            }
        )
        return ""

    def _func_sub(m):
        name = m.group(1).strip()
        args = {
            pm.group(1).strip(): _coerce(pm.group(2))
            for pm in _PARAM_RE.finditer(m.group(2))
        }
        calls.append(
            {
                "type": "tool_use",
                "id": "toolu_" + uuid.uuid4().hex[:24],
                "name": name,
                "input": args,
            }
        )
        return ""

    out = _TOOLCALL_JSON_RE.sub(_json_sub, text)
    out = _FUNC_RE.sub(_func_sub, out)
    if calls:  # only scrub wrapper tags if we actually extracted something
        out = re.sub(r"</?tool_call>", "", out)
    return out.strip(), calls


# ---------------------------------------------------------------------------
# non-streaming response translation: OpenAI completion -> Anthropic message
# ---------------------------------------------------------------------------
def openai_to_anthropic(resp, req_model):
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    native_calls = msg.get("tool_calls") or []
    blocks = []
    salvaged = []
    content = msg.get("content") or ""
    if content and not native_calls:
        content, salvaged = salvage_tool_calls(content)
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in native_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments")}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:24]),
                "name": fn.get("name"),
                "input": args,
            }
        )
    blocks.extend(salvaged)
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    stop = FINISH_MAP.get(choice.get("finish_reason"), "end_turn")
    if native_calls or salvaged:
        stop = "tool_use"
    usage = resp.get("usage", {}) or {}
    return {
        "id": resp.get("id") or new_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": req_model,
        "content": blocks,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def sse(event, data):
    return ("event: %s\ndata: %s\n\n" % (event, json.dumps(data))).encode()


def estimate_tokens(body):
    """Rough char/4 estimate; good enough for context-window display."""
    n = 0
    s = body.get("system")
    if s:
        n += len(s if isinstance(s, str) else _text_from_content(s))
    for m in body.get("messages", []):
        n += len(_text_from_content(m.get("content")))
    for t in body.get("tools", []) or []:
        n += len(json.dumps(t))
    return max(1, n // 4)


# ---------------------------------------------------------------------------
# Dataset logging helper
# ---------------------------------------------------------------------------
_DATASET_QUEUE_MAX = int(os.environ.get("LLAMA_PROXY_DATASET_QUEUE_MAX", "10000"))
_dataset_queue = queue.Queue(maxsize=_DATASET_QUEUE_MAX)
_dataset_dropped = 0
_dataset_dropped_lock = threading.Lock()


def _enqueue_dataset(task):
    """Non-blocking enqueue. If the writer falls behind, drop the oldest
    pending task and count the drop instead of buffering unbounded memory."""
    global _dataset_dropped
    while True:
        try:
            _dataset_queue.put_nowait(task)
            return
        except queue.Full:
            try:
                _dataset_queue.get_nowait()
            except queue.Empty:
                pass
            with _dataset_dropped_lock:
                _dataset_dropped += 1


def _flatten_content(content):
    if isinstance(content, str):
        return content
    if not content:
        return ""
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict):
            t = b.get("type")
            if t == "text":
                parts.append(b.get("text", ""))
            elif t == "tool_use":
                call_obj = {
                    "name": b.get("name"),
                    "arguments": b.get("input", {})
                }
                parts.append(f"\n<tool_call>\n{json.dumps(call_obj)}\n</tool_call>\n")
            elif t == "tool_result":
                result = b.get("content")
                if isinstance(result, list):
                    result_str = _flatten_content(result)
                elif isinstance(result, str):
                    result_str = result
                else:
                    result_str = json.dumps(result)
                parts.append(f"\n<tool_response>\n{result_str}\n</tool_response>\n")
            elif t == "thinking":
                parts.append(f"\n<thinking>\n{b.get('thinking', '')}\n</thinking>\n")
            elif t == "redacted_thinking":
                parts.append(f"\n<redacted_thinking>\n{b.get('data', '')}\n</redacted_thinking>\n")
            elif t == "image":
                parts.append("[image]")
    return "".join(parts)


def _has_tool_calls(item):
    """True if any message in the conversation referenced a tool."""
    for m in item.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


def _build_dataset_item(body, response_content):
    try:
        req_model = body.get("model", MODEL)
        system = body.get("system")
        tools = body.get("tools")
        messages = list(body.get("messages", []))

        assistant_message = {"role": "assistant", "content": response_content}
        messages.append(assistant_message)

        system_str = ""
        if system:
            system_str = system if isinstance(system, str) else _flatten_content(system)

        messages_flat = []
        if system_str:
            messages_flat.append({"role": "system", "content": system_str})
        for msg in messages:
            messages_flat.append({
                "role": msg.get("role"),
                "content": _flatten_content(msg.get("content"))
            })

        conversations = []
        if system_str:
            conversations.append({"from": "system", "value": system_str})
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                from_val = "human"
            elif role == "assistant":
                from_val = "gpt"
            elif role == "tool":
                from_val = "function"
            else:
                from_val = role
            conversations.append({
                "from": from_val,
                "value": _flatten_content(msg.get("content"))
            })

        item = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": req_model,
            "system": system_str,
            "messages": messages,
            "messages_flat": messages_flat,
            "conversations": conversations,
            "has_tool_calls": _has_tool_calls(
                {"messages": messages}
            ),
        }
        if tools:
            item["tools"] = tools
        return item
    except Exception as e:
        log("Error building dataset item:", e)
        return None


def resolve_dataset_paths():
    """Resolve (jsonl_path, db_path) using the same precedence as the writer.
    Exposed so external tools (e.g. localagent.sh info) can find the DB."""
    dataset_path = os.environ.get("LLAMA_PROXY_DATASET")
    if not dataset_path:
        if LOG_PATH:
            log_dir = os.path.dirname(LOG_PATH)
            dataset_path = os.path.join(log_dir, "dataset.jsonl")
        else:
            dataset_path = os.path.expanduser("~/.local/state/localagent/dataset.jsonl")
    db_path = dataset_path.rsplit(".", 1)[0] + ".db"
    return dataset_path, db_path


_DATASET_SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    system TEXT,
    messages TEXT NOT NULL,
    messages_flat TEXT NOT NULL,
    conversations TEXT NOT NULL,
    tools TEXT,
    has_tool_calls INTEGER NOT NULL DEFAULT 0
)
"""


def _ensure_schema(conn):
    cur = conn.cursor()
    cur.execute(_DATASET_SCHEMA)
    cur.execute("PRAGMA user_version")
    ver = cur.fetchone()[0]
    if ver < 1:
        # Add has_tool_calls to pre-existing tables that predate the column
        try:
            cur.execute("ALTER TABLE dataset_calls ADD COLUMN has_tool_calls INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        cur.execute("PRAGMA user_version = 1")
        conn.commit()


def _migrate_jsonl(conn, jsonl_path):
    """One-shot: import a legacy dataset.jsonl into the SQLite DB.
    Idempotent — only runs if the DB has zero rows AND the file exists.
    Runs in the writer thread so proxy startup isn't blocked."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM dataset_calls")
    if cur.fetchone()[0] > 0:
        return
    if not os.path.exists(jsonl_path):
        return
    try:
        size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
    except OSError:
        return
    if size_mb > 256:
        log(f"Database migration: skipping {size_mb:.0f}MB JSONL (set LLAMA_PROXY_DATASET to a smaller file or import manually)")
        return
    log(f"Database migration: importing {size_mb:.1f}MB JSONL into SQLite...")
    count = 0
    batch = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    log("Migration: skipping malformed line:", e)
                    continue
                has_tools = 0
                for m in item.get("messages", []) or []:
                    content = m.get("content")
                    if isinstance(content, list):
                        for b in content:
                            if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"):
                                has_tools = 1
                                break
                    if has_tools:
                        break
                cur.execute(
                    "INSERT INTO dataset_calls (timestamp, model, system, messages, messages_flat, conversations, tools, has_tool_calls) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item.get("timestamp"),
                        item.get("model", ""),
                        item.get("system", ""),
                        json.dumps(item.get("messages", [])),
                        json.dumps(item.get("messages_flat", [])),
                        json.dumps(item.get("conversations", [])),
                        json.dumps(item["tools"]) if "tools" in item else None,
                        has_tools,
                    ),
                )
                count += 1
                batch += 1
                if batch >= 500:
                    conn.commit()
                    batch = 0
        conn.commit()
        log(f"Database migration: imported {count} entries from JSONL")
    except Exception as e:
        log("Database migration failed:", e, traceback.format_exc())


def _maybe_rotate(db_path):
    """If LLAMA_PROXY_DATASET_ROTATE_MB is set and the DB exceeds that size,
    rename it aside and start a fresh one. Best-effort: rename errors are logged."""
    rotate_mb = os.environ.get("LLAMA_PROXY_DATASET_ROTATE_MB")
    if not rotate_mb:
        return
    try:
        limit = int(rotate_mb) * 1024 * 1024
    except ValueError:
        log("LLAMA_PROXY_DATASET_ROTATE_MB is not an int; ignoring")
        return
    try:
        size = os.path.getsize(db_path)
    except OSError:
        return
    if size < limit:
        return
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rotated = db_path.rsplit(".", 1)[0] + f".{ts}.db"
    try:
        os.rename(db_path, rotated)
        log(f"Dataset rotation: {db_path} ({size // (1024*1024)}MB) -> {rotated}")
    except OSError as e:
        log(f"Dataset rotation: rename failed: {e}")


_INSERT_SQL = (
    "INSERT INTO dataset_calls "
    "(timestamp, model, system, messages, messages_flat, conversations, tools, has_tool_calls) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


def _dataset_writer_thread():
    global _dataset_dropped
    dataset_path, db_path = resolve_dataset_paths()
    try:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    except Exception:
        pass

    _maybe_rotate(db_path)

    import sqlite3
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    _ensure_schema(conn)
    _migrate_jsonl(conn, dataset_path)

    write_count = 0
    while True:
        try:
            task = _dataset_queue.get()
            if task is None:
                break

            task_type = task.get("type")
            item = None
            if task_type == "parsed":
                item = _build_dataset_item(task["body"], task["response_content"])
            elif task_type == "passthrough_sync":
                try:
                    body = json.loads(task["body_raw"])
                    resp_obj = json.loads(task["response_bytes"])
                    response_content = resp_obj.get("content", [])
                    if response_content:
                        item = _build_dataset_item(body, response_content)
                except Exception as e:
                    log("Dataset writer: failed to parse passthrough_sync:", e, traceback.format_exc())
            elif task_type == "passthrough_stream":
                try:
                    body = json.loads(task["body_raw"])
                    reconstructor = SSEAssistantResponseReconstructor()
                    raw = task.get("raw_bytes", b"")
                    for line_bytes in raw.splitlines(keepends=True):
                        reconstructor.feed_line(line_bytes)
                    response_content = reconstructor.get_content()
                    if response_content:
                        item = _build_dataset_item(body, response_content)
                    else:
                        log("Dataset writer: reconstructor returned empty content for stream")
                except Exception as e:
                    log("Dataset writer: failed to parse passthrough_stream:", e, traceback.format_exc())
            elif task_type == "openai_parsed":
                if "response_content" in task:
                    item = _build_dataset_item_openai(task["body"], task["response_content"])
                else:
                    try:
                        resp_obj = json.loads(task["response_bytes"])
                        choice = (resp_obj.get("choices") or [{}])[0]
                        msg = choice.get("message", {})
                        content = msg.get("content") or ""
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                        tool_calls = msg.get("tool_calls") or []
                        response_content = []
                        if reasoning:
                            response_content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
                        if content:
                            response_content.append({"type": "text", "text": content})
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                            except Exception:
                                args = {"_raw": fn.get("arguments")}
                            response_content.append({
                                "type": "tool_use",
                                "id": tc.get("id"),
                                "name": fn.get("name"),
                                "input": args,
                            })
                        if not response_content:
                            response_content.append({"type": "text", "text": ""})
                        item = _build_dataset_item_openai(task["body"], response_content)
                    except Exception as e:
                        log("Dataset writer: failed to parse openai_parsed:", e, traceback.format_exc())
            elif task_type == "openai_stream":
                try:
                    body = task["body"]
                    content_parts = []
                    reasoning_parts = []
                    tool_calls = {}  # index -> {id,name,args}
                    raw = task.get("raw_bytes", b"")
                    for line_bytes in raw.splitlines(keepends=True):
                        line = line_bytes.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta", {}) or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        if delta.get("reasoning_content"):
                            reasoning_parts.append(delta["reasoning_content"])
                        elif delta.get("reasoning"):
                            reasoning_parts.append(delta["reasoning"])
                        for tc in delta.get("tool_calls", []) or []:
                            idx = tc.get("index", 0)
                            slot = tool_calls.setdefault(idx, {"id": None, "name": "", "args": ""})
                            fn = tc.get("function", {}) or {}
                            if fn.get("id"):
                                slot["id"] = fn["id"]
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["args"] += fn["arguments"]
                    content = "".join(content_parts)
                    reasoning = "".join(reasoning_parts)
                    response_content = []
                    if reasoning:
                        response_content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
                    if content:
                        response_content.append({"type": "text", "text": content})
                    for idx in sorted(tool_calls.keys()):
                        t = tool_calls[idx]
                        try:
                            args = json.loads(t["args"] or "{}")
                        except Exception:
                            args = {"_raw": t["args"]}
                        response_content.append({
                            "type": "tool_use",
                            "id": t["id"] or ("toolu_" + uuid.uuid4().hex[:24]),
                            "name": t["name"],
                            "input": args,
                        })
                    if not response_content:
                        response_content.append({"type": "text", "text": ""})
                    item = _build_dataset_item_openai(body, response_content)
                except Exception as e:
                    log("Dataset writer: failed to parse openai_stream:", e, traceback.format_exc())

            if not item:
                continue

            tools_json = json.dumps(item["tools"]) if item.get("tools") else None
            conn.execute(
                _INSERT_SQL,
                (
                    item["timestamp"],
                    item["model"],
                    item.get("system", ""),
                    json.dumps(item["messages"]),
                    json.dumps(item["messages_flat"]),
                    json.dumps(item["conversations"]),
                    tools_json,
                    int(bool(item.get("has_tool_calls"))),
                ),
            )
            conn.commit()  # commit per row: WAL keeps the cost low, and external readers (exporter, test harnesses) see writes immediately
            write_count += 1
            if _dataset_dropped:
                with _dataset_dropped_lock:
                    dropped = _dataset_dropped
                    _dataset_dropped = 0
                if dropped:
                    log(f"Dataset writer: dropped {dropped} tasks due to backlog")
        except Exception as e:
            log("Dataset writer thread loop error:", e, traceback.format_exc())

    try:
        conn.commit()
        conn.close()
    except Exception:
        pass


def start_dataset_writer():
    """Spawn the background writer thread. Called by main(); safe to call once."""
    global _writer
    if _writer is not None and _writer.is_alive():
        return _writer
    _writer = threading.Thread(target=_dataset_writer_thread, daemon=True)
    _writer.start()
    return _writer


_writer = None


class SSEAssistantResponseReconstructor:
    _ERRORS_LOGGED = 0
    _ERRORS_LOGGED_MAX = 5

    def __init__(self):
        self.blocks = {}

    def feed_line(self, line_bytes):
        try:
            line = line_bytes.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                return
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                return
            data = json.loads(payload)
            t = data.get("type")
            if t == "content_block_start":
                idx = data.get("index")
                cb = data.get("content_block", {})
                cb_type = cb.get("type")
                if cb_type == "text":
                    self.blocks[idx] = {"type": "text", "text": cb.get("text", "")}
                elif cb_type == "thinking":
                    self.blocks[idx] = {
                        "type": "thinking",
                        "thinking": cb.get("thinking", ""),
                        "signature": cb.get("signature", ""),
                    }
                elif cb_type == "redacted_thinking":
                    self.blocks[idx] = {
                        "type": "redacted_thinking",
                        "data": cb.get("data", ""),
                    }
                elif cb_type == "tool_use":
                    self.blocks[idx] = {
                        "type": "tool_use",
                        "id": cb.get("id"),
                        "name": cb.get("name"),
                        "input_str": "",
                    }
            elif t == "content_block_delta":
                idx = data.get("index")
                delta = data.get("delta", {})
                dt = delta.get("type")
                if idx in self.blocks:
                    if dt == "text_delta":
                        self.blocks[idx]["text"] += delta.get("text", "")
                    elif dt == "thinking_delta":
                        self.blocks[idx]["thinking"] += delta.get("thinking", "")
                    elif dt == "signature_delta":
                        self.blocks[idx]["signature"] += delta.get("signature", "")
                    elif dt == "redacted_thinking_delta":
                        self.blocks[idx]["data"] += delta.get("data", "")
                    elif dt == "input_json_delta":
                        self.blocks[idx]["input_str"] += delta.get("partial_json", "")
        except Exception as e:
            if SSEAssistantResponseReconstructor._ERRORS_LOGGED < SSEAssistantResponseReconstructor._ERRORS_LOGGED_MAX:
                SSEAssistantResponseReconstructor._ERRORS_LOGGED += 1
                log("SSEAssistantResponseReconstructor: feed_line error:", e)

    def get_content(self):
        content_list = []
        for idx in sorted(self.blocks.keys()):
            block = self.blocks[idx]
            if block["type"] == "text":
                content_list.append({"type": "text", "text": block["text"]})
            elif block["type"] == "thinking":
                content_list.append(
                    {
                        "type": "thinking",
                        "thinking": block["thinking"],
                        "signature": block["signature"],
                    }
                )
            elif block["type"] == "redacted_thinking":
                content_list.append(
                    {
                        "type": "redacted_thinking",
                        "data": block["data"],
                    }
                )
            elif block["type"] == "tool_use":
                try:
                    inp = json.loads(block["input_str"])
                except Exception:
                    inp = block["input_str"]
                content_list.append(
                    {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": inp,
                    }
                )
        return content_list


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence default logging
        pass

    def _read_raw(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _passthrough(self, raw):
        """Relay an Anthropic-format request verbatim to api.anthropic.com and stream
        the response straight back. Used for every non-local model (e.g. Opus)."""
        log("Proxy: passthrough %s %s" % (self.command, self.path))
        if (self.headers.get("Upgrade") or "").lower() == "websocket":
            log("Proxy: WARNING WebSocket upgrade not supported, relaying as plain HTTP")

        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _DROP_HEADERS
        }
        headers["Host"] = AN_HOST
        if raw:
            headers["Content-Length"] = str(len(raw))

        try:
            body = json.loads(raw)
            is_stream = body.get("stream", False)
        except Exception:
            body = None
            is_stream = False

        try:
            c = anthropic_conn()
            c.request(self.command, self.path, body=raw or None, headers=headers)
            r = c.getresponse()
        except Exception as e:  # noqa
            log("passthrough error:", e)
            return self._json(
                502,
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )

        self.send_response(r.status)
        ct = r.getheader("Content-Type", "application/json")
        # Forward upstream response headers (request-id, anthropic-ratelimit-*,
        # retry-after, location, ...) so /rc clients can correlate and back off.
        # Hop-by-hop headers (incl. content-length/transfer-encoding) are dropped
        # because we delimit the relayed response by connection close instead.
        for k, v in r.getheaders():
            if k.lower() in _DROP_HEADERS or k.lower() == "content-type":
                continue
            self.send_header(k, v)
        self.send_header("Content-Type", ct)
        # delimit by connection close so we don't have to re-chunk streamed SSE
        self.send_header("Connection", "close")
        self.end_headers()

        is_sse_response = "text/event-stream" in (ct or "").lower()
        path_only = self.path.split("?", 1)[0]
        capture_response = (
            r.status == 200 and body is not None
            and path_only.startswith("/v1/messages")
        )

        if is_stream or is_sse_response:
            response_buf = io.BytesIO()
            try:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
                    if capture_response:
                        response_buf.write(line)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError as e:
                # e.g. TimeoutError on an idle upstream SSE socket: tear down
                # quietly instead of letting socketserver dump a traceback.
                log("Proxy: passthrough stream aborted:", e)
            finally:
                c.close()

            if capture_response:
                log("Proxy: queuing streaming passthrough task...")
                _enqueue_dataset(
                    {
                        "type": "passthrough_stream",
                        "body_raw": raw,
                        "raw_bytes": response_buf.getvalue(),
                    }
                )
            else:
                log("Proxy: skipping logging for streaming passthrough (capture_response=%s, raw_bytes=%d)" % (capture_response, len(response_buf.getvalue())))
        else:
            response_bytes = b""
            try:
                response_bytes = r.read()
                self.wfile.write(response_bytes)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError as e:
                log("Proxy: passthrough read aborted:", e)
            finally:
                c.close()

            if capture_response and response_bytes:
                log("Proxy: queuing sync passthrough task...")
                _enqueue_dataset(
                    {
                        "type": "passthrough_sync",
                        "body_raw": raw,
                        "response_bytes": response_bytes,
                    }
                )

    def _opencode_anthropic_passthrough(self, raw, core_model):
        """Relay an Anthropic-format request to OpenCode Go endpoint using its Anthropic API."""
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _DROP_HEADERS
        }
        headers["Host"] = OC_HOST
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        if OPENCODE_API_KEY:
            headers["x-api-key"] = OPENCODE_API_KEY
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"

        try:
            body = json.loads(raw)
            body["model"] = core_model
            payload = json.dumps(body).encode("utf-8")
        except Exception as e:
            return self._json(
                400,
                {"type": "error", "error": {"type": "invalid_request_error", "message": f"Malformed JSON: {e}"}},
            )

        headers["Content-Length"] = str(len(payload))
        is_stream = body.get("stream", False)

        try:
            c = opencode_conn()
            req_path = _oc.path.rstrip('/') + "/v1/messages"
            c.request(self.command, req_path, body=payload, headers=headers)
            r = c.getresponse()
        except Exception as e:
            log("opencode anthropic passthrough error:", e)
            return self._json(
                502,
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )

        self.send_response(r.status)
        ct = r.getheader("Content-Type", "application/json")
        self.send_header("Content-Type", ct)
        self.send_header("Connection", "close")
        self.end_headers()

        capture_response = r.status == 200

        if is_stream:
            response_buf = io.BytesIO()
            try:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
                    if capture_response:
                        response_buf.write(line)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                c.close()

            if capture_response:
                log("Proxy: queuing streaming opencode anthropic task...")
                body_original = json.loads(raw)
                _enqueue_dataset(
                    {
                        "type": "passthrough_stream",
                        "body_raw": json.dumps(body_original),
                        "raw_bytes": response_buf.getvalue(),
                    }
                )
        else:
            response_bytes = b""
            try:
                response_bytes = r.read()
                self.wfile.write(response_bytes)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                c.close()

            if capture_response and response_bytes:
                log("Proxy: queuing sync opencode anthropic task...")
                body_original = json.loads(raw)
                _enqueue_dataset(
                    {
                        "type": "passthrough_sync",
                        "body_raw": json.dumps(body_original),
                        "response_bytes": response_bytes,
                    }
                )

    def _opencode_openai_messages(self, body, core_model):
        """Translate Anthropic /v1/messages -> OpenAI /v1/chat/completions for OpenCode Go."""
        if core_model in OPENCODE_RESPONSES_ONLY:
            log("opencode: %s is Responses-only; rejecting /v1/messages" % core_model)
            return self._json(
                400,
                {"type": "error",
                 "error": {
                     "type": "invalid_request_error",
                     "message": (
                         "%s is only served by OpenCode Go over the OpenAI Responses API. "
                         "Send it to /v1/responses (pi: api \"openai-responses\")." % core_model
                     ),
                 }},
            )
        oai = anthropic_to_openai(body)
        oai["model"] = core_model
        if "chat_template_kwargs" in oai:
            oai.pop("chat_template_kwargs")
        stream = oai["stream"]
        payload = json.dumps(oai).encode("utf-8")

        req_model = body.get("model")
        log(
            "→ /v1/messages (opencode-openai)",
            "stream" if stream else "sync",
            "model=%s" % req_model,
            "msgs=%d" % len(oai["messages"]),
        )

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        if OPENCODE_API_KEY:
            headers["Authorization"] = f"Bearer {OPENCODE_API_KEY}"

        try:
            c = opencode_conn()
            req_path = _oc.path.rstrip('/') + "/v1/chat/completions"
            c.request(
                "POST",
                req_path,
                body=payload,
                headers=headers,
            )
            r = c.getresponse()
        except Exception as e:
            log("opencode upstream error:", e)
            return self._json(
                502,
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )

        if r.status != 200:
            err = r.read()
            c.close()
            log("opencode upstream status", r.status, err[:300])
            return self._json(
                r.status,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": err.decode("utf-8", "replace"),
                    },
                },
            )

        if not stream:
            data = json.loads(r.read())
            c.close()
            anth = openai_to_anthropic(data, req_model)
            _enqueue_dataset({
                "type": "parsed",
                "body": body,
                "response_content": anth["content"],
            })
            return self._json(200, anth)

        # Streaming: translate OpenAI SSE -> Anthropic SSE
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._stream(r, req_model, body)
            self.wfile.write(b"0\r\n\r\n")  # final chunk
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected mid-stream")
        finally:
            c.close()

    def _opencode_openai_chat_completions(self, body, raw, core_model):
        """Relay an OpenAI completions request directly to OpenCode Go completions endpoint."""
        if core_model in OPENCODE_RESPONSES_ONLY:
            log("opencode: %s is Responses-only; rejecting /v1/chat/completions" % core_model)
            return self._json(
                400,
                {"error": {
                    "type": "invalid_request_error",
                    "message": (
                        "%s is only served by OpenCode Go over the OpenAI Responses API. "
                        "Send it to /v1/responses (pi: api \"openai-responses\")." % core_model
                    ),
                }},
            )
        body_original = json.loads(raw)
        body["model"] = core_model
        payload = json.dumps(body).encode("utf-8")
        is_stream = body.get("stream", False)

        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _DROP_HEADERS
        }
        headers["Host"] = OC_HOST
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        headers["Content-Length"] = str(len(payload))
        if OPENCODE_API_KEY:
            headers["Authorization"] = f"Bearer {OPENCODE_API_KEY}"

        try:
            c = opencode_conn()
            req_path = _oc.path.rstrip('/') + "/v1/chat/completions"
            c.request("POST", req_path, body=payload, headers=headers)
            r = c.getresponse()
        except Exception as e:
            log("opencode openai completions error:", e)
            return self._json(
                502,
                {"error": {"message": str(e), "type": "api_error"}},
            )

        self.send_response(r.status)
        ct = r.getheader("Content-Type", "application/json")
        self.send_header("Content-Type", ct)
        self.send_header("Connection", "close")
        self.end_headers()

        capture_response = r.status == 200

        if r.status != 200:
            # Error bodies are plain JSON even when the client asked for a
            # stream; pushing them through the SSE relay would append a
            # synthesized finish_reason chunk to the error payload.
            try:
                err = r.read()
                log("opencode upstream status", r.status, err[:300])
                self.wfile.write(err)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                c.close()
            return

        if is_stream:
            response_buf = io.BytesIO()
            saw_finish_reason = False
            saw_tool_calls = False
            terminal_emitted = False
            stream_metadata = {}

            def relay(line):
                self.wfile.write(line)
                self.wfile.flush()
                if capture_response:
                    response_buf.write(line)

            def emit_terminal():
                nonlocal terminal_emitted
                if saw_finish_reason or terminal_emitted:
                    return
                finish_reason = "tool_calls" if saw_tool_calls else "stop"
                terminal = dict(stream_metadata)
                terminal["choices"] = [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }]
                relay(
                    b"data: "
                    + json.dumps(terminal, separators=(",", ":")).encode("utf-8")
                    + b"\n\n"
                )
                terminal_emitted = True
                log(
                    "OpenCode stream ended without finish_reason; synthesized %s"
                    % finish_reason
                )

            try:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    if stripped.startswith(b"data:"):
                        payload = stripped[5:].strip()
                        if payload == b"[DONE]":
                            emit_terminal()
                        else:
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                chunk = None
                            if isinstance(chunk, dict):
                                for key in (
                                    "id", "object", "created", "model",
                                    "system_fingerprint", "service_tier",
                                ):
                                    if chunk.get(key) not in (None, ""):
                                        stream_metadata[key] = chunk[key]
                                for choice in chunk.get("choices") or []:
                                    if choice.get("finish_reason") is not None:
                                        saw_finish_reason = True
                                    if (choice.get("delta") or {}).get("tool_calls"):
                                        saw_tool_calls = True
                    relay(line)
                emit_terminal()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                c.close()

            if capture_response:
                log("Proxy: queuing opencode OpenAI streaming task...")
                _enqueue_dataset({
                    "type": "openai_stream",
                    "body": body_original,
                    "raw_bytes": response_buf.getvalue(),
                })
        else:
            response_bytes = b""
            try:
                response_bytes = r.read()
                self.wfile.write(response_bytes)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                c.close()

            if capture_response and response_bytes:
                log("Proxy: queuing opencode OpenAI sync task...")
                _enqueue_dataset({
                    "type": "openai_parsed",
                    "body": body_original,
                    "response_bytes": response_bytes,
                })
            else:
                log("Proxy: skipping logging for sync passthrough (capture_response=%s, has_bytes=%s)" % (capture_response, response_bytes is not None))

    def _opencode_openai_responses(self, raw, core_model):
        """Relay an OpenAI Responses-API request verbatim to OpenCode Go's
        /v1/responses. Some models (gpt-5.6-luna) are only served over the
        Responses endpoint; pass through with the model id normalized and let
        the upstream SSE stream straight back (no reconstruction needed — the
        Responses protocol carries its own response.completed event)."""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        body["model"] = core_model
        payload = json.dumps(body).encode("utf-8")
        log("→ /v1/responses (opencode) model=%s stream=%s keys=%s" % (
            core_model, body.get("stream"), sorted(body.keys())))

        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _DROP_HEADERS
        }
        headers["Host"] = OC_HOST
        headers["Content-Length"] = str(len(payload))
        if OPENCODE_API_KEY:
            headers["Authorization"] = f"Bearer {OPENCODE_API_KEY}"

        try:
            c = opencode_conn()
            req_path = _oc.path.rstrip('/') + "/v1/responses"
            c.request("POST", req_path, body=payload, headers=headers)
            r = c.getresponse()
            log("opencode responses upstream status", r.status)
        except Exception as e:
            log("opencode responses upstream error:", e)
            return self._json(
                502,
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )

        self.send_response(r.status)
        ct = r.getheader("Content-Type", "application/json")
        for k, v in r.getheaders():
            if k.lower() in _DROP_HEADERS or k.lower() == "content-type":
                continue
            self.send_header(k, v)
        self.send_header("Content-Type", ct)
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                line = r.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as e:
            log("opencode responses stream aborted:", e)
        finally:
            c.close()
            log("opencode responses relay done")

    def _wants_anthropic_format(self):
        """Guess whether the client expects Anthropic-style /v1/models or OpenAI-style.
        Default to Anthropic to preserve existing Claude Code gateway behavior."""
        ua = (self.headers.get("User-Agent") or "").lower()
        if "openai" in ua or "opencode" in ua or "codex" in ua:
            return False
        if self.headers.get("OpenAI-Organization") or self.headers.get("OpenAI-Project"):
            return False
        if self.headers.get("anthropic-version"):
            return True
        if self.headers.get("x-api-key"):
            return True
        # Fallback: preserve the original Anthropic format for clients we can't identify.
        return True

    def do_GET(self):
        if self.path == "/health":
            return self._json(
                200,
                {
                    "ok": True,
                    "model": MODEL,
                    "alias": LOCAL_ALIAS,
                    "passthrough": ANTHROPIC_UP,
                },
            )
        # Only the exact list endpoint gets the merged local+upstream response;
        # GET /v1/models/{model_id} (Anthropic "Get a Model") relays verbatim below.
        if self.path.split("?", 1)[0] == "/v1/models":
            is_anthropic = self._wants_anthropic_format()
            # Fetch upstream models so the proxy can be used as a full gateway.
            upstream_data = []
            if is_anthropic:
                try:
                    headers = {
                        k: v
                        for k, v in self.headers.items()
                        if k.lower() not in _DROP_HEADERS
                    }
                    headers["Host"] = AN_HOST
                    c = anthropic_conn()
                    c.request("GET", self.path, headers=headers)
                    r = c.getresponse()
                    payload = r.read()
                    c.close()
                    if r.status == 200:
                        upstream_data = json.loads(payload).get("data", [])
                except Exception as e:  # noqa
                    log("models passthrough failed:", e)
            else:
                try:
                    headers = {
                        k: v
                        for k, v in self.headers.items()
                        if k.lower() not in _DROP_HEADERS
                    }
                    headers["Host"] = OA_HOST
                    c = openai_conn()
                    c.request("GET", self.path, headers=headers)
                    r = c.getresponse()
                    payload = r.read()
                    c.close()
                    if r.status == 200:
                        upstream_data = json.loads(payload).get("data", [])
                except Exception as e:  # noqa
                    log("OpenAI models passthrough failed:", e)

            # Local model entry: include BOTH OpenAI and Anthropic fields so the
            # same response works for clients expecting either format.
            local_entry = {
                "id": LOCAL_ALIAS,
                "object": "model",
                "type": "model",
                "display_name": "Local (%s)" % MODEL,
                "created": 0,
                "created_at": "",
                "owned_by": "llamacpp",
            }

            opencode_entries = []
            for model_id in sorted(OPENCODE_GO_PROVIDER.keys()):
                opencode_entries.append({
                    "id": f"opencode-go/{model_id}",
                    "object": "model",
                    "type": "model",
                    "display_name": f"OpenCode Go - {model_id}",
                    "created": 0,
                    "created_at": "",
                    "owned_by": "opencode",
                })

            data = [local_entry] + opencode_entries + upstream_data

            return self._json(
                200,
                {
                    "object": "list",
                    "data": data,
                    "has_more": False,
                    "first_id": LOCAL_ALIAS,
                    "last_id": data[-1]["id"] if data else LOCAL_ALIAS,
                },
            )
        if self.path.startswith("/v1/") or self.path.startswith("/api/"):
            return self._passthrough(self._read_raw())
        self._json(404, {"error": "not found"})

    def _route_other_method(self):
        """Relay any Anthropic-API-surface request (required for Claude Code /rc)."""
        path_only = self.path.split("?", 1)[0]
        if path_only.startswith("/v1/") or path_only.startswith("/api/"):
            return self._passthrough(self._read_raw())
        return self._json(404, {"error": "not found"})

    def do_DELETE(self):
        return self._route_other_method()

    def do_PUT(self):
        return self._route_other_method()

    def do_PATCH(self):
        return self._route_other_method()

    def do_OPTIONS(self):
        return self._route_other_method()

    def do_HEAD(self):
        return self._route_other_method()

    def do_POST(self):
        raw = self._read_raw()
        path_only = self.path.split("?", 1)[0]
        try:
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                # JSON that isn't an object (list/string/number/null) cannot carry
                # a "model" key — treat it exactly like a non-JSON body below.
                raise ValueError("non-object JSON body")
        except Exception:
            # Non-JSON (or non-object JSON) body: cannot be a local-model request.
            # Relay verbatim if it targets the Anthropic API surface; otherwise
            # reject as before.
            if path_only.startswith("/v1/") or path_only.startswith("/api/"):
                return self._passthrough(raw)
            return self._json(
                400,
                {"type": "error",
                 "error": {"type": "invalid_request_error", "message": "invalid JSON body"}},
            )

        req_model = body.get("model", MODEL)
        opencode_core_model, opencode_proto = get_opencode_model_and_protocol(req_model)

        # OpenAI-compatible chat completions endpoint (used by opencode, Codex, pi, etc.)
        if self.path.startswith("/v1/chat/completions"):
            if opencode_core_model:
                return self._opencode_openai_chat_completions(body, raw, opencode_core_model)
            return self._openai_chat_completions(body, raw)

        # OpenAI Responses API (used by pi for models flagged openai-responses,
        # e.g. gpt-5.6-luna which opencode only serves over /v1/responses).
        if self.path.startswith("/v1/responses"):
            if opencode_core_model:
                return self._opencode_openai_responses(raw, opencode_core_model)
            return self._openai_chat_completions(body, raw)

        # Intercept Anthropic messages requests for opencode models
        if self.path.startswith("/v1/messages") and opencode_core_model:
            if opencode_proto == "anthropic":
                return self._opencode_anthropic_passthrough(raw, opencode_core_model)
            else:
                return self._opencode_openai_messages(body, opencode_core_model)

        # ROUTER: only the exact local surface with a local model stays local;
        # everything else Anthropic-shaped relays verbatim (required for /rc).
        is_local = is_local_model(req_model)
        local_path = path_only in ("/v1/messages", "/v1/messages/count_tokens")
        if (path_only.startswith("/v1/") or path_only.startswith("/api/")) and not (
            is_local and local_path
        ):
            return self._passthrough(raw)

        if path_only.endswith("/count_tokens"):
            return self._json(200, {"input_tokens": estimate_tokens(body)})

        if not self.path.startswith("/v1/messages"):
            return self._json(404, {"error": "not found"})

        oai = anthropic_to_openai(body)
        stream = oai["stream"]
        payload = json.dumps(oai).encode()
        log(
            "→ /v1/messages",
            "stream" if stream else "sync",
            "msgs=%d" % len(oai["messages"]),
            "tools=%d" % len(oai.get("tools", [])),
        )

        try:
            c = upstream_conn()
            c.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            r = c.getresponse()
        except Exception as e:  # noqa
            log("upstream error:", e)
            return self._json(
                502,
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )

        if r.status != 200:
            err = r.read()
            c.close()
            log("upstream status", r.status, err[:300])
            return self._json(
                r.status,
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": err.decode("utf-8", "replace"),
                    },
                },
            )

        if not stream:
            data = json.loads(r.read())
            c.close()
            anth = openai_to_anthropic(data, req_model)
            _enqueue_dataset({
                "type": "parsed",
                "body": body,
                "response_content": anth["content"],
            })
            return self._json(200, anth)

        # streaming: translate OpenAI SSE -> Anthropic SSE (chunked transfer)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._stream(r, req_model, body)
            self.wfile.write(b"0\r\n\r\n")  # final chunk
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected mid-stream")
        finally:
            c.close()

    def _openai_chat_completions(self, body, raw):
        """Route OpenAI-format requests: local models -> llama-server, cloud models
        -> OpenAI-compatible passthrough. Both paths are logged to the dataset."""
        req_model = body.get("model", "")

        if not is_local_model(req_model):
            return self._openai_passthrough(raw)

        # Resolve local aliases to the actual upstream model id.
        body["model"] = MODEL
        _inject_thinking(body)
        stream = bool(body.get("stream"))
        payload = json.dumps(body).encode()
        log(
            "→ /v1/chat/completions",
            "stream" if stream else "sync",
            "model=%s (local)" % req_model,
        )

        try:
            c = upstream_conn()
            c.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            r = c.getresponse()
        except Exception as e:  # noqa
            log("upstream error:", e)
            return self._json(
                502,
                {"error": {"message": str(e), "type": "api_error"}},
            )

        if r.status != 200:
            err = r.read()
            c.close()
            log("upstream status", r.status, err[:300])
            return self._json(
                r.status,
                {"error": {"message": err.decode("utf-8", "replace"), "type": "api_error"}},
            )

        if not stream:
            data = json.loads(r.read())
            c.close()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            response_content = []
            if content:
                response_content.append({"type": "text", "text": content})
            for tc in tool_calls:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {"_raw": fn.get("arguments")}
                response_content.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "input": args,
                })
            if not response_content:
                response_content.append({"type": "text", "text": ""})
            _enqueue_dataset({
                "type": "openai_parsed",
                "body": body,
                "response_content": response_content,
            })
            return self._json(200, data)

        # Streaming: proxy OpenAI SSE straight through and log at the end.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        response_buf = io.BytesIO()
        try:
            while True:
                line = r.readline()
                if not line:
                    break
                self._w(line)
                response_buf.write(line)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            c.close()

        # Log the streamed response.
        _enqueue_dataset({
            "type": "openai_stream",
            "body": body,
            "raw_bytes": response_buf.getvalue(),
        })
        self._w(b"")
        self.wfile.flush()

    def _openai_passthrough(self, raw):
        """Relay an OpenAI-format request verbatim to the upstream OpenAI-compatible
        API and stream the response back. Used for non-local (cloud) models so all
        agent traffic is captured in the dataset."""
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _DROP_HEADERS
        }
        headers["Host"] = OA_HOST
        if raw:
            headers["Content-Length"] = str(len(raw))

        try:
            body = json.loads(raw) if raw else {}
            is_stream = body.get("stream", False)
        except Exception:
            body = None
            is_stream = False

        try:
            c = openai_conn()
            c.request("POST", "/v1/chat/completions", body=raw or None, headers=headers)
            r = c.getresponse()
        except Exception as e:  # noqa
            log("openai passthrough error:", e)
            return self._json(
                502,
                {"error": {"message": str(e), "type": "api_error"}},
            )

        self.send_response(r.status)
        ct = r.getheader("Content-Type", "application/json")
        self.send_header("Content-Type", ct)
        self.send_header("Connection", "close")
        self.end_headers()

        capture_response = r.status == 200 and body is not None

        if is_stream:
            response_buf = io.BytesIO()
            try:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    self.wfile.write(line)
                    self.wfile.flush()
                    if capture_response:
                        response_buf.write(line)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                c.close()

            if capture_response:
                log("Proxy: queuing OpenAI streaming passthrough task...")
                _enqueue_dataset({
                    "type": "openai_stream",
                    "body": body,
                    "raw_bytes": response_buf.getvalue(),
                })
        else:
            response_bytes = b""
            try:
                response_bytes = r.read()
                self.wfile.write(response_bytes)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                c.close()

            if capture_response and response_bytes:
                log("Proxy: queuing OpenAI sync passthrough task...")
                _enqueue_dataset({
                    "type": "openai_parsed",
                    "body": body,
                    "response_bytes": response_bytes,
                })

    def _w(self, chunk):
        # HTTP/1.1 chunked framing: <hex-len>\r\n<data>\r\n
        self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
        self.wfile.flush()

    def _emit_text_block(self, idx, text):
        self._w(
            sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        if text:
            self._w(
                sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )
        self._w(sse("content_block_stop", {"type": "content_block_stop", "index": idx}))

    def _emit_tool_block(self, idx, tid, name, input_obj):
        self._w(
            sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tid,
                        "name": name,
                        "input": {},
                    },
                },
            )
        )
        self._w(
            sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(input_obj),
                    },
                },
            )
        )
        self._w(sse("content_block_stop", {"type": "content_block_stop", "index": idx}))

    def _stream(self, r, req_model, body=None):
        # Buffer the whole local response, then emit clean Anthropic events. This lets us
        # SALVAGE tool calls that a weak model emitted as text (see salvage_tool_calls).
        # Only local-model responses use this path; Opus is passthrough, so live-streaming
        # of the watched session is unaffected.
        self._w(
            sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": new_msg_id(),
                        "type": "message",
                        "role": "assistant",
                        "model": req_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        )
        self._w(sse("ping", {"type": "ping"}))

        content = []
        tools = {}  # oai index -> {"id","name","args"}
        finish_reason = None
        out_tokens = 0
        last_ping = time.time()

        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                finish_reason = finish_reason or "stop"
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {}) or {}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            u = chunk.get("usage")
            if u and u.get("completion_tokens"):
                out_tokens = u["completion_tokens"]
            if delta.get("content"):
                content.append(delta["content"])
            for tc in delta.get("tool_calls", []) or []:
                slot = tools.setdefault(
                    tc.get("index", 0), {"id": None, "name": "", "args": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            # keepalive ping during long generations (protocol allows pings anywhere)
            if time.time() - last_ping > 5:
                self._w(sse("ping", {"type": "ping"}))
                last_ping = time.time()

        text = "".join(content)
        native = [tools[k] for k in sorted(tools)]
        salvaged = []
        if not native:  # only salvage when llama.cpp gave us no tool_calls
            text, salvaged = salvage_tool_calls(text)

        idx = 0
        response_content = []
        if text.strip():
            self._emit_text_block(idx, text)
            response_content.append({"type": "text", "text": text})
            idx += 1
        for t in native:
            try:
                args = json.loads(t["args"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": t["args"]}
            self._emit_tool_block(
                idx,
                t["id"] or ("toolu_" + uuid.uuid4().hex[:24]),
                t["name"] or "",
                args,
            )
            response_content.append(
                {
                    "type": "tool_use",
                    "id": t["id"] or ("toolu_" + uuid.uuid4().hex[:24]),
                    "name": t["name"] or "",
                    "input": args,
                }
            )
            idx += 1
        for t in salvaged:
            self._emit_tool_block(idx, t["id"], t["name"], t["input"])
            response_content.append(
                {
                    "type": "tool_use",
                    "id": t["id"],
                    "name": t["name"],
                    "input": t["input"],
                }
            )
            idx += 1
        if idx == 0:  # nothing at all -> empty text block
            self._emit_text_block(0, "")
            response_content.append({"type": "text", "text": ""})



        stop = (
            "tool_use"
            if (native or salvaged)
            else FINISH_MAP.get(finish_reason, "end_turn")
        )
        self._w(
            sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop, "stop_sequence": None},
                    "usage": {"output_tokens": out_tokens},
                },
            )
        )
        self._w(sse("message_stop", {"type": "message_stop"}))

        _enqueue_dataset({
            "type": "parsed",
            "body": body,
            "response_content": response_content,
        })


def main():
    start_dataset_writer()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
        srv.shutdown()


if __name__ == "__main__":
    main()
