#!/usr/bin/env python3
"""
anthropic_proxy.py — zero-dependency translation proxy.

Exposes the Anthropic Messages API (POST /v1/messages, POST /v1/messages/count_tokens)
and ROUTES by model name:
  - a "local" model  -> translate Anthropic <-> OpenAI and serve from llama-server
  - anything else    -> transparently pass through to api.anthropic.com

That router behaviour is what lets a *native* Claude Code subagent run on the local
model while the main session stays on Opus: point the whole session at this proxy
(ANTHROPIC_BASE_URL) and give the subagent `model: local-<x>`. Opus traffic is relayed
to Anthropic untouched; only the local model is translated to llama-server.

It also still works as a plain Anthropic->OpenAI shim for one model (the `localagent`
CLI uses it this way; passthrough simply never fires for local-only use):

    ANTHROPIC_BASE_URL=http://127.0.0.1:8090  ANTHROPIC_API_KEY=local  claude --bare -p "..."

Stdlib only. Python 3.10+.

Env:
  LLAMA_PROXY_PORT          listen port              (default 8090)
  LLAMA_PROXY_HOST          listen host              (default 127.0.0.1)
  LLAMA_BASE                llama-server base url     (default http://127.0.0.1:8089)
  LLAMA_MODEL               force upstream model id   (default: auto-detect from /v1/models)
  LOCAL_MODEL_ALIAS         extra name that routes local (default "local-llama"; any
                            model whose name starts with "local" also routes local)
  ANTHROPIC_PASSTHROUGH_BASE upstream for non-local models (default https://api.anthropic.com)
  LLAMA_DISABLE_THINKING    "0" keeps reasoning_content (default disables it)
  LLAMA_PROXY_LOG           append a debug log here   (default: none; stderr only)
"""

import http.client
import json
import os
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PORT = int(os.environ.get("LLAMA_PROXY_PORT", "8090"))
HOST = os.environ.get("LLAMA_PROXY_HOST", "127.0.0.1")
LLAMA_BASE = os.environ.get("LLAMA_BASE", "http://127.0.0.1:8089").rstrip("/")
FORCED_MODEL = os.environ.get("LLAMA_MODEL")
LOG_PATH = os.environ.get("LLAMA_PROXY_LOG")
# Many local models (gemma, qwen3) emit verbose reasoning_content that burns the
# token budget before producing an answer/tool-call. We validate actions, not the
# chain-of-thought, so disable thinking by default. Set LLAMA_DISABLE_THINKING=0 to keep it.
DISABLE_THINKING = os.environ.get("LLAMA_DISABLE_THINKING", "1") != "0"

LOCAL_ALIAS = os.environ.get("LOCAL_MODEL_ALIAS", "local-llama")
ANTHROPIC_UP = os.environ.get(
    "ANTHROPIC_PASSTHROUGH_BASE", "https://api.anthropic.com"
).rstrip("/")

_up = urlparse(LLAMA_BASE)
UP_HOST = _up.hostname
UP_PORT = _up.port or (443 if _up.scheme == "https" else 80)
UP_HTTPS = _up.scheme == "https"

_an = urlparse(ANTHROPIC_UP)
AN_HOST = _an.hostname
AN_PORT = _an.port or (443 if _an.scheme == "https" else 80)
AN_HTTPS = _an.scheme == "https"

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
    """Which requests get served by llama-server vs. passed through to Anthropic."""
    if not name:
        return False
    n = name.lower()
    return (
        name == MODEL
        or name == FORCED_MODEL
        or name == LOCAL_ALIAS
        or n.startswith("local")
    )


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
    if DISABLE_THINKING:
        out["chat_template_kwargs"] = {"enable_thinking": False}

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
        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _DROP_HEADERS
        }
        headers["Host"] = AN_HOST
        if raw:
            headers["Content-Length"] = str(len(raw))
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
        self.send_header("Content-Type", ct)
        # delimit by connection close so we don't have to re-chunk streamed SSE
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
        finally:
            c.close()

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
        if self.path.startswith("/v1/models"):
            # gateway discovery: merge Anthropic's catalogue with the local alias
            local_entry = {
                "type": "model",
                "id": LOCAL_ALIAS,
                "display_name": "Local (%s)" % MODEL,
                "created_at": "",
            }
            data = []
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
                    data = json.loads(payload).get("data", [])
            except Exception as e:  # noqa
                log("models passthrough failed:", e)
            return self._json(
                200,
                {
                    "data": [local_entry] + data,
                    "has_more": False,
                    "first_id": LOCAL_ALIAS,
                    "last_id": data[-1]["id"] if data else LOCAL_ALIAS,
                },
            )
        self._json(404, {"error": "not found"})

    def do_POST(self):
        raw = self._read_raw()
        try:
            body = json.loads(raw) if raw else {}
        except Exception as e:  # noqa
            return self._json(
                400,
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": str(e)},
                },
            )

        req_model = body.get("model", MODEL)

        # ROUTER: non-local models are relayed verbatim to Anthropic.
        if self.path.startswith("/v1/") and not is_local_model(req_model):
            return self._passthrough(raw)

        if self.path.endswith("/count_tokens"):
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
            return self._json(200, openai_to_anthropic(data, req_model))

        # streaming: translate OpenAI SSE -> Anthropic SSE (chunked transfer)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._stream(r, req_model)
            self.wfile.write(b"0\r\n\r\n")  # final chunk
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected mid-stream")
        finally:
            c.close()

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

    def _stream(self, r, req_model):
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
        if text.strip():
            self._emit_text_block(idx, text)
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
            idx += 1
        for t in salvaged:
            self._emit_tool_block(idx, t["id"], t["name"], t["input"])
            idx += 1
        if idx == 0:  # nothing at all -> empty text block
            self._emit_text_block(0, "")

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


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
        srv.shutdown()


if __name__ == "__main__":
    main()
