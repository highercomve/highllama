#!/usr/bin/env python3
"""
localagent/jevcompact.py — Jev-selected, locally-written context compaction.

When a client (Claude Code /compact, opencode's compaction agent, pi's context
checkpoint) asks its model to summarize the conversation, the proxy can answer
that request itself:

  1. If the whole conversation fits the local model's budget, it all goes in:
     measured on real compactions, dropping messages then only loses detail.
  2. Otherwise Jev (TypeSafe System One, https://docs.typesafe.ai) judges every
     message: "is this still needed to continue the task?" — one Noul per item,
     batched — and code keeps every user turn, the recent tail and the items
     Jev scored as needed, up to the budget.
  3. The local highllama model writes the summary from that subset, using the
     client's own compaction instructions so the output format still matches.

FAULT TOLERANT BY CONSTRUCTION: try_compact() returns None on any problem (no
API key, local model not loaded, Jev error, empty summary, ...) and the proxy
then relays the ORIGINAL request unchanged. Nothing is written to the client
until a summary exists.

Env:
  LOCALAGENT_JEV_COMPACT        "1" enables it              (default 0)
  TYPESAFE_API_KEY              Jev API key                 (required)
  TYPESAFE_BASE                 API base                    (default https://api.typesafe.ai)
  TYPESAFE_MODEL                Jev model id                (default jev-latest)
  LOCALAGENT_JEV_THRESHOLD      keep items with p(needed) >= this (default 0.4)
  LOCALAGENT_JEV_BUDGET_TOKENS  max transcript tokens sent to the local model
                                (default 48000; also capped by its context)
  LOCALAGENT_JEV_COMPACT_MODE   "jev" (default) or "full": skip Jev selection and
                                keep newest-first, as a baseline for comparison
  LOCALAGENT_JEV_COMPACT_EVAL   "1": after each compaction, score its coverage with
                                Jev and save a record (default 0; never blocks)
  LOCALAGENT_JEV_COMPACT_DIR    where records go (~/.local/state/localagent/compactions)

CLI:  python3 jevcompact.py replay <request-or-record.json> [--mode jev|full|both]
      python3 jevcompact.py eval   <request-or-record.json> <summary.txt>

Stdlib only.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

JEV_BASE = os.environ.get("TYPESAFE_BASE", "https://api.typesafe.ai").rstrip("/")
JEV_MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
THRESHOLD = float(os.environ.get("LOCALAGENT_JEV_THRESHOLD", "0.4"))
BUDGET_TOKENS = int(os.environ.get("LOCALAGENT_JEV_BUDGET_TOKENS", "48000"))

CHARS_PER_TOKEN = 3.5
KEEP_TAIL = 6              # most recent items are always kept
ITEM_MAX_CHARS = 8000      # per-item cap in the local transcript (head+tail)
JEV_ITEM_CHARS = 2000      # per-item cap inside a Jev question
JEV_BATCH_CHARS = 60000    # ~17k tokens of questions per Jev request (limit 64k)
SUMMARY_MAX_TOKENS = 8192

# Markers of each client's compaction request.
CLAUDE_CODE_MARK = "Your task is to create a detailed summary of"
OPENCODE_MARK = "You are a context summarization agent"
PI_MARK = "You are a context summarization assistant"

_PI_ITEM_SPLIT = re.compile(
    r"\n\n(?=\[(?:User|Assistant|Assistant thinking|Assistant tool calls|Tool result)\]: )"
)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def is_enabled() -> bool:
    return (os.environ.get("LOCALAGENT_JEV_COMPACT", "0") != "0"
            and bool(os.environ.get("TYPESAFE_API_KEY")))


# ---------------------------------------------------------------------------
# detection and extraction
# ---------------------------------------------------------------------------
def _text(content: Any) -> str:
    """Flatten Anthropic/OpenAI content (str | list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content if isinstance(content, list) else [content]:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t in ("text", "input_text", "output_text"):
            parts.append(b.get("text", ""))
        elif t == "thinking":
            parts.append("[thinking] " + b.get("thinking", ""))
        elif t == "tool_use":
            parts.append("[tool call] %s(%s)" % (
                b.get("name", ""), json.dumps(b.get("input", {}), ensure_ascii=False)))
        elif t == "tool_result":
            parts.append("[tool result] " + _text(b.get("content")))
    return "\n".join(p for p in parts if p)


def _normalize(body: Dict[str, Any]) -> Dict[str, Any]:
    """Map an OpenAI Responses body (instructions + input items) to the
    messages shape; Anthropic and chat-completions bodies pass through."""
    if "messages" in body or "input" not in body:
        return body
    inp = body.get("input")
    if isinstance(inp, str):
        inp = [{"role": "user", "content": inp}]
    msgs: List[Dict[str, Any]] = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    for it in inp or []:
        if not isinstance(it, dict):
            continue
        t = it.get("type")
        if t == "function_call":
            msgs.append({"role": "assistant", "content": "[tool call] %s(%s)" % (
                it.get("name", ""), it.get("arguments", ""))})
        elif t == "function_call_output":
            out = it.get("output")
            msgs.append({"role": "tool", "content": out if isinstance(out, str) else _text(out)})
        elif it.get("role"):
            msgs.append({"role": it["role"], "content": it.get("content")})
    return {"messages": msgs}


def _system_text(body: Dict[str, Any]) -> str:
    sys_parts = [_text(body.get("system"))]
    for m in body.get("messages") or []:
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            sys_parts.append(_text(m.get("content")))
    return "\n".join(p for p in sys_parts if p)


def _chat_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [m for m in body.get("messages") or []
            if isinstance(m, dict) and m.get("role") not in ("system", "developer")]


def detect(body: Dict[str, Any]) -> Optional[str]:
    """Return "claude-code" | "opencode" | "pi" for a compaction request, else None."""
    if not isinstance(body, dict):
        return None
    body = _normalize(body)
    msgs = _chat_messages(body)
    if not msgs or msgs[-1].get("role") != "user":
        return None
    last = _text(msgs[-1].get("content"))
    system = _system_text(body)
    if PI_MARK in system and "<conversation>" in last:
        return "pi"
    if OPENCODE_MARK in system and len(msgs) > 1:
        return "opencode"
    if CLAUDE_CODE_MARK in last and len(msgs) > 1:
        return "claude-code"
    return None


def _message_items(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for m in msgs:
        role = m.get("role")
        text = _text(m.get("content"))
        for tc in m.get("tool_calls") or []:  # OpenAI assistant tool calls
            fn = tc.get("function") or {}
            text += "\n[tool call] %s(%s)" % (fn.get("name", ""), fn.get("arguments", ""))
        text = text.strip()
        if not text:
            continue
        content = m.get("content")
        is_tool = role == "tool" or (
            isinstance(content, list) and content
            and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content))
        kind = "tool" if is_tool else ("user" if role == "user" else "assistant")
        if is_tool and text.startswith("[tool result] "):
            text = text[len("[tool result] "):]
        label = {"tool": "Tool result", "user": "User", "assistant": "Assistant"}[kind]
        items.append({"kind": kind, "text": "[%s]: %s" % (label, text)})
    return items


def extract(body: Dict[str, Any], kind: str) -> Tuple[List[Dict[str, Any]], str, str]:
    """Split a compaction request into (items, instruction, system)."""
    body = _normalize(body)
    msgs = _chat_messages(body)
    system = _system_text(body)
    if kind == "pi":
        prompt = _text(msgs[-1].get("content"))
        start = prompt.index("<conversation>") + len("<conversation>")
        end = prompt.rindex("</conversation>")
        instruction = prompt[end + len("</conversation>"):].strip()
        items = []
        for chunk in _PI_ITEM_SPLIT.split(prompt[start:end].strip("\n")):
            if not chunk.strip():
                continue
            k = ("user" if chunk.startswith("[User]") else
                 "tool" if chunk.startswith("[Tool result]") else "assistant")
            items.append({"kind": k, "text": chunk})
        return items, instruction, system
    instruction = _text(msgs[-1].get("content")).strip()
    return _message_items(msgs[:-1]), instruction, system


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return "%s\n[... %d chars omitted ...]\n%s" % (text[:head], len(text) - limit, text[-tail:])


# ---------------------------------------------------------------------------
# Jev scoring
# ---------------------------------------------------------------------------
def _jev_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(
        JEV_BASE + "/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"],
                 "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def _task_state(items: List[Dict[str, Any]]) -> Dict[str, str]:
    users = [it["text"] for it in items if it["kind"] == "user"]
    return {
        "original_request": _clip(users[0], 3000) if users else "",
        "latest_user_messages": _clip("\n\n".join(users[-3:]), 4000) if users else "",
    }


QUESTION = (
    "`item` is one message from a coding-agent conversation that is about to be "
    "compacted into a summary. Will the agent still need the information in `item` "
    "to continue the work described in `state`?"
)
CRITERIA = {
    "true": "It holds something the summary must keep: the goal or a requirement, a "
            "decision, a file path or code that was changed, an error still being "
            "worked on, a result or finding, or a pending next step.",
    "false": "It is routine or superseded: a listing or read with nothing kept from "
             "it, a retried or abandoned step, chit-chat, or detail already captured "
             "by later messages.",
}


def score_items(items: List[Dict[str, Any]]) -> List[float]:
    """p(needed) for every item, via batched parallel Jev requests."""
    state = _task_state(items)
    batches: List[List[int]] = [[]]
    size = 0
    for i, it in enumerate(items):
        n = min(len(it["text"]), JEV_ITEM_CHARS) + len(QUESTION)
        if batches[-1] and size + n > JEV_BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(i)
        size += n

    def run(idx: List[int]) -> Dict[int, float]:
        questions = {
            "i%d" % i: {
                "type": "noul",
                "instructions": {"item": _clip(items[i]["text"], JEV_ITEM_CHARS),
                                 "question": QUESTION},
                "criteria": CRITERIA,
            }
            for i in idx
        }
        resp = _jev_request({"model": JEV_MODEL, "state": state, "questions": questions})
        answers = resp["answers"]
        return {i: float(answers["i%d" % i]["noul"]) for i in idx}

    scores: Dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(batches))) as ex:
        for part in ex.map(run, batches):
            scores.update(part)
    return [scores[i] for i in range(len(items))]


def fits(items: List[Dict[str, Any]], budget_chars: int) -> bool:
    """Does the whole (per-item clipped) conversation fit the budget?"""
    return sum(min(len(it["text"]), ITEM_MAX_CHARS) for it in items) <= budget_chars


def select(items: List[Dict[str, Any]], scores: List[float], budget_chars: int) -> List[int]:
    """Indices to keep, in conversation order."""
    n = len(items)
    cost = [min(len(it["text"]), ITEM_MAX_CHARS) for it in items]
    must = set(range(max(0, n - KEEP_TAIL), n))
    must.update(i for i, it in enumerate(items) if it["kind"] == "user")
    keep, used = set(), 0
    # Mandatory items first, newest first, so an oversized history still fits.
    for i in sorted(must, reverse=True):
        if used + cost[i] <= budget_chars or not keep:
            keep.add(i)
            used += cost[i]
    ranked = sorted((i for i in range(n) if i not in keep and scores[i] >= THRESHOLD),
                    key=lambda i: scores[i], reverse=True)
    for i in ranked:
        if used + cost[i] <= budget_chars:
            keep.add(i)
            used += cost[i]
    return sorted(keep)


def render(items: List[Dict[str, Any]], keep: List[int]) -> str:
    out, prev = [], -1
    for i in keep:
        if i - prev > 1:
            out.append("[... %d messages omitted as not needed ...]" % (i - prev - 1))
        out.append(_clip(items[i]["text"], ITEM_MAX_CHARS))
        prev = i
    if len(items) - 1 > prev:
        out.append("[... %d messages omitted as not needed ...]" % (len(items) - 1 - prev))
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# local model
# ---------------------------------------------------------------------------
def _get_json(url: str, timeout: float = 3) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _parse_ctx(v: str) -> int:
    v = v.strip().lower()
    if v.endswith("k"):
        return int(float(v[:-1]) * 1024)
    return int(v)


def local_status(llama_base: str) -> Optional[Tuple[str, int]]:
    """(model_id, per-slot ctx) of a LOADED local chat model, or None.

    Handles both a plain llama-server (loaded once /health is ok) and router
    mode, where /v1/models reports each model's status.value.
    """
    data = _get_json(llama_base + "/v1/models")
    models = [m for m in (data.get("data") or data.get("models") or [])
              if "embed" not in (m.get("id") or "").lower()]
    for m in models:
        status = m.get("status")
        if status is None:
            continue
        if status.get("value") != "loaded":
            continue
        args = status.get("args") or []
        ctx, par = 32768, 1
        for flag, val in zip(args, args[1:]):
            if flag in ("--ctx-size", "-c"):
                ctx = _parse_ctx(val)
            elif flag in ("--parallel", "-np"):
                par = max(1, int(val))
        return m.get("id"), ctx // par
    if models and all(m.get("status") is None for m in models):
        req = urllib.request.Request(llama_base + "/health")
        with urllib.request.urlopen(req, timeout=3) as r:
            if r.status != 200:
                return None
        props = _get_json(llama_base + "/props")
        gen = props.get("default_generation_settings") or {}
        ctx = int(gen.get("n_ctx") or 0) or 32768
        par = int(props.get("total_slots") or 1)
        return models[0].get("id"), ctx // max(1, par)
    return None


def summarize_local(llama_base: str, model: str, system: str, transcript: str,
                    instruction: str, max_tokens: int) -> str:
    user = (
        "<conversation>\n%s\n</conversation>\n\n"
        "(The conversation above is a selection: messages judged not needed were "
        "omitted and very long ones clipped. Summarize what is there.)\n\n%s"
        % (transcript, instruction)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or (
                "You are a context summarization assistant. Do NOT continue the "
                "conversation. ONLY output the requested summary.")},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        llama_base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    text = _THINK_RE.sub("", choice["message"].get("content") or "").strip()
    if choice.get("finish_reason") == "length":
        raise RuntimeError("local summary hit max_tokens")
    return text


# ---------------------------------------------------------------------------
# evaluation: how much of the conversation does a summary preserve?
# ---------------------------------------------------------------------------
_PATH_RE = re.compile(r"(?:[\w.-]+/)*[\w.-]+\.(?:py|sh|md|js|ts|tsx|json|go|rs|c|h|cc|cpp|"
                      r"zig|toml|ya?ml|ini|conf|txt|html|css|sql|bb|bbappend|dts)\b")
_ERR_RE = re.compile(r"Traceback|Error\b|error:|FAILED|failed|exception", re.IGNORECASE)


def facts(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Facts a good summary must keep, picked by code (no model involved):
    every user request, the files touched, errors in the last third of the
    conversation, and the last assistant message (the current state)."""
    out: List[Dict[str, str]] = []
    for it in items:
        if it["kind"] == "user":
            out.append({"kind": "request", "fact": _clip(it["text"], 600)})
    paths: List[str] = []
    for it in items:
        # Only paths the agent acted on (tool-call arguments), not every name
        # that scrolled past in an `ls` or grep result.
        calls = "\n".join(line for line in it["text"].splitlines()
                          if "[tool call]" in line or line.startswith("[Assistant tool calls]"))
        if it["kind"] == "assistant" and calls:
            for p in _PATH_RE.findall(calls):
                name = p.rsplit("/", 1)[-1]
                if name not in paths:
                    paths.append(name)
    out += [{"kind": "file", "fact": "The file `%s` was read, changed or discussed." % p}
            for p in paths[:25]]
    tail = items[len(items) * 2 // 3:]
    errs = [it for it in tail if it["kind"] == "tool" and _ERR_RE.search(it["text"])]
    out += [{"kind": "error", "fact": _clip(it["text"], 500)} for it in errs[-5:]]
    last = [it for it in items if it["kind"] == "assistant" and not it["text"].startswith(
        ("[Assistant tool calls]", "[Assistant thinking]"))]
    if last:
        out.append({"kind": "state", "fact": _clip(last[-1]["text"], 1500)})
    return out


def coverage(summary: str, fact_list: List[Dict[str, str]]) -> Dict[str, Any]:
    """Jev judges, per fact, whether the summary preserves it."""
    questions = {
        "f%d" % i: {
            "type": "noul",
            "instructions": {
                "fact": f["fact"],
                "question": "`state` is a summary of a coding-agent conversation, written "
                            "so another agent can continue the work. Does it preserve "
                            "`fact` (literally or in substance) well enough to act on it?",
            },
        }
        for i, f in enumerate(fact_list)
    }
    resp = _jev_request({"model": JEV_MODEL, "state": summary, "questions": questions})
    probs = [float(resp["answers"]["f%d" % i]["noul"]) for i in range(len(fact_list))]
    by_kind: Dict[str, List[float]] = {}
    for f, p in zip(fact_list, probs):
        by_kind.setdefault(f["kind"], []).append(p)
    return {
        "score": round(sum(probs) / len(probs), 3) if probs else None,
        "by_kind": {k: round(sum(v) / len(v), 3) for k, v in by_kind.items()},
        "missing": [dict(f, p=round(p, 2)) for f, p in zip(fact_list, probs) if p < 0.5],
    }


def _save_record(record: Dict[str, Any]) -> str:
    d = os.path.expanduser(os.environ.get(
        "LOCALAGENT_JEV_COMPACT_DIR", "~/.local/state/localagent/compactions"))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "%s-%s-%s.json" % (
        time.strftime("%Y%m%d-%H%M%S"), record["kind"], record["mode"]))
    with open(path, "w") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def compact(body: Dict[str, Any], llama_base: str, mode: str = "jev") -> Dict[str, Any]:
    """Run one compaction. Raises on any problem; try_compact() wraps it.

    mode "jev": everything if it fits the budget, else Jev picks what the
    local model sees (the product path).
    mode "full": no Jev — newest messages first until the budget is full
    (the baseline to compare against).
    """
    kind = detect(body)
    if not kind:
        raise ValueError("not a compaction request")
    t0 = time.time()
    status = local_status(llama_base)
    if not status:
        raise LookupError("local model not loaded")
    model, ctx = status
    items, instruction, system = extract(body, kind)
    if len(items) < 2 or not instruction:
        raise ValueError("nothing to compact")
    max_tokens = min(SUMMARY_MAX_TOKENS, ctx // 4)
    prompt_overhead = (len(instruction) + len(system)) / CHARS_PER_TOKEN + 500
    budget_tokens = min(BUDGET_TOKENS, ctx - max_tokens - prompt_overhead)
    if budget_tokens < 4000:
        raise ValueError("local ctx %d too small" % ctx)
    budget_chars = int(budget_tokens * CHARS_PER_TOKEN)
    if mode == "jev" and not fits(items, budget_chars):
        selection, scores = "jev", score_items(items)
    else:
        selection, scores = ("all" if fits(items, budget_chars) else "newest"), [1.0] * len(items)
    t_jev = time.time() - t0
    keep = select(items, scores, budget_chars)
    transcript = render(items, keep)
    text = summarize_local(llama_base, model, system, transcript, instruction, max_tokens)
    if not text:
        raise ValueError("empty local summary")
    stats = {
        "mode": mode, "selection": selection, "items": len(items), "kept": len(keep),
        "chars_in": sum(len(it["text"]) for it in items),
        "chars_kept": len(transcript), "summary_chars": len(text),
        "jev_s": round(t_jev, 1), "total_s": round(time.time() - t0, 1),
        "local_model": model,
    }
    return {"kind": kind, "text": text, "stats": stats, "items": items,
            "scores": scores, "keep": keep}


def evaluate_and_save(body: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    cov = coverage(result["text"], facts(result["items"]))
    record = {
        "kind": result["kind"], "mode": result["stats"]["mode"],
        "stats": result["stats"], "coverage": cov, "summary": result["text"],
        "items": [dict(it, score=round(sc, 3), kept=i in set(result["keep"]))
                  for i, (it, sc) in enumerate(zip(result["items"], result["scores"]))],
        "request": body,
    }
    record["path"] = _save_record(record)
    return record


def try_compact(body: Dict[str, Any], llama_base: str, log=print) -> Optional[Dict[str, Any]]:
    """Answer a compaction request with Jev + the local model.

    Returns {"kind", "text", "stats"} or None — None means "relay the original
    request unchanged". Never raises.
    """
    if not is_enabled():
        return None
    kind = detect(body)
    if not kind:
        return None
    mode = os.environ.get("LOCALAGENT_JEV_COMPACT_MODE", "jev")
    try:
        result = compact(body, llama_base, mode)
    except LookupError as e:
        log("jev-compact: %s compaction, %s -> normal compaction" % (kind, e))
        return None
    except Exception as e:  # any failure -> the untouched request goes out as usual
        log("jev-compact: %s compaction failed (%s: %s) -> normal compaction"
            % (kind, type(e).__name__, e))
        return None
    log("jev-compact: %s compaction served locally %s" % (kind, json.dumps(result["stats"])))
    if os.environ.get("LOCALAGENT_JEV_COMPACT_EVAL", "0") != "0":
        try:  # evaluation is diagnostics only: it never blocks the reply
            rec = evaluate_and_save(body, result)
            log("jev-compact: coverage %s %s missing=%d -> %s" % (
                rec["coverage"]["score"], json.dumps(rec["coverage"]["by_kind"]),
                len(rec["coverage"]["missing"]), rec["path"]))
        except Exception as e:
            log("jev-compact: evaluation failed (%s: %s)" % (type(e).__name__, e))
    return result


# ---------------------------------------------------------------------------
# CLI: replay a saved request, or score an existing summary
# ---------------------------------------------------------------------------
def _print_record(rec: Dict[str, Any]) -> None:
    st, cov = rec["stats"], rec["coverage"]
    print("== %s/%s  kept %d/%d items (%d -> %d chars)  summary %d chars  %.1fs" % (
        st["mode"], st.get("selection", "?"), st["kept"], st["items"], st["chars_in"], st["chars_kept"],
        st["summary_chars"], st["total_s"]))
    print("   coverage %s  %s" % (cov["score"], cov["by_kind"]))
    for m in cov["missing"]:
        print("   missing %.2f [%s] %s" % (m["p"], m["kind"], m["fact"][:110].replace("\n", " ")))
    print("   saved:", rec["path"])


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="jevcompact.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="rerun a compaction request (JSON body or saved record)")
    r.add_argument("request")
    r.add_argument("--mode", choices=["jev", "full", "both"], default="both")
    r.add_argument("--llama", default=os.environ.get("LLAMA_BASE", "http://127.0.0.1:8089"))
    e = sub.add_parser("eval", help="score an existing summary against a request")
    e.add_argument("request")
    e.add_argument("summary", help="file with the summary text")
    a = ap.parse_args(argv)

    with open(a.request) as f:
        body = json.load(f)
    body = body.get("request", body)  # a saved record carries the request
    kind = detect(body)
    if not kind:
        print("not a compaction request", file=sys.stderr)
        return 2
    if a.cmd == "eval":
        items, _, _ = extract(body, kind)
        with open(a.summary) as f:
            cov = coverage(f.read(), facts(items))
        print("coverage %s  %s" % (cov["score"], cov["by_kind"]))
        for m in cov["missing"]:
            print("   missing %.2f [%s] %s" % (m["p"], m["kind"], m["fact"][:110].replace("\n", " ")))
        return 0
    for mode in (["jev", "full"] if a.mode == "both" else [a.mode]):
        _print_record(evaluate_and_save(body, compact(body, a.llama, mode)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
