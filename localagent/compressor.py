#!/usr/bin/env python3
"""
localagent/compressor.py — native context compression layer for localagent proxy.

Features:
1. Semantic Chunk Pruning via local highllama embeddings (/v1/embeddings).
2. Fallback to deterministic Head/Tail truncation if embeddings unavailable.
3. Native Spillover (CCR): writes raw uncompressed outputs to disk so agents
   can inspect them with Read if needed.
4. Output Shaper (optional, disabled by default): preserves user thinking effort/budget 100%.
5. Terminal & JSON Cleaner: minifies JSON and strips ANSI escape codes.
6. Token Savings Reporter (like rtk gain): SQLite tracking of events & ASCII dashboard.

Zero external dependencies — Python standard library only.
"""

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Configuration via environment.
# Compression is opt-in while it is still in testing: it needs the embeddings
# server running, and the shell wrappers only start that when it is switched on.
COMPRESS_ENABLED = os.environ.get("LOCALAGENT_COMPRESS", "0") != "0"
PRESERVE_RECENT_TURNS = int(os.environ.get("LOCALAGENT_COMPRESS_PRESERVE_TURNS", "2"))
MAX_LINES = int(os.environ.get("LOCALAGENT_COMPRESS_MAX_LINES", "50"))
MAX_CHARS = int(os.environ.get("LOCALAGENT_COMPRESS_MAX_CHARS", "1500"))
HEAD_LINES = int(os.environ.get("LOCALAGENT_COMPRESS_HEAD_LINES", "15"))
TAIL_LINES = int(os.environ.get("LOCALAGENT_COMPRESS_TAIL_LINES", "15"))
# highllama serves embeddings from a dedicated embedding-model server on :8091 (the chat
# server on :8089 no longer answers /v1/embeddings).
EMBED_BASE = os.environ.get("LOCALAGENT_EMBED_BASE", "http://127.0.0.1:8091").rstrip("/")
# Disabled by default: preserves user's thinking effort / budget setting 100%
THINKING_DAMPEN = os.environ.get("LOCALAGENT_THINKING_DAMPEN", "0") == "1"
ROUTINE_THINKING_BUDGET = int(os.environ.get("LOCALAGENT_ROUTINE_THINKING_BUDGET", "1500"))

STATE_DIR = os.environ.get(
    "LOCALAGENT_STATE",
    os.path.join(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")), "localagent"),
)
SPILL_DIR = os.environ.get("LOCALAGENT_SPILL_DIR", os.path.join(STATE_DIR, "spill"))
COMPRESSION_DB = os.environ.get("LOCALAGENT_COMPRESSION_DB", os.path.join(STATE_DIR, "compression.db"))

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def is_compression_enabled() -> bool:
    return os.environ.get("LOCALAGENT_COMPRESS", "0") != "0"


def _get_db(db_path: str = COMPRESSION_DB) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS compression_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        source TEXT NOT NULL,
        method TEXT NOT NULL,
        orig_chars INTEGER NOT NULL,
        final_chars INTEGER NOT NULL,
        orig_tokens INTEGER NOT NULL,
        final_tokens INTEGER NOT NULL,
        saved_tokens INTEGER NOT NULL,
        duration_ms REAL NOT NULL DEFAULT 0.0
    );
    """)
    return conn


def record_compression_event(
    endpoint: str,
    source: str,
    method: str,
    orig_chars: int,
    final_chars: int,
    duration_ms: float = 0.0,
    db_path: str = COMPRESSION_DB,
):
    """Records a single compression event to SQLite."""
    try:
        orig_tokens = max(1, orig_chars // 4)
        final_tokens = max(1, final_chars // 4)
        saved_tokens = max(0, orig_tokens - final_tokens)
        ts = datetime.now(timezone.utc).isoformat()
        conn = _get_db(db_path)
        with conn:
            conn.execute(
                """
                INSERT INTO compression_events
                (timestamp, endpoint, source, method, orig_chars, final_chars, orig_tokens, final_tokens, saved_tokens, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    endpoint,
                    source,
                    method,
                    orig_chars,
                    final_chars,
                    orig_tokens,
                    final_tokens,
                    saved_tokens,
                    duration_ms,
                ),
            )
        conn.close()
    except Exception:
        pass


def clean_terminal_noise(text: str) -> str:
    """Strips ANSI escape sequences, carriage returns, and excessive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "")
    text = ANSI_ESCAPE_RE.sub("", text)
    # Collapse 3+ consecutive newlines into 2
    return re.sub(r"\n{3,}", "\n\n", text)


def minify_json_if_applicable(text: str) -> str:
    """Minifies JSON content by stripping unnecessary whitespace."""
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, separators=(",", ":"))
        except Exception:
            pass
    return text


def spill_content(text: str, spill_dir: str = SPILL_DIR) -> Tuple[str, str]:
    """Saves raw uncompressed text to disk and returns (sha256_prefix, file_path)."""
    try:
        os.makedirs(spill_dir, exist_ok=True)
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        file_path = os.path.join(spill_dir, f"{h}.txt")
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        return h, file_path
    except Exception:
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
        return h, f"/tmp/localagent-spill-{h}.txt"


def chunk_text(text: str, chunk_lines: int = 35, overlap: int = 5) -> List[Tuple[int, int, str]]:
    """Splits text into line-based overlapping chunks, returning (start_line, end_line, chunk_text)."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    chunks = []
    i = 0
    total = len(lines)
    while i < total:
        end = min(i + chunk_lines, total)
        chunk_str = "".join(lines[i:end])
        chunks.append((i + 1, end, chunk_str))
        if end >= total:
            break
        i += chunk_lines - overlap
    return chunks


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def fetch_embeddings(
    texts: List[str], embed_base: str = EMBED_BASE, timeout: float = 2.0
) -> Optional[List[List[float]]]:
    """Calls the local /v1/embeddings endpoint. Returns None on failure or timeout."""
    try:
        url = f"{embed_base.rstrip('/')}/v1/embeddings"
        payload = json.dumps({"input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            res = json.loads(resp.read().decode("utf-8"))
            data = res.get("data", [])
            if len(data) != len(texts):
                return None
            data.sort(key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in data]
    except Exception:
        return None


def prune_text(
    text: str,
    query: str = "",
    max_lines: int = MAX_LINES,
    max_chars: int = MAX_CHARS,
    embed_base: str = EMBED_BASE,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compresses text if it exceeds line or character thresholds:
    1. Minifies JSON / cleans terminal noise.
    2. Spills full uncompressed raw text to disk.
    3. Tries semantic chunk pruning via embeddings.
    4. Falls back to head/tail truncation if embeddings fail.
    """
    cleaned = clean_terminal_noise(text)
    cleaned = minify_json_if_applicable(cleaned)

    lines = cleaned.splitlines()
    orig_chars = len(text)
    orig_lines = len(lines)

    if orig_chars < max_chars and orig_lines < max_lines:
        return cleaned, {"pruned": False, "orig_chars": orig_chars, "final_chars": len(cleaned)}

    # Always spill full original for lossless recovery
    h, spill_path = spill_content(text)

    # 1. Attempt Semantic Chunk Pruning if query is present
    if query and query.strip() and orig_lines > 40:
        chunks = chunk_text(cleaned, chunk_lines=35, overlap=5)
        if len(chunks) >= 3:
            texts_to_embed = [query.strip()] + [c[2] for c in chunks]
            embeddings = fetch_embeddings(texts_to_embed, embed_base=embed_base)
            if embeddings and len(embeddings) == len(texts_to_embed):
                query_vec = embeddings[0]
                chunk_embs = embeddings[1:]

                scored = []
                for idx, (s, e, c_text) in enumerate(chunks):
                    sim = cosine_similarity(query_vec, chunk_embs[idx])
                    scored.append((sim, idx, s, e, c_text))

                # Take top 3 most relevant chunks
                scored.sort(key=lambda x: x[0], reverse=True)
                top_k = scored[:3]
                top_k.sort(key=lambda x: x[1])

                out = [
                    f"[localagent semantic compression: kept {len(top_k)}/{len(chunks)} relevant sections ({orig_lines} lines -> ~{len(top_k)*30} lines)]",
                    f"[Full original content saved to: {spill_path} — inspect with Read if needed]\n---",
                ]
                for sim, _, s, e, c_text in top_k:
                    out.append(f"[Lines {s}-{e} | Relevance: {sim:.2f}]:\n{c_text.strip()}\n---")

                pruned_result = "\n".join(out)
                return pruned_result, {
                    "pruned": True,
                    "method": "semantic_embedding",
                    "orig_chars": orig_chars,
                    "final_chars": len(pruned_result),
                    "spill_path": spill_path,
                }

    # 2. Fallback: Deterministic Head/Tail Truncation
    head = lines[:HEAD_LINES]
    tail = lines[-TAIL_LINES:] if len(lines) > HEAD_LINES else []
    omitted = len(lines) - len(head) - len(tail)

    out = head + [
        f"\n[... localagent truncated {omitted} lines ({orig_chars} chars) ...]",
        f"[Full original saved to: {spill_path} — inspect with Read if needed]\n",
    ] + tail

    pruned_result = "\n".join(out)
    return pruned_result, {
        "pruned": True,
        "method": "head_tail",
        "orig_chars": orig_chars,
        "final_chars": len(pruned_result),
        "spill_path": spill_path,
    }


def is_routine_tool_return(message: Dict[str, Any]) -> bool:
    """Returns True if a turn only contains routine, non-error tool results."""
    content = message.get("content")
    if not isinstance(content, list):
        return False

    has_tool_result = False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            has_tool_result = True
            if block.get("is_error"):
                return False
            block_content = str(block.get("content", "")).lower()
            if "error:" in block_content or "traceback" in block_content or "failed" in block_content:
                return False

    return has_tool_result


def extract_user_query(messages: List[Dict[str, Any]]) -> str:
    """Extracts the most recent user instruction/question for semantic relevance matching."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            elif isinstance(content, list):
                text_parts = [
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") in ("text", "input_text")
                ]
                if text_parts:
                    return " ".join(text_parts).strip()
    return ""


def compress_anthropic_payload(body: Dict[str, Any], provider: str = "anthropic") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compresses an Anthropic Messages API payload (/v1/messages).
    Returns (modified_body, stats_dict).
    """
    if not is_compression_enabled():
        return body, {"modified": False}

    messages = body.get("messages", [])
    if not messages:
        return body, {"modified": False}

    modified = False
    stats: Dict[str, Any] = {"modified": False, "pruned_blocks": 0, "chars_saved": 0}

    # Map tool_use_ids to tool names (Read, Bash, etc.)
    tool_id_to_name: Dict[str, str] = {}
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tid = b.get("id")
                    tname = b.get("name")
                    if tid and tname:
                        tool_id_to_name[tid] = tname

    # 1. Output Shaper: Dampen thinking budget on routine tool responses
    if THINKING_DAMPEN and "thinking" in body and isinstance(body["thinking"], dict):
        last_turn = messages[-1]
        if last_turn.get("role") == "user" and is_routine_tool_return(last_turn):
            curr_budget = body["thinking"].get("budget_tokens", 8192)
            if curr_budget > ROUTINE_THINKING_BUDGET:
                body["thinking"]["budget_tokens"] = ROUTINE_THINKING_BUDGET
                stats["thinking_dampened"] = True
                modified = True
                record_compression_event(
                    endpoint=provider,
                    source="Thinking Budget",
                    method="thinking_dampen",
                    orig_chars=curr_budget * 4,
                    final_chars=ROUTINE_THINKING_BUDGET * 4,
                    duration_ms=0.0,
                )

    # 2. Extract recent query context for semantic pruning
    query = extract_user_query(messages)

    # 3. Context Pruning for turns outside the preserve window
    num_to_inspect = max(0, len(messages) - PRESERVE_RECENT_TURNS)
    for i in range(num_to_inspect):
        msg = messages[i]
        content = msg.get("content")

        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    res_content = block.get("content")
                    tid = block.get("tool_use_id", "")
                    src_name = tool_id_to_name.get(tid, "Tool Result")
                    if isinstance(res_content, str) and len(res_content) > MAX_CHARS:
                        t0 = time.perf_counter()
                        pruned_res, block_stats = prune_text(res_content, query=query)
                        dur_ms = (time.perf_counter() - t0) * 1000.0
                        if block_stats.get("pruned"):
                            block["content"] = pruned_res
                            modified = True
                            stats["pruned_blocks"] += 1
                            c_saved = block_stats["orig_chars"] - block_stats["final_chars"]
                            stats["chars_saved"] += c_saved
                            record_compression_event(
                                endpoint=provider,
                                source=src_name,
                                method=block_stats.get("method", "prune"),
                                orig_chars=block_stats["orig_chars"],
                                final_chars=block_stats["final_chars"],
                                duration_ms=dur_ms,
                            )
        elif isinstance(content, str) and len(content) > MAX_CHARS and msg.get("role") != "system":
            t0 = time.perf_counter()
            pruned_res, block_stats = prune_text(content, query=query)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            if block_stats.get("pruned"):
                msg["content"] = pruned_res
                modified = True
                stats["pruned_blocks"] += 1
                c_saved = block_stats["orig_chars"] - block_stats["final_chars"]
                stats["chars_saved"] += c_saved
                record_compression_event(
                    endpoint=provider,
                    source=f"User Turn {i}",
                    method=block_stats.get("method", "prune"),
                    orig_chars=block_stats["orig_chars"],
                    final_chars=block_stats["final_chars"],
                    duration_ms=dur_ms,
                )

    stats["modified"] = modified
    return body, stats


def compress_openai_payload(body: Dict[str, Any], provider: str = "openai") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compresses an OpenAI Chat Completions API payload (/v1/chat/completions).
    Returns (modified_body, stats_dict).
    """
    if not is_compression_enabled():
        return body, {"modified": False}

    messages = body.get("messages", [])
    if not messages and isinstance(body.get("input"), list):
        messages = body["input"]  # OpenAI Responses API: /v1/responses `input` list
    if not messages:
        return body, {"modified": False}

    modified = False
    stats: Dict[str, Any] = {"modified": False, "pruned_blocks": 0, "chars_saved": 0}

    # Map tool_call_id to function name
    tool_id_to_name: Dict[str, str] = {}
    for m in messages:
        if m.get("role") == "assistant" and "tool_calls" in m:
            for tc in m.get("tool_calls", []):
                t_id = tc.get("id")
                func_name = tc.get("function", {}).get("name")
                if t_id and func_name:
                    tool_id_to_name[t_id] = func_name

    # 1. Output Shaper: Dampen reasoning effort on routine tool responses
    if THINKING_DAMPEN and "reasoning_effort" in body:
        last_turn = messages[-1]
        if last_turn.get("role") == "tool" or (
            last_turn.get("role") == "user" and is_routine_tool_return(last_turn)
        ):
            if body.get("reasoning_effort") in ("high", "medium"):
                body["reasoning_effort"] = "low"
                stats["thinking_dampened"] = True
                modified = True
                record_compression_event(
                    endpoint=provider,
                    source="Reasoning Effort",
                    method="effort_dampen",
                    orig_chars=4000,
                    final_chars=1000,
                    duration_ms=0.0,
                )

    # 2. Extract recent query
    query = extract_user_query(messages)

    # 3. Context Pruning
    num_to_inspect = max(0, len(messages) - PRESERVE_RECENT_TURNS)
    for i in range(num_to_inspect):
        msg = messages[i]
        content = msg.get("content")
        role = msg.get("role")

        if msg.get("type") == "function_call_output":
            # Responses API tool result: the big text is in `output`.
            if isinstance(msg.get("output"), str) and len(msg["output"]) > MAX_CHARS:
                t0 = time.perf_counter()
                pruned_res, block_stats = prune_text(msg["output"], query=query)
                dur_ms = (time.perf_counter() - t0) * 1000.0
                if block_stats.get("pruned"):
                    msg["output"] = pruned_res
                    modified = True
                    stats["pruned_blocks"] += 1
                    c_saved = block_stats["orig_chars"] - block_stats["final_chars"]
                    stats["chars_saved"] += c_saved
                    record_compression_event(
                        endpoint=provider,
                        source=msg.get("call_id", "Tool Result"),
                        method=block_stats.get("method", "prune"),
                        orig_chars=block_stats["orig_chars"],
                        final_chars=block_stats["final_chars"],
                        duration_ms=dur_ms,
                    )
            continue

        if role == "tool" and isinstance(content, str) and len(content) > MAX_CHARS:
            t_id = msg.get("tool_call_id", "")
            src_name = tool_id_to_name.get(t_id, "Tool Result")
            t0 = time.perf_counter()
            pruned_res, block_stats = prune_text(content, query=query)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            if block_stats.get("pruned"):
                msg["content"] = pruned_res
                modified = True
                stats["pruned_blocks"] += 1
                c_saved = block_stats["orig_chars"] - block_stats["final_chars"]
                stats["chars_saved"] += c_saved
                record_compression_event(
                    endpoint=provider,
                    source=src_name,
                    method=block_stats.get("method", "prune"),
                    orig_chars=block_stats["orig_chars"],
                    final_chars=block_stats["final_chars"],
                    duration_ms=dur_ms,
                )
        elif isinstance(content, str) and len(content) > MAX_CHARS and role != "system":
            t0 = time.perf_counter()
            pruned_res, block_stats = prune_text(content, query=query)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            if block_stats.get("pruned"):
                msg["content"] = pruned_res
                modified = True
                stats["pruned_blocks"] += 1
                c_saved = block_stats["orig_chars"] - block_stats["final_chars"]
                stats["chars_saved"] += c_saved
                record_compression_event(
                    endpoint=provider,
                    source=f"Message Turn {i}",
                    method=block_stats.get("method", "prune"),
                    orig_chars=block_stats["orig_chars"],
                    final_chars=block_stats["final_chars"],
                    duration_ms=dur_ms,
                )

    stats["modified"] = modified
    return body, stats


def _fmt_tokens(n: int) -> str:
    """Formats a token count into a readable string (e.g. 1.2M, 345.1K)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_time(ms: float) -> str:
    """Formats milliseconds into readable time (e.g. 23m15s, 1.4s, 240ms)."""
    if ms >= 60_000:
        mins = int(ms // 60_000)
        secs = int((ms % 60_000) // 1_000)
        return f"{mins}m{secs:02d}s"
    elif ms >= 1_000:
        return f"{ms / 1_000:.1f}s"
    return f"{int(ms)}ms"


PROVIDER_LABELS = {
    "anthropic": "Anthropic API (claude)",
    "opencode": "OpenCode Go",
    "openai": "OpenAI-compatible",
    "local": "Local llama-server",
    "gemini": "Google Antigravity (agy)",
    "antigravity": "Google Antigravity (agy)",
}


def provider_label(endpoint: str) -> str:
    return PROVIDER_LABELS.get(endpoint, endpoint)


def _provider_breakdown(cur) -> List[Tuple[str, int, int, int, float]]:
    cur.execute("""
        SELECT endpoint, COUNT(*), SUM(saved_tokens), SUM(orig_tokens), SUM(duration_ms)
        FROM compression_events
        GROUP BY endpoint
        ORDER BY SUM(saved_tokens) DESC
    """)
    return [(e, c or 0, s or 0, o or 0, d or 0.0) for e, c, s, o, d in cur.fetchall()]


def generate_gain_report(db_path: str = COMPRESSION_DB) -> str:
    """Generates an ASCII token savings report modeled after rtk gain."""
    if not os.path.exists(db_path):
        return "No compression events recorded yet. Run agent sessions with LOCALAGENT_COMPRESS=1."

    conn = _get_db(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*),
            SUM(orig_tokens),
            SUM(final_tokens),
            SUM(saved_tokens),
            SUM(duration_ms)
        FROM compression_events
    """)
    row = cur.fetchone()
    if not row or not row[0] or row[0] == 0:
        conn.close()
        return "No compression events recorded yet. Run agent sessions with LOCALAGENT_COMPRESS=1."

    total_events, total_orig, total_final, total_saved, total_duration = row
    total_events = total_events or 0
    total_orig = total_orig or 0
    total_final = total_final or 0
    total_saved = total_saved or 0
    total_duration = total_duration or 0.0

    avg_duration = (total_duration / total_events) if total_events else 0.0
    overall_pct = (total_saved / total_orig * 100.0) if total_orig else 0.0

    # Meter bar (24 blocks)
    meter_blocks = int(round((overall_pct / 100.0) * 24))
    meter_blocks = max(0, min(24, meter_blocks))
    meter_bar = "█" * meter_blocks + "░" * (24 - meter_blocks)

    cur.execute("""
        SELECT
            source,
            COUNT(*),
            SUM(saved_tokens),
            SUM(orig_tokens),
            SUM(duration_ms)
        FROM compression_events
        GROUP BY source
        ORDER BY SUM(saved_tokens) DESC
    """)
    breakdown = cur.fetchall()
    providers = _provider_breakdown(cur)
    conn.close()

    max_saved_tool = breakdown[0][2] if breakdown and breakdown[0][2] else 1

    lines = [
        "LocalAgent Token Savings (Global Scope)",
        "═" * 60,
        "",
        f"Total events:      {total_events}",
        f"Input tokens:      {_fmt_tokens(total_orig)}",
        f"Output tokens:     {_fmt_tokens(total_final)}",
        f"Tokens saved:      {_fmt_tokens(total_saved)} ({overall_pct:.1f}%)",
        f"Total exec time:   {_fmt_time(total_duration)} (avg {_fmt_time(avg_duration)})",
        f"Efficiency meter:  {meter_bar} {overall_pct:.1f}%",
        "",
        "By Provider (where the saved tokens would have been billed)",
        "─" * 72,
        f"  {'Provider':<28} {'Count':>5}   {'Saved':>7}    {'Avg%':>5}   {'Share':>5}  {'Impact':<10}",
        "─" * 72,
    ]
    for ep, count, saved, orig, duration in providers:
        pct = (saved / orig * 100.0) if orig else 0.0
        share = (saved / total_saved * 100.0) if total_saved else 0.0
        blocks = max(0, min(10, int(round(share / 10))))
        label = provider_label(ep)
        label = (label[:25] + "...") if len(label) > 28 else label
        lines.append(
            f"  {label:<28} {count:5d}   {_fmt_tokens(saved):>7}   {pct:5.1f}%  {share:5.1f}%  {'█' * blocks + '░' * (10 - blocks)}"
        )
    lines += [
        "",
        "By Tool / Source",
        "─" * 72,
        f"  #  {'Source / Tool':<25} {'Count':>5}   {'Saved':>7}    {'Avg%':>5}    {'Time':>5}  {'Impact':<10}",
        "─" * 72,
    ]

    for rank, (src, count, saved, orig, duration) in enumerate(breakdown, 1):
        saved = saved or 0
        orig = orig or 0
        duration = duration or 0.0
        pct = (saved / orig * 100.0) if orig else 0.0
        impact_blocks = int(round((saved / max_saved_tool) * 10)) if max_saved_tool > 0 else 0
        impact_blocks = max(0, min(10, impact_blocks))
        impact_bar = "█" * impact_blocks + "░" * (10 - impact_blocks)
        truncated_src = (src[:22] + "...") if len(src) > 25 else src
        lines.append(
            f" {rank:2d}.  {truncated_src:<25} {count:5d}   {_fmt_tokens(saved):>7}   {pct:5.1f}%   {_fmt_time(duration):>5}  {impact_bar}"
        )

    lines.append("─" * 72)
    return "\n".join(lines)


def get_compression_stats_json(db_path: str = COMPRESSION_DB) -> Dict[str, Any]:
    """Returns compression metrics as a dictionary for API/JSON consumers."""
    if not os.path.exists(db_path):
        return {"total_events": 0, "tokens_saved": 0, "savings_pct": 0.0, "by_tool": []}

    conn = _get_db(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*), SUM(orig_tokens), SUM(final_tokens), SUM(saved_tokens), SUM(duration_ms)
        FROM compression_events
    """)
    total_events, total_orig, total_final, total_saved, total_duration = cur.fetchone()
    total_events = total_events or 0
    total_orig = total_orig or 0
    total_final = total_final or 0
    total_saved = total_saved or 0
    total_duration = total_duration or 0.0
    savings_pct = (total_saved / total_orig * 100.0) if total_orig else 0.0

    cur.execute("""
        SELECT source, COUNT(*), SUM(saved_tokens), SUM(orig_tokens), SUM(duration_ms)
        FROM compression_events
        GROUP BY source
        ORDER BY SUM(saved_tokens) DESC
    """)
    by_tool = [
        {
            "source": r[0],
            "count": r[1],
            "saved_tokens": r[2] or 0,
            "orig_tokens": r[3] or 0,
            "savings_pct": ((r[2] or 0) / (r[3] or 1) * 100.0),
            "duration_ms": r[4] or 0.0,
        }
        for r in cur.fetchall()
    ]
    by_provider = [
        {
            "provider": ep,
            "label": provider_label(ep),
            "count": c,
            "saved_tokens": sv,
            "orig_tokens": o,
            "savings_pct": round((sv / o * 100.0) if o else 0.0, 2),
            "duration_ms": round(d, 2),
        }
        for ep, c, sv, o, d in _provider_breakdown(cur)
    ]
    conn.close()
    return {
        "total_events": total_events,
        "orig_tokens": total_orig,
        "final_tokens": total_final,
        "tokens_saved": total_saved,
        "savings_pct": round(savings_pct, 2),
        "total_duration_ms": round(total_duration, 2),
        "by_tool": by_tool,
        "by_provider": by_provider,
    }


def reset_compression_stats(db_path: str = COMPRESSION_DB):
    """Clears all recorded compression events."""
    if os.path.exists(db_path):
        conn = _get_db(db_path)
        with conn:
            conn.execute("DELETE FROM compression_events")
        conn.close()


# ---------------------------------------------------------------------------
# Gemini / Google Cloud Code Assist (agy, gemini-cli) payloads
# ---------------------------------------------------------------------------

def _gemini_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Cloud Code Assist wraps the GenerateContentRequest as body["request"];
    the raw Gemini API sends it at the top level. Return whichever holds
    "contents"."""
    req = body.get("request")
    if isinstance(req, dict) and "contents" in req:
        return req
    return body


def _gemini_last_user_text(contents: List[Dict[str, Any]]) -> str:
    for c in reversed(contents):
        if c.get("role") != "user":
            continue
        for p in c.get("parts") or []:
            if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
                return p["text"][:500]
    return ""


def _prune_gemini_value(value: Any, query: str, source: str, stats: Dict[str, Any], depth: int = 0, provider: str = "gemini") -> Any:
    """Recursively prune oversized string leaves inside a functionResponse
    payload. Returns the (possibly) rewritten value."""
    if isinstance(value, str):
        if len(value) <= MAX_CHARS:
            return value
        t0 = time.perf_counter()
        pruned, bstats = prune_text(value, query=query)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        if not bstats.get("pruned"):
            return value
        stats["modified"] = True
        stats["pruned_blocks"] += 1
        stats["chars_saved"] += bstats["orig_chars"] - bstats["final_chars"]
        record_compression_event(
            endpoint=provider,
            source=source,
            method=bstats.get("method", "prune"),
            orig_chars=bstats["orig_chars"],
            final_chars=bstats["final_chars"],
            duration_ms=dur_ms,
        )
        return pruned
    if depth >= 3:
        return value
    if isinstance(value, dict):
        return {k: _prune_gemini_value(v, query, source, stats, depth + 1, provider) for k, v in value.items()}
    if isinstance(value, list):
        return [_prune_gemini_value(v, query, source, stats, depth + 1, provider) for v in value]
    return value


def compress_gemini_payload(body: Dict[str, Any], provider: str = "gemini") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compress a Gemini GenerateContentRequest (raw API or Cloud Code Assist
    wrapped). Tool outputs live in parts[].functionResponse.response; every
    oversized string leaf in there (outside the most recent turns) is pruned.
    """
    stats: Dict[str, Any] = {"modified": False, "pruned_blocks": 0, "chars_saved": 0}
    if not is_compression_enabled():
        return body, stats

    req = _gemini_request(body)
    contents = req.get("contents")
    if not isinstance(contents, list) or not contents:
        return body, stats

    query = _gemini_last_user_text(contents)

    # Keep the last N user turns untouched (same policy as the other formats).
    user_idx = [i for i, c in enumerate(contents) if isinstance(c, dict) and c.get("role") == "user"]
    cutoff = user_idx[-PRESERVE_RECENT_TURNS] if len(user_idx) >= PRESERVE_RECENT_TURNS else 0

    for i, content in enumerate(contents):
        if i >= cutoff or not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            fr = part.get("functionResponse")
            if not isinstance(fr, dict) or "response" not in fr:
                continue
            source = fr.get("name") or "functionResponse"
            fr["response"] = _prune_gemini_value(fr["response"], query, source, stats, provider=provider)

    return body, stats
