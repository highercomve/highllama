# Code review: `localagent` dataset logging

Scope: `localagent/proxy.py`, `localagent/export_dataset.py`,
`localagent/localagent.sh`, and the design doc `dataset_logging.md`. The proxy
is a stdlib-only Anthropic↔OpenAI translator that routes "local" model names to
`llama-server` and everything else to `api.anthropic.com`. The dataset
logging is meant to capture every prompt/answer pair so they can be exported as
ShareGPT/OpenAI/JSONL for fine-tuning.

The headline finding is that **local-model calls are not captured at all** —
the writer thread only consumes tasks the proxy pushes for passthrough
(remote) responses. The rest of the findings are correctness/perf issues
in the pipeline that does exist.

---

## 1. Critical — local-model traffic is not logged

`dataset_logging.md:6` calls this out as intentional ("to keep the session
logs clean"), but for a training dataset it's the opposite of what you want:
the local model is exactly the brain being fine-tuned/distilled. As written,
the SQLite DB contains only the Opus/remote calls.

Two places should enqueue to `_dataset_queue` and don't:

- **Non-streaming local** — `Handler.do_POST` builds `openai_to_anthropic(data,
  req_model)` and returns it via `self._json(200, ...)`. No
  `_dataset_queue.put_nowait(...)` afterwards
  (`proxy.py:966-969`).
- **Streaming local** — `_stream()` (called from the streaming branch of
  `do_POST`) already builds a `response_content` list mirroring the same
  content blocks it sends to the client
  (`proxy.py:1046-1163`). It's the perfect capture point — the
  `response_content` list contains the cleaned `text` + native + salvaged
  tool blocks in Anthropic form. After
  `self._w(sse("message_stop", ...))` (`proxy.py:1181`) you just
  need:
  ```python
  _dataset_queue.put_nowait({
      "type": "parsed",
      "body": body,
      "response_content": response_content,
  })
  ```
  `body` is already in scope from the caller (`proxy.py:978`).

The existing `parsed` task type in the writer thread
(`proxy.py:642-645`) is exactly shaped for this — no writer-side
changes required. Mirror with a `local_sync` task for the non-streaming
branch or reuse `parsed` for both.

---

## 2. Dead `bytes` vs `str` comparison

`SSEAssistantResponseReconstructor.feed_line()`
(`proxy.py:697`):
```python
payload = line[5:].strip()             # str
if not payload or payload == "[DONE]" or payload == b"[DONE]":
```
`payload` is a `str` by that point, so `payload == b"[DONE]"` can never
match. Harmless because the `str` check above it covers the case, but it's
dead code that confuses readers. Drop it.

---

## 3. SQLite — one connection per write, no tuning

`_write_to_sqlite()` (`proxy.py:592-613`) opens a new
`sqlite3.connect(db_path)` for every logged call. Under load that means:
per-call setup, fsync on commit, no prepared-statement cache.

Two changes, both in `_dataset_writer_thread` (the writer is single-threaded,
so a single connection is safe):

```python
import sqlite3
conn = sqlite3.connect(db_path, timeout=10)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA busy_timeout=10000")
# in the loop:
cursor = conn.cursor()
cursor.execute("INSERT INTO dataset_calls ...", (...))
conn.commit()
```

WAL also lets `localagent export` read the DB while the proxy is still
writing.

---

## 4. JSONL migration blocks proxy startup

`_init_sqlite_db()` (`proxy.py:534-589`) reads the entire legacy
JSONL into memory and inserts line-by-line **synchronously, before the
writer thread starts its `get()` loop**. A 1 GB JSONL = multi-minute proxy
downtime, plus peak memory of ~the full file.

Mitigations:

- Detect migration size up front; skip with a warning if > N MB; do it
  lazily in the writer loop with batching.
- Stream the JSONL with an explicit batch + `conn.commit()` every K rows.
- Run the migration **after** the writer has started (defer to first task)
  so the proxy is serving requests while the import grinds.

---

## 5. Queue is unbounded

`_dataset_queue = queue.Queue()` (`proxy.py:448`) — no `maxsize`.
If SQLite stalls (disk full, lock), the proxy thread buffers the whole
conversation history in memory via `put_nowait` succeeding forever. Use a
bounded queue with drop-oldest on overflow and a counter:

```python
_dataset_queue = queue.Queue(maxsize=10_000)
def _enqueue(task):
    try:
        _dataset_queue.put_nowait(task)
    except queue.Full:
        # drop oldest, push newest, increment a counter
        ...
```

This is the kind of bug that shows up as "proxy OOM killed itself" three
weeks in.

---

## 6. ShareGPT `tool` role is not standard

`_build_dataset_item()` (`proxy.py:507-516`):
```python
from_val = "human" if role == "user" else ("gpt" if role == "assistant" else role)
```
For `role == "tool"` the value is the literal string `"tool"`. Most
ShareGPT trainers (and the canonical schema used by `axolotl`,
`LLaMA-Factory`, `sharegpt4v` loaders) only know `human`/`gpt`/`system`/
`function`. Two reasonable fixes:

- Map `"tool"` → `"function"`.
- Keep the tool role but emit a structured tool-call field on the message
  instead of a sentinel `<tool_call>` string in `value`.

If you keep the sentinel, also fix `--has-tools` in
`export_dataset.py:117-119`, which substring-matches the literal
`"<tool_call>"` in `value` — easy to false-positive on any prompt that
mentions the word.

---

## 7. `--has-tools` is stringly-typed

`export_dataset.py:114-122` decides "has tools" by searching the flattened
text for `"<tool_call>"` or `"tool_use"`. That depends on
`_flatten_content()` always emitting those literal markers and never
appearing in user input. It works, but it's brittle. Cleaner: at write
time, store a structured `has_tool_calls` boolean in the schema
(`ALTER TABLE dataset_calls ADD COLUMN has_tools INTEGER`) and filter on
that. Costs ~1 byte per row, removes the heuristic.

---

## 8. `tools` serialization is inconsistent

`_build_dataset_item()` adds `"tools"` to the item only if truthy
(`proxy.py:526-527`). `_write_to_sqlite()` writes
`json.dumps(item.get("tools", []))` if the key is present, else `NULL`
(`proxy.py:607`). So an empty `tools: []` is stored as the string
`"[]"`, but a missing `tools` is stored as SQL `NULL`. Pick one (recommend
`NULL` for both — remove the `if tools:` guard or replace
`if "tools" in item` with `if item.get("tools")`).

---

## 9. Log spam on the hot path

Every successful insert logs `"Dataset writer: successfully logged call to
SQLite database!"` (`proxy.py:611`) and every task logs
`"Dataset writer thread: processing task type: ..."`
(`proxy.py:639`). At Opus token rates this is hundreds of
lines/sec into `LLAMA_PROXY_LOG` if it's set. Drop the success line,
demote task pickup to debug, keep only failure/skip lines.

---

## 10. `passthrough_stream` keeps the full response in RAM as a list of byte chunks

`_passthrough()` builds `raw_lines` by appending every SSE line byte string
(`proxy.py:804-815`). For a 200 k-token Opus response, this is
multiple MB held in memory until the response ends, then handed to the
worker, then re-iterated. Cheaper:

```python
buf = io.BytesIO()
while True:
    line = r.readline()
    if not line:
        break
    self.wfile.write(line)
    self.wfile.flush()
    if capture_response:
        buf.write(line)
# then:
buf.getvalue()  # one bytes object, one allocation
```

Or wrap the queue payload as `(body_raw, response_bytes)` and skip
`SSEAssistantResponseReconstructor` for the trivial case (the `raw_lines`
shape is only there to drive the reconstructor).

---

## 11. `SSEAssistantResponseReconstructor` swallows errors

`feed_line()` does `except Exception: pass`
(`proxy.py:723-724`) and `get_content()` silently returns
whatever it managed to build. If a malformed event breaks reconstruction
you get a half-empty assistant message in the dataset with no diagnostic.
Log the first few failures at debug level so future bugs are debuggable.

---

## 12. No retention policy

`dataset.db` grows without bound. For a long-lived training pipeline, add a
startup-time rotation (e.g. > 5 GB → rename to `dataset.<timestamp>.db`
and start fresh), exposed via `LLAMA_PROXY_DATASET_ROTATE`. Cheap
insurance.

---

## 13. Module-level side effects

`_writer = threading.Thread(...); _writer.start()`
(`proxy.py:683-684`) runs at `import` time, which means
importing `anthropic_proxy` for tests (or `python3 -c "import
anthropic_proxy"`) starts the writer thread, calls `detect_model()` (a
network call), and creates the DB. Wrap behind
`if __name__ == "__main__":` or a `main()` bootstrap so the module is
import-safe.

---

## 14. `localagent info` env handling

`cmd_info` (`localagent.sh:249-292`) reads `LLAMA_PROXY_DATASET` and
applies the `.jsonl`→`.db` substitution to find the DB, but the proxy
itself picks its `db_path` only via the writer's resolution
(`proxy.py:617-625`). If the env var differs between proxy
startup and `localagent info` (e.g. systemd unit vs. shell), they
disagree. Make `cmd_info` call `proxy_start`'s path-resolution helper, or
stash the resolved path in the pidfile env.

---

## 15. No tests

There are no test files for either the proxy or the exporter. The dataset
logging path is exactly the kind of thing that silently corrupts training
data if it mis-serializes tool results or drops messages. Minimum viable
tests to add:

- Round-trip: feed a synthetic Anthropic request + streaming response to
  `_passthrough` and assert the queued task reconstructs the same tool
  calls.
- `_build_dataset_item()` with a body containing `tool_use` +
  `tool_result` blocks + a system prompt with `cache_control` → assert
  the `conversations` round-trip is faithful.
- `_flatten_content()` for the four block types (text, tool_use,
  tool_result, image).
- Exporter: `--has-tools` and `--min-turns` filters.
- Migration: write a small JSONL → start the proxy → assert rows appear
  in SQLite.

---

## Suggested fix order

1. **Enqueue from the local streaming + non-streaming paths** (#1) —
   unblocks the actual training use case.
2. **SQLite tuning + single-connection writer** (#3, #5) — prevents OOM
   and slowdowns once the writer is actually being fed local traffic.
3. **WAL + bounded queue together** — same PR.
4. **Fix the dead `bytes`/`str` compare (#2) and the inconsistent `tools`
   serialization (#8)** — two-line fixes, removes confusion.
5. **Defer/block JSONL migration** (#4).
6. **Tests around `_build_dataset_item` + exporter filters** (#15).

The rest (#6, #7, #9–#14) are quality-of-life; useful but not blocking.
