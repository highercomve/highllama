#!/usr/bin/env python3
"""Tests for proxy.py dataset logging helpers.

Stdlib unittest only — no pytest, no extra deps. Run with:
    python3 -m unittest localagent.test_proxy -v
"""
import http.client
import json
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import uuid
from unittest import mock

from http.server import ThreadingHTTPServer

# Make the localagent package importable when run from the repo root or the
# localagent dir.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

# Force a stable model id so we don't depend on a running llama-server.
os.environ.setdefault("LLAMA_MODEL", "test-model")

import proxy as ap  # noqa: E402


def _datasets_available():
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


class TestInjectThinking(unittest.TestCase):
    def test_default_enables_thinking(self):
        body = {"model": "test-model"}
        ap._inject_thinking(body)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})

    def test_preserves_explicit_false(self):
        body = {"model": "test-model", "chat_template_kwargs": {"enable_thinking": False}}
        ap._inject_thinking(body)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_preserves_existing_kwargs(self):
        body = {"model": "test-model", "chat_template_kwargs": {"reasoning_effort": "high"}}
        ap._inject_thinking(body)
        self.assertEqual(body["chat_template_kwargs"], {"reasoning_effort": "high", "enable_thinking": True})

    def test_disable_forces_off(self):
        with mock.patch.object(ap, "DISABLE_THINKING", True):
            body = {"model": "test-model", "chat_template_kwargs": {"reasoning_effort": "high"}}
            ap._inject_thinking(body)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})


class TestFlattenContent(unittest.TestCase):
    def test_string_passthrough(self):
        self.assertEqual(ap._flatten_content("hello"), "hello")

    def test_none_and_empty(self):
        self.assertEqual(ap._flatten_content(None), "")
        self.assertEqual(ap._flatten_content([]), "")

    def test_text_block(self):
        self.assertEqual(
            ap._flatten_content([{"type": "text", "text": "hi"}]),
            "hi",
        )

    def test_tool_use_block(self):
        content = [
            {
                "type": "tool_use",
                "id": "toolu_x",
                "name": "Read",
                "input": {"file_path": "/x"},
            }
        ]
        out = ap._flatten_content(content)
        self.assertIn("<tool_call>", out)
        self.assertIn('"name": "Read"', out)
        self.assertIn('"file_path": "/x"', out)

    def test_tool_result_block_string(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_x",
                "content": "file contents",
            }
        ]
        out = ap._flatten_content(content)
        self.assertIn("<tool_response>", out)
        self.assertIn("file contents", out)

    def test_tool_result_block_list(self):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_x",
                "content": [
                    {"type": "text", "text": "line 1\n"},
                    {"type": "text", "text": "line 2\n"},
                ],
            }
        ]
        out = ap._flatten_content(content)
        self.assertIn("line 1", out)
        self.assertIn("line 2", out)

    def test_image_block(self):
        self.assertEqual(
            ap._flatten_content([{"type": "image", "source": {...}}]),
            "[image]",
        )

    def test_mixed_blocks(self):
        content = [
            {"type": "text", "text": "before "},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}},
            {"type": "text", "text": " after"},
        ]
        out = ap._flatten_content(content)
        self.assertIn("before ", out)
        self.assertIn("<tool_call>", out)
        self.assertIn(" after", out)

    def test_thinking_blocks(self):
        content = [
            {"type": "thinking", "thinking": "why not this"},
            {"type": "redacted_thinking", "data": "opaque_signature"},
            {"type": "text", "text": "final answer"}
        ]
        out = ap._flatten_content(content)
        self.assertIn("<thinking>\nwhy not this\n</thinking>", out)
        self.assertIn("<redacted_thinking>\nopaque_signature\n</redacted_thinking>", out)
        self.assertIn("final answer", out)


class TestBuildDatasetItem(unittest.TestCase):
    def _build(self, **overrides):
        body = {
            "model": "test-model",
            "system": "you are a helper",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "hello"}
                ]},
            ],
        }
        body.update(overrides)
        response_content = [{"type": "text", "text": "world"}]
        return ap._build_dataset_item(body, response_content)

    def test_basic_shape(self):
        item = self._build()
        self.assertIsNotNone(item)
        self.assertEqual(item["model"], "test-model")
        self.assertEqual(item["system"], "you are a helper")
        # messages includes the appended assistant
        self.assertEqual(len(item["messages"]), 3)
        self.assertEqual(item["messages"][-1]["role"], "assistant")
        self.assertEqual(item["messages"][-1]["content"], [{"type": "text", "text": "world"}])
        self.assertFalse(item["has_tool_calls"])

    def test_messages_flat_includes_system(self):
        item = self._build()
        self.assertEqual(item["messages_flat"][0], {"role": "system", "content": "you are a helper"})
        # user / assistant (response) follow
        self.assertEqual([m["role"] for m in item["messages_flat"]], ["system", "user", "assistant", "assistant"])

    def test_sharegpt_tool_role_becomes_function(self):
        body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "read foo"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
        item = ap._build_dataset_item(body, [])
        # The tool_result is wrapped in a user message; check it stays as user
        # in the sharegpt output (the user role contains the tool_result)
        froms = [m["from"] for m in item["conversations"]]
        self.assertIn("human", froms)
        self.assertIn("gpt", froms)
        # No raw "tool" should appear — we map it to function only when the
        # message itself is role=tool, which Claude Code doesn't emit
        self.assertNotIn("tool", froms)
        self.assertTrue(item["has_tool_calls"])

    def test_explicit_tool_role_message_maps_to_function(self):
        # Synthetic case: a tool-role message at the top level (e.g. an
        # OpenAI-style conversation fed through a custom translator).
        body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}
                ]},
                {"role": "tool", "content": "result string", "tool_call_id": "t1"},
            ],
        }
        item = ap._build_dataset_item(body, [])
        froms = [m["from"] for m in item["conversations"]]
        self.assertIn("function", froms)
        self.assertNotIn("tool", froms)
        self.assertTrue(item["has_tool_calls"])

    def test_tools_only_kept_when_truthy(self):
        item_empty = self._build(tools=[])
        self.assertNotIn("tools", item_empty)

        item_real = self._build(tools=[{"name": "Bash", "description": "shell", "input_schema": {}}])
        self.assertIn("tools", item_real)
        self.assertEqual(item_real["tools"][0]["name"], "Bash")

    def test_has_tool_calls_detects_use_and_result(self):
        # tool_use
        body_a = {"model": "m", "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]}
        ]}
        self.assertTrue(ap._build_dataset_item(body_a, [])["has_tool_calls"])
        # tool_result
        body_b = {"model": "m", "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}
        ]}
        self.assertTrue(ap._build_dataset_item(body_b, [])["has_tool_calls"])
        # plain text only
        body_c = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        self.assertFalse(ap._build_dataset_item(body_c, [])["has_tool_calls"])

    def test_system_as_list_of_blocks(self):
        body = {
            "model": "m",
            "system": [{"type": "text", "text": "part1 "}, {"type": "text", "text": "part2"}],
            "messages": [],
        }
        item = ap._build_dataset_item(body, [])
        self.assertEqual(item["system"], "part1 part2")
        self.assertEqual(item["conversations"][0], {"from": "system", "value": "part1 part2"})

    def test_openai_messages_flat_reasoning(self):
        # OpenAI style message with reasoning_content
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "final reply", "reasoning_content": "some thought"}
        ]
        flat = ap._openai_messages_flat(messages)
        self.assertEqual(flat[0], {"role": "user", "content": "hello"})
        self.assertEqual(flat[1]["role"], "assistant")
        self.assertIn("<thinking>\nsome thought\n</thinking>", flat[1]["content"])
        self.assertIn("final reply", flat[1]["content"])


class TestResolveDatasetPaths(unittest.TestCase):
    def test_explicit_env_wins(self):
        with mock.patch.dict(os.environ, {"LLAMA_PROXY_DATASET": "/tmp/x/dataset.jsonl"}, clear=False):
            jsonl, db = ap.resolve_dataset_paths()
        self.assertEqual(jsonl, "/tmp/x/dataset.jsonl")
        self.assertEqual(db, "/tmp/x/dataset.db")

    def test_explicit_env_no_extension(self):
        with mock.patch.dict(os.environ, {"LLAMA_PROXY_DATASET": "/tmp/x/myset"}, clear=False):
            jsonl, db = ap.resolve_dataset_paths()
        self.assertEqual(jsonl, "/tmp/x/myset")
        self.assertEqual(db, "/tmp/x/myset.db")

    def test_falls_back_to_log_dir(self):
        env = {"LLAMA_PROXY_DATASET": "", "LLAMA_PROXY_LOG": "/var/log/lp/dataset.proxy.log"}
        with mock.patch.dict(os.environ, env, clear=False):
            # We must rebuild LOG_PATH because module captured it at import.
            with mock.patch.object(ap, "LOG_PATH", "/var/log/lp/dataset.proxy.log"):
                jsonl, db = ap.resolve_dataset_paths()
        self.assertEqual(jsonl, "/var/log/lp/dataset.jsonl")
        self.assertEqual(db, "/var/log/lp/dataset.db")

    def test_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(ap, "LOG_PATH", None):
                jsonl, db = ap.resolve_dataset_paths()
        self.assertTrue(jsonl.endswith("dataset.jsonl"))
        self.assertTrue(db.endswith("dataset.db"))


class TestEnqueueDataset(unittest.TestCase):
    def setUp(self):
        # Use a tiny bounded queue so we can force overflow deterministically.
        self._saved_queue = ap._dataset_queue
        self._saved_max = ap._DATASET_QUEUE_MAX
        ap._dataset_queue = ap.queue.Queue(maxsize=2)
        ap._dataset_dropped = 0

    def tearDown(self):
        ap._dataset_queue = self._saved_queue
        ap._dataset_dropped = 0

    def test_basic_enqueue(self):
        ap._enqueue_dataset({"type": "x", "n": 1})
        self.assertEqual(ap._dataset_queue.qsize(), 1)

    def test_drop_oldest_on_overflow(self):
        ap._enqueue_dataset({"type": "x", "n": 1})
        ap._enqueue_dataset({"type": "x", "n": 2})
        ap._enqueue_dataset({"type": "x", "n": 3})  # queue is now [2, 3], 1 was dropped
        self.assertEqual(ap._dataset_queue.qsize(), 2)
        self.assertEqual(ap._dataset_dropped, 1)
        items = [ap._dataset_queue.get_nowait() for _ in range(2)]
        self.assertEqual([i["n"] for i in items], [2, 3])


class TestSSEResponseReconstructor(unittest.TestCase):
    def test_drops_done_marker(self):
        r = ap.SSEAssistantResponseReconstructor()
        r.feed_line(b"event: message_stop\ndata: [DONE]\n\n")
        self.assertEqual(r.get_content(), [])

    def test_reconstructs_text_and_tool_blocks(self):
        r = ap.SSEAssistantResponseReconstructor()
        r.feed_line(b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n')
        r.feed_line(b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}\n\n')
        r.feed_line(b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}\n\n')
        r.feed_line(b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"Bash","input":{}}}\n\n')
        r.feed_line(b'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"cmd\\":\\"ls\\"}"}}\n\n')
        r.feed_line(b'data: [DONE]\n\n')
        out = r.get_content()
        self.assertEqual(out[0], {"type": "text", "text": "hello world"})
        self.assertEqual(out[1]["name"], "Bash")
        self.assertEqual(out[1]["input"], {"cmd": "ls"})

    def test_swallows_malformed_lines(self):
        r = ap.SSEAssistantResponseReconstructor()
        r.feed_line(b"not a data line\n")
        r.feed_line(b"data: {not json\n")
        r.feed_line(b"data: [DONE]\n")
        self.assertEqual(r.get_content(), [])

    def test_reconstructs_thinking_blocks(self):
        r = ap.SSEAssistantResponseReconstructor()
        # Test standard thinking
        r.feed_line(b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}\n\n')
        r.feed_line(b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thinking flow "}}\n\n')
        r.feed_line(b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"continues"}}\n\n')
        r.feed_line(b'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc"}}\n\n')
        
        # Test redacted thinking
        r.feed_line(b'data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":""}}\n\n')
        r.feed_line(b'data: {"type":"content_block_delta","index":1,"delta":{"type":"redacted_thinking_delta","data":"opaque_123"}}\n\n')
        
        # Test final text response
        r.feed_line(b'data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":"hello"}}\n\n')
        
        out = r.get_content()
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], {"type": "thinking", "thinking": "thinking flow continues", "signature": "sig_abc"})
        self.assertEqual(out[1], {"type": "redacted_thinking", "data": "opaque_123"})
        self.assertEqual(out[2], {"type": "text", "text": "hello"})


class TestEnsureSchema(unittest.TestCase):
    def test_creates_table_and_adds_column_on_legacy_db(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db")
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE dataset_calls ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "timestamp TEXT, model TEXT, system TEXT, "
                "messages TEXT, messages_flat TEXT, conversations TEXT, tools TEXT)"
            )
            conn.commit()
            ap._ensure_schema(conn)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(dataset_calls)").fetchall()]
            self.assertIn("has_tool_calls", cols)
            ver = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(ver, 1)
            conn.close()

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "t.db")
            conn = sqlite3.connect(db)
            ap._ensure_schema(conn)
            ap._ensure_schema(conn)  # second call must not blow up
            cols = [r[1] for r in conn.execute("PRAGMA table_info(dataset_calls)").fetchall()]
            self.assertIn("has_tool_calls", cols)
            conn.close()


class TestWriterIntegration(unittest.TestCase):
    """End-to-end: enqueue -> writer thread picks up -> row appears in SQLite."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._db = os.path.join(self._tmp.name, "dataset.db")
        self._jsonl = os.path.join(self._tmp.name, "dataset.jsonl")
        self._env_patches = [
            mock.patch.dict(os.environ, {"LLAMA_PROXY_DATASET": self._jsonl}, clear=False),
        ]
        for p in self._env_patches:
            p.start()
        # Reset queue + writer state for isolation
        self._saved_queue = ap._dataset_queue
        self._saved_dropped = ap._dataset_dropped
        ap._dataset_queue = ap.queue.Queue(maxsize=1000)
        ap._dataset_dropped = 0
        ap.start_dataset_writer()

    def tearDown(self):
        # Best-effort: ask writer to exit by poisoning the queue, then wait
        try:
            ap._dataset_queue.put_nowait(None)
        except Exception:
            pass
        # Writer thread is a daemon; give it a moment, then move on
        time.sleep(0.1)
        for p in self._env_patches:
            p.stop()
        ap._dataset_queue = self._saved_queue
        ap._dataset_dropped = self._saved_dropped
        self._tmp.cleanup()

    def _wait_for_row(self, sql, params=()):
        deadline = time.time() + 5
        while time.time() < deadline:
            if os.path.exists(self._db):
                conn = sqlite3.connect(self._db, timeout=1)
                try:
                    try:
                        cur = conn.execute(sql, params)
                        rows = cur.fetchall()
                    except sqlite3.OperationalError:
                        # writer hasn't created the table yet
                        rows = []
                finally:
                    conn.close()
                if rows:
                    return rows
            time.sleep(0.05)
        return None

    def test_parsed_task_round_trip(self):
        body = {
            "model": "test-model",
            "system": "be brief",
            "messages": [
                {"role": "user", "content": "say hi"},
            ],
        }
        response_content = [{"type": "text", "text": "hi back"}]
        ap._enqueue_dataset({"type": "parsed", "body": body, "response_content": response_content})

        rows = self._wait_for_row(
            "SELECT model, has_tool_calls, conversations, tools FROM dataset_calls"
        )
        self.assertIsNotNone(rows, "writer never produced a row")
        row = rows[0]
        self.assertEqual(row[0], "test-model")
        self.assertEqual(row[1], 0)
        conv = json.loads(row[2])
        self.assertEqual(conv[-1], {"from": "gpt", "value": "hi back"})
        self.assertIsNone(row[3])

    def test_tool_use_sets_has_tool_calls(self):
        body = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "run ls"},
            ],
        }
        response_content = [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}}
        ]
        ap._enqueue_dataset({"type": "parsed", "body": body, "response_content": response_content})

        rows = self._wait_for_row("SELECT has_tool_calls FROM dataset_calls")
        self.assertIsNotNone(rows, "writer never produced a row")
        self.assertEqual(rows[0][0], 1)

    def test_openai_stream_reasoning_integration(self):
        body = {
            "model": "test-model-reasoning",
            "messages": [{"role": "user", "content": "solve math"}],
        }
        # Simulate OpenAI SSE stream chunks containing reasoning_content
        raw_stream = (
            b'data: {"choices": [{"delta": {"reasoning_content": "let us think about "}}]}\n'
            b'data: {"choices": [{"delta": {"reasoning_content": "the solution"}}]}\n'
            b'data: {"choices": [{"delta": {"content": " 42"}}]}\n'
            b"data: [DONE]\n"
        )
        ap._enqueue_dataset({
            "type": "openai_stream",
            "body": body,
            "raw_bytes": raw_stream,
        })
        
        rows = self._wait_for_row("SELECT messages_flat, conversations FROM dataset_calls WHERE model = 'test-model-reasoning'")
        self.assertIsNotNone(rows, "writer never produced a row")
        
        # Check messages_flat
        flat = json.loads(rows[0][0])
        self.assertEqual(flat[-1]["role"], "assistant")
        self.assertIn("<thinking>\nlet us think about the solution\n</thinking>\n 42", flat[-1]["content"])
        
        # Check conversations
        conv = json.loads(rows[0][1])
        self.assertEqual(conv[-1]["from"], "gpt")
        self.assertIn("<thinking>\nlet us think about the solution\n</thinking>\n 42", conv[-1]["value"])


@unittest.skipUnless(
    _datasets_available(),
    "huggingface `datasets` not installed; install to run the round-trip test",
)
class TestExporterHFRoundTrip(unittest.TestCase):
    """End-to-end: write items via the writer, export them with
    `export_dataset.py`, then load with `datasets.load_dataset` to confirm
    the output is compatible with the HF datasets training ecosystem."""

    @classmethod
    def setUpClass(cls):
        from datasets import load_dataset  # noqa: F401  (import-time check)
        cls._load_dataset = staticmethod(load_dataset)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._db = os.path.join(self._tmp.name, "ds.db")
        self._jsonl = os.path.join(self._tmp.name, "ds.jsonl")
        self._env_patches = [
            mock.patch.dict(os.environ, {"LLAMA_PROXY_DATASET": self._jsonl}, clear=False),
        ]
        for p in self._env_patches:
            p.start()

        self._saved_queue = ap._dataset_queue
        self._saved_dropped = ap._dataset_dropped
        ap._dataset_queue = ap.queue.Queue(maxsize=1000)
        ap._dataset_dropped = 0
        ap.start_dataset_writer()
        self._seed_items()
        # Drain
        deadline = time.time() + 5
        n = 0
        while time.time() < deadline:
            if os.path.exists(self._db):
                conn = sqlite3.connect(self._db, timeout=1)
                try:
                    try:
                        n = conn.execute("SELECT COUNT(*) FROM dataset_calls").fetchone()[0]
                    except sqlite3.OperationalError:
                        # writer hasn't created the table yet
                        n = 0
                finally:
                    conn.close()
            if n >= 3:
                break
            time.sleep(0.05)
        self.assertGreaterEqual(n, 3, "writer didn't ingest seeded items")

    def tearDown(self):
        try:
            ap._dataset_queue.put_nowait(None)
        except Exception:
            pass
        time.sleep(0.1)
        for p in self._env_patches:
            p.stop()
        ap._dataset_queue = self._saved_queue
        ap._dataset_dropped = self._saved_dropped
        self._tmp.cleanup()

    def _seed_items(self):
        # 1. plain chat
        ap._enqueue_dataset({
            "type": "parsed",
            "body": {"model": "m", "system": "be brief", "messages": [{"role": "user", "content": "hi"}]},
            "response_content": [{"type": "text", "text": "hello"}],
        })
        # 2. tool use
        ap._enqueue_dataset({
            "type": "parsed",
            "body": {"model": "m", "messages": [{"role": "user", "content": "list"}]},
            "response_content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"cmd": "ls"}}],
        })
        # 3. tool result + assistant follow-up
        ap._enqueue_dataset({
            "type": "parsed",
            "body": {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "what's in foo?"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/foo"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "stuff"}]},
                ],
            },
            "response_content": [{"type": "text", "text": "foo contains stuff"}],
        })

    def _run_exporter(self, fmt):
        out_path = os.path.join(self._tmp.name, f"out.{fmt}")
        import importlib
        if "export_dataset" in sys.modules:
            importlib.reload(sys.modules["export_dataset"])
        else:
            import export_dataset  # noqa: F401
        old_argv = sys.argv
        sys.argv = ["export", "--db-path", self._db, "--format", fmt, "--output", out_path]
        try:
            sys.modules["export_dataset"].main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
        return out_path

    def test_sharegpt_loads_with_datasets(self):
        out = self._run_exporter("sharegpt")
        ds = self._load_dataset("json", data_files=out)["train"]
        self.assertIn("conversations", ds.column_names)
        self.assertEqual(len(ds), 3)
        # First item: system in conversations + top-level
        first = ds[0]
        conv = first["conversations"]
        self.assertEqual(conv[0], {"from": "system", "value": "be brief"})
        self.assertEqual(conv[1], {"from": "human", "value": "hi"})
        self.assertEqual(conv[2], {"from": "gpt", "value": "hello"})
        self.assertEqual(first["system"], "be brief")

    def test_sharegpt_tool_call_loads(self):
        out = self._run_exporter("sharegpt")
        ds = self._load_dataset("json", data_files=out)["train"]
        # Second item: assistant message contains the <tool_call> marker
        tool_item = ds[1]
        last = tool_item["conversations"][-1]
        self.assertEqual(last["from"], "gpt")
        self.assertIn("<tool_call>", last["value"])
        self.assertIn('"name": "Bash"', last["value"])

    def test_openai_loads_with_datasets(self):
        out = self._run_exporter("openai")
        ds = self._load_dataset("json", data_files=out)["train"]
        self.assertIn("messages", ds.column_names)
        self.assertEqual(len(ds), 3)
        msgs = ds[0]["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": "be brief"})
        self.assertEqual(msgs[1], {"role": "user", "content": "hi"})
        self.assertEqual(msgs[2], {"role": "assistant", "content": "hello"})

    def test_jsonl_loads_with_datasets(self):
        out = self._run_exporter("jsonl")
        ds = self._load_dataset("json", data_files=out)["train"]
        for col in ("conversations", "messages", "messages_flat"):
            self.assertIn(col, ds.column_names)
        # Spot-check that the original Anthropic block structure is preserved
        # in the jsonl `messages` field
        m = ds[1]["messages"]
        self.assertEqual(m[-1]["content"][0]["type"], "tool_use")
        self.assertEqual(m[-1]["content"][0]["name"], "Bash")


class TestOpenCodeRouting(unittest.TestCase):
    def test_model_matching_with_prefix(self):
        # Known openai model in provider dict
        model, proto = ap.get_opencode_model_and_protocol("opencode-go/kimi-k3")
        self.assertEqual(model, "kimi-k3")
        self.assertEqual(proto, "openai")

        model, proto = ap.get_opencode_model_and_protocol("opencode-go/gpt-5.6-luna")
        self.assertEqual(model, "gpt-5.6-luna")
        self.assertEqual(proto, "openai")

        model, proto = ap.get_opencode_model_and_protocol("opencode-go/kimi-k2.7-code")
        self.assertEqual(model, "kimi-k2.7-code")
        self.assertEqual(proto, "openai")

        # Known anthropic model in provider dict
        model, proto = ap.get_opencode_model_and_protocol("opencode-qwen3.7-max")
        self.assertEqual(model, "qwen3.7-max")
        self.assertEqual(proto, "anthropic")

        # Newer models in the provider dict
        for name, expected_proto in [
            ("glm-5.3", "openai"), ("glm-5.3-flash", "openai"),
            ("grok-4.5", "openai"), ("grok-4.6", "openai"),
            ("hy3", "openai"), ("hy3-preview", "openai"), ("hy4-preview", "openai"),
            ("kimi-k2.5", "openai"), ("longcat-2.0", "openai"),
            ("mimo-v2-omni", "openai"), ("mimo-v2-pro", "openai"),
            ("muse-spark-1.2-contributor", "openai"),
            ("deepseek-v4-flash-vision-exp", "openai"),
            ("qwen3.5-plus", "anthropic"),
            ("qwen3.8-flash", "anthropic"), ("qwen3.8-max", "anthropic"),
        ]:
            model, proto = ap.get_opencode_model_and_protocol(f"opencode-go/{name}")
            self.assertEqual(model, name)
            self.assertEqual(proto, expected_proto)

        # Responses-only models are flagged as such
        for name in ("gpt-5.6-luna", "grok-4.5", "grok-4.6"):
            self.assertIn(name, ap.OPENCODE_RESPONSES_ONLY)
            self.assertIn(name, ap.OPENCODE_GO_PROVIDER)

        # Unknown model starting with opencode- (should fallback to openai by default)
        model, proto = ap.get_opencode_model_and_protocol("opencode-go/new-unknown-model")
        self.assertEqual(model, "new-unknown-model")
        self.assertEqual(proto, "openai")

        # Unknown model starting with opencode- containing qwen (should fallback to anthropic)
        model, proto = ap.get_opencode_model_and_protocol("opencode-new-qwen-model")
        self.assertEqual(model, "new-qwen-model")
        self.assertEqual(proto, "anthropic")

        # No prefix but exists in dict
        model, proto = ap.get_opencode_model_and_protocol("kimi-k2.7-code")
        self.assertEqual(model, "kimi-k2.7-code")
        self.assertEqual(proto, "openai")

        # No prefix and doesn't exist in dict
        model, proto = ap.get_opencode_model_and_protocol("other-random-model")
        self.assertIsNone(model)
        self.assertIsNone(proto)

    def test_load_dotenv(self):
        # Test that load_dotenv reads k=v lines
        with tempfile.TemporaryDirectory() as d:
            dotenv_path = os.path.join(d, ".env")
            with open(dotenv_path, "w") as f:
                f.write("TEST_ENV_VAR_X = val_x\n")
                f.write("# comment line\n")
                f.write("TEST_ENV_VAR_Y='val_y'\n")
            
            with mock.patch("os.path.abspath") as mock_abs:
                mock_abs.return_value = os.path.join(d, "proxy.py")
                ap.load_dotenv()
                
            self.assertEqual(os.environ.get("TEST_ENV_VAR_X"), "val_x")
            self.assertEqual(os.environ.get("TEST_ENV_VAR_Y"), "val_y")


# ---------------------------------------------------------------------------
# Anthropic passthrough tests (live ephemeral server + patched upstreams)
# ---------------------------------------------------------------------------

class _FakeUpstreamResponse:
    def __init__(self, status=200, body=b"{}", content_type="application/json",
                 lines=None, headers=None):
        self.status = status
        self._body = body
        self._lines = list(lines or [])
        self._ct = content_type
        self._headers = list(headers or [])

    def getheader(self, name, default=None):
        if name.lower() == "content-type":
            return self._ct
        for k, v in self._headers:
            if k.lower() == name.lower():
                return v
        return default

    def getheaders(self):
        return [("Content-Type", self._ct)] + self._headers

    def read(self):
        return self._body

    def readline(self):
        return self._lines.pop(0) if self._lines else b""


class _FakeUpstreamConn:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def request(self, method, path, body=None, headers=None):
        self.requests.append({"method": method, "path": path,
                              "body": body, "headers": dict(headers or {})})

    def getresponse(self):
        return self.response

    def close(self):
        pass


class TestAnthropicPassthrough(unittest.TestCase):
    """End-to-end proxy dispatch with faked upstream Anthropic/llama-server."""

    def setUp(self):
        self._saved_queue = ap._dataset_queue
        self._saved_dropped = ap._dataset_dropped
        ap._dataset_queue = queue.Queue(maxsize=1000)
        ap._dataset_dropped = 0

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), ap.Handler)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        self._patches = []
        self._anthropic_conn_factory = lambda response: _FakeUpstreamConn(response)
        self._upstream_conn_factory = lambda response: _FakeUpstreamConn(response)
        self._opencode_conn_factory = lambda response: _FakeUpstreamConn(response)
        self._patch_conn("anthropic_conn", self._anthropic_conn_factory)
        self._patch_conn("upstream_conn", self._upstream_conn_factory)
        self._patch_conn("opencode_conn", self._opencode_conn_factory)

    def tearDown(self):
        self._server.shutdown()
        self._thread.join(timeout=5)
        for p in self._patches:
            p.stop()
        ap._dataset_queue = self._saved_queue
        ap._dataset_dropped = self._saved_dropped

    def _patch_conn(self, name, factory):
        """Patch ap.<name> so each call returns a new fake using factory(response)."""
        calls = []

        def _make_conn(response=None):
            conn = factory(response)
            calls.append(conn)
            return conn

        p = mock.patch.object(ap, name, _make_conn)
        p.start()
        self._patches.append(p)
        attr_name = f"_{name}_calls"
        setattr(self, attr_name, calls)
        return _make_conn

    def _set_anthropic_response(self, response):
        def _factory(_response=None):
            conn = self._anthropic_conn_factory(response)
            self._anthropic_conn_calls.append(conn)
            return conn
        for p in self._patches:
            if p.attribute == "anthropic_conn":
                p.stop()
                self._patches.remove(p)
                break
        p = mock.patch.object(ap, "anthropic_conn", _factory)
        p.start()
        self._patches.append(p)

    def _set_upstream_response(self, response):
        def _factory(_response=None):
            conn = self._upstream_conn_factory(response)
            self._upstream_conn_calls.append(conn)
            return conn
        for p in self._patches:
            if p.attribute == "upstream_conn":
                p.stop()
                self._patches.remove(p)
                break
        p = mock.patch.object(ap, "upstream_conn", _factory)
        p.start()
        self._patches.append(p)

    def _set_opencode_response(self, response):
        def _factory(_response=None):
            conn = self._opencode_conn_factory(response)
            self._opencode_conn_calls.append(conn)
            return conn
        for p in self._patches:
            if p.attribute == "opencode_conn":
                p.stop()
                self._patches.remove(p)
                break
        p = mock.patch.object(ap, "opencode_conn", _factory)
        p.start()
        self._patches.append(p)

    def _request(self, method, path, body=None, headers=None):
        headers = dict(headers or {})
        if body is not None and "Content-Length" not in headers:
            if isinstance(body, bytes):
                headers["Content-Length"] = str(len(body))
            else:
                body = body.encode("utf-8")
                headers["Content-Length"] = str(len(body))
        conn = http.client.HTTPConnection("127.0.0.1", self._port)
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, resp_headers, resp.read()
        finally:
            conn.close()

    def _pop_anthropic_request(self):
        self.assertTrue(self._anthropic_conn_calls, "no anthropic conn created")
        return self._anthropic_conn_calls[-1].requests[-1]

    def _pop_upstream_request(self):
        self.assertTrue(self._upstream_conn_calls, "no upstream conn created")
        return self._upstream_conn_calls[-1].requests[-1]

    def test_post_v1_messages_passthrough_sync(self):
        payload = json.dumps({"model": "claude-3-opus-20240229", "max_tokens": 1024,
                              "messages": [{"role": "user", "content": "hi"}]}).encode()
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{"id":"msg_1"}'))
        status, headers, body = self._request(
            "POST", "/v1/messages", body=payload,
            headers={"x-api-key": "sk-test", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"id":"msg_1"}')
        req = self._pop_anthropic_request()
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual(req["body"], payload)
        forwarded = {k.lower(): v for k, v in req["headers"].items()}
        self.assertEqual(forwarded.get("x-api-key"), "sk-test")
        self.assertEqual(forwarded.get("host"), ap.AN_HOST)
        for hop in ("connection", "transfer-encoding", "accept-encoding", "keep-alive", "proxy-connection"):
            self.assertNotIn(hop, forwarded)

        task = ap._dataset_queue.get(timeout=1)
        self.assertEqual(task["type"], "passthrough_sync")
        self.assertEqual(task["body_raw"], payload)
        self.assertEqual(task["response_bytes"], b'{"id":"msg_1"}')

    def test_post_v1_messages_passthrough_stream(self):
        payload = json.dumps({"model": "claude-3-opus-20240229", "stream": True,
                              "messages": [{"role": "user", "content": "hi"}]}).encode()
        lines = [b"event: ping\n", b"data: {}\n", b"\n"]
        self._set_anthropic_response(_FakeUpstreamResponse(
            content_type="text/event-stream", lines=list(lines)))
        status, headers, body = self._request(
            "POST", "/v1/messages", body=payload,
            headers={"x-api-key": "sk-test", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, b"".join(lines))

        task = ap._dataset_queue.get(timeout=1)
        self.assertEqual(task["type"], "passthrough_stream")
        self.assertEqual(task["body_raw"], payload)
        self.assertEqual(task["raw_bytes"], b"".join(lines))

    def test_get_unknown_v1_path_passthrough(self):
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{"data":[]}'))
        status, headers, body = self._request("GET", "/v1/organizations")
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"data":[]}')
        req = self._pop_anthropic_request()
        self.assertEqual(req["method"], "GET")
        self.assertEqual(req["path"], "/v1/organizations")

    def test_get_sse_long_poll_streams(self):
        lines = [b"event: rc-event\n", b"data: {\"x\":1}\n", b"\n", b"event: rc-event\n", b"data: {\"x\":2}\n", b"\n"]
        self._set_anthropic_response(_FakeUpstreamResponse(
            content_type="text/event-stream", lines=list(lines)))
        status, headers, body = self._request("GET", "/v1/some-rc-stream")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"".join(lines))
        self.assertIn("text/event-stream", headers.get("content-type", "").lower())

    def test_delete_api_path_passthrough(self):
        self._set_anthropic_response(_FakeUpstreamResponse(status=204, body=b""))
        status, headers, body = self._request("DELETE", "/api/some-rc-resource")
        self.assertEqual(status, 204)
        req = self._pop_anthropic_request()
        self.assertEqual(req["method"], "DELETE")
        self.assertEqual(req["path"], "/api/some-rc-resource")

    def test_put_and_patch_dispatch(self):
        for method in ("PUT", "PATCH"):
            with self.subTest(method=method):
                self._set_anthropic_response(_FakeUpstreamResponse(body=b'{"ok":true}'))
                status, headers, body = self._request(method, "/api/x", body=b'{"a":1}',
                                                      headers={"Content-Type": "application/json"})
                self.assertEqual(status, 200)
                req = self._pop_anthropic_request()
                self.assertEqual(req["method"], method)
                self.assertEqual(req["path"], "/api/x")
                self.assertEqual(req["body"], b'{"a":1}')

    def test_non_api_path_still_404(self):
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{"x":1}'))
        status, headers, body = self._request("DELETE", "/notapi")
        self.assertEqual(status, 404)
        self.assertEqual(self._anthropic_conn_calls, [])

    def test_post_non_json_body_api_passthrough(self):
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{"received":true}'))
        status, headers, body = self._request(
            "POST", "/api/telemetry", body=b"\x00binary",
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(status, 200)
        req = self._pop_anthropic_request()
        self.assertEqual(req["body"], b"\x00binary")
        self.assertEqual(req["headers"].get("Content-Type"), "application/octet-stream")
        self.assertTrue(ap._dataset_queue.empty())

    def test_drop_headers_not_forwarded(self):
        payload = json.dumps({"model": "claude-3-opus-20240229",
                              "messages": [{"role": "user", "content": "hi"}]}).encode()
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{}'))
        self._request(
            "POST", "/v1/messages", body=payload,
            headers={
                "x-api-key": "sk-test",
                "connection": "keep-alive",
                "transfer-encoding": "chunked",
                "accept-encoding": "gzip",
                "keep-alive": "timeout=5",
                "proxy-connection": "keep-alive",
                "Content-Type": "application/json",
            },
        )
        req = self._pop_anthropic_request()
        forwarded = {k.lower() for k in req["headers"].keys()}
        for h in ("connection", "transfer-encoding", "accept-encoding", "keep-alive", "proxy-connection"):
            self.assertNotIn(h, forwarded, f"hop-by-hop header {h!r} leaked")
        self.assertEqual(req["headers"].get("Host"), ap.AN_HOST)

    def test_local_model_still_routes_local(self):
        payload = json.dumps({"model": "local-llama", "stream": False,
                              "messages": [{"role": "user", "content": "hi"}]}).encode()
        self._set_upstream_response(_FakeUpstreamResponse(body=json.dumps({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode()))
        status, headers, body = self._request("POST", "/v1/messages", body=payload,
                                              headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"], [{"type": "text", "text": "hi"}])
        self.assertEqual(self._anthropic_conn_calls, [])

    def test_local_count_tokens_still_local(self):
        payload = json.dumps({"model": "local-llama",
                              "messages": [{"role": "user", "content": "hello world"}]}).encode()
        status, headers, body = self._request("POST", "/v1/messages/count_tokens", body=payload,
                                              headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn("input_tokens", data)
        self.assertGreater(data["input_tokens"], 0)
        self.assertEqual(self._anthropic_conn_calls, [])

    def test_opencode_stream_adds_missing_finish_reason(self):
        payload = json.dumps({
            "model": "opencode-go/glm-5.2",
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }).encode()
        lines = [
            b'data: {"choices":[{"delta":{"content":"Ready."},"finish_reason":null}]}\n',
            b'\n',
            b'data: [DONE]\n',
            b'\n',
        ]
        self._set_opencode_response(_FakeUpstreamResponse(
            content_type="text/event-stream", lines=lines,
        ))

        status, headers, body = self._request(
            "POST", "/v1/chat/completions", body=payload,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("content-type", ""))
        data_lines = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
        self.assertEqual(json.loads(data_lines[0])["choices"][0]["delta"]["content"], "Ready.")
        self.assertEqual(json.loads(data_lines[1])["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data_lines[2], b"[DONE]")

    def test_opencode_stream_preserves_finish_reason(self):
        payload = json.dumps({
            "model": "opencode-go/glm-5.2",
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }).encode()
        lines = [
            b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"length"}]}\n',
            b'data: [DONE]\n',
        ]
        self._set_opencode_response(_FakeUpstreamResponse(
            content_type="text/event-stream", lines=lines,
        ))

        status, _, body = self._request(
            "POST", "/v1/chat/completions", body=payload,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(status, 200)
        data_lines = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
        self.assertEqual(len(data_lines), 2)
        self.assertEqual(json.loads(data_lines[0])["choices"][0]["finish_reason"], "length")

    def test_opencode_stream_adds_finish_reason_at_eof(self):
        payload = json.dumps({
            "model": "glm-5.2",
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }).encode()
        lines = [
            b'data: {"id":"gen-1","object":"chat.completion.chunk",'
            b'"created":1,"model":"gpt-5.6-luna",'
            b'"choices":[{"delta":{"content":"done"},"finish_reason":null}]}\n',
            b'data: {"id":"gen-1","object":"chat.completion.chunk",'
            b'"created":1,"model":"gpt-5.6-luna","choices":[],'
            b'"usage":{"completion_tokens":1}}\n',
        ]
        self._set_opencode_response(_FakeUpstreamResponse(
            content_type="text/event-stream", lines=lines,
        ))

        status, _, body = self._request(
            "POST", "/v1/chat/completions", body=payload,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(status, 200)
        data_lines = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
        terminal = json.loads(data_lines[-1])
        self.assertEqual(terminal["id"], "gen-1")
        self.assertEqual(terminal["model"], "gpt-5.6-luna")
        self.assertEqual(terminal["choices"][0]["finish_reason"], "stop")

    def test_opencode_stream_error_relayed_verbatim(self):
        # Upstream error bodies are plain JSON; no synthesized SSE chunk may be appended.
        payload = json.dumps({
            "model": "opencode-go/glm-5.2",
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }).encode()
        err = b'{"type":"error","error":{"type":"error","message":"Internal server error"}}'
        self._set_opencode_response(_FakeUpstreamResponse(status=500, body=err))

        status, headers, body = self._request(
            "POST", "/v1/chat/completions", body=payload,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(status, 500)
        self.assertEqual(body, err)
        self.assertNotIn(b"finish_reason", body)

    def test_opencode_responses_only_model_rejected_on_chat_completions(self):
        payload = json.dumps({
            "model": "opencode-go/gpt-5.6-luna",
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }).encode()
        self._set_opencode_response(_FakeUpstreamResponse(status=500, body=b'{}'))

        status, _, body = self._request(
            "POST", "/v1/chat/completions", body=payload,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(status, 400)
        data = json.loads(body)
        self.assertIn("/v1/responses", data["error"]["message"])
        self.assertEqual(self._opencode_conn_calls if hasattr(self, "_opencode_conn_calls") else [], [])

    def test_opencode_responses_only_model_rejected_on_messages(self):
        # grok-4.6 is Responses-only: /v1/messages must fail fast with an
        # actionable error instead of an opaque upstream 500.
        payload = json.dumps({
            "model": "opencode-go/grok-4.6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "test"}],
        }).encode()
        self._set_opencode_response(_FakeUpstreamResponse(status=500, body=b'{}'))

        status, _, body = self._request(
            "POST", "/v1/messages", body=payload,
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(status, 400)
        data = json.loads(body)
        self.assertIn("/v1/responses", data["error"]["message"])
        self.assertEqual(self._opencode_conn_calls if hasattr(self, "_opencode_conn_calls") else [], [])

    def test_api_post_not_captured_in_dataset(self):
        payload = json.dumps({"model": "claude-3-opus-20240229"}).encode()
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{}'))
        status, headers, body = self._request("POST", "/api/foo", body=payload,
                                              headers={"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertTrue(ap._dataset_queue.empty())

    def test_post_json_array_body_api_passthrough(self):
        """JSON bodies that are not objects (e.g. telemetry arrays) relay verbatim."""
        payload = json.dumps([{"event": "x"}]).encode()
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{"received":true}'))
        status, headers, body = self._request(
            "POST", "/api/telemetry", body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"received":true}')
        req = self._pop_anthropic_request()
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["path"], "/api/telemetry")
        self.assertEqual(req["body"], payload)
        self.assertTrue(ap._dataset_queue.empty())

    def test_post_non_dict_json_non_api_path_400(self):
        payload = json.dumps(["not", "an", "object"]).encode()
        self._set_anthropic_response(_FakeUpstreamResponse(body=b'{}'))
        status, headers, body = self._request(
            "POST", "/notapi", body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(self._anthropic_conn_calls, [])

    def test_get_model_by_id_passthrough(self):
        """GET /v1/models/{id} (Anthropic 'Get a Model') relays verbatim, not the merged list."""
        upstream = json.dumps({"id": "claude-3-opus-20240229", "type": "model"}).encode()
        self._set_anthropic_response(_FakeUpstreamResponse(body=upstream))
        status, headers, body = self._request(
            "GET", "/v1/models/claude-3-opus-20240229",
            headers={"x-api-key": "sk-test"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, upstream)
        req = self._pop_anthropic_request()
        self.assertEqual(req["method"], "GET")
        self.assertEqual(req["path"], "/v1/models/claude-3-opus-20240229")

    def test_get_models_list_still_merged(self):
        """GET /v1/models (with or without query) keeps the merged local+upstream list."""
        upstream = json.dumps({"data": [
            {"id": "claude-3-opus-20240229", "type": "model"},
        ]}).encode()
        for path in ("/v1/models", "/v1/models?limit=5"):
            with self.subTest(path=path):
                self._set_anthropic_response(_FakeUpstreamResponse(body=upstream))
                status, headers, body = self._request(
                    "GET", path, headers={"x-api-key": "sk-test"})
                self.assertEqual(status, 200)
                data = json.loads(body)
                self.assertEqual(data["object"], "list")
                ids = [m["id"] for m in data["data"]]
                self.assertEqual(ids[0], ap.LOCAL_ALIAS)
                self.assertIn("claude-3-opus-20240229", ids)

    def test_upstream_response_headers_forwarded(self):
        payload = json.dumps({"model": "claude-3-opus-20240229",
                              "messages": [{"role": "user", "content": "hi"}]}).encode()
        self._set_anthropic_response(_FakeUpstreamResponse(
            body=b'{"id":"msg_1"}',
            headers=[
                ("request-id", "req_abc123"),
                ("anthropic-ratelimit-requests-remaining", "42"),
                ("retry-after", "7"),
                ("Content-Length", "999"),        # hop-by-hop: must be dropped
                ("Transfer-Encoding", "chunked"),  # hop-by-hop: must be dropped
            ],
        ))
        status, headers, body = self._request(
            "POST", "/v1/messages", body=payload,
            headers={"x-api-key": "sk-test", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, b'{"id":"msg_1"}')
        self.assertEqual(headers.get("request-id"), "req_abc123")
        self.assertEqual(headers.get("anthropic-ratelimit-requests-remaining"), "42")
        self.assertEqual(headers.get("retry-after"), "7")
        # Response is delimited by connection close, so length framing is dropped.
        self.assertNotEqual(headers.get("content-length"), "999")
        self.assertNotIn("transfer-encoding", headers)
        self.assertEqual(headers.get("connection"), "close")

    # --- x-opencode-session attribution -------------------------------------

    def _oc_headers(self):
        self.assertTrue(self._opencode_conn_calls, "no opencode conn created")
        return self._opencode_conn_calls[-1].requests[-1]["headers"]

    def test_opencode_session_from_claude_code_metadata(self):
        sid = "77222f22-4593-4400-9b09-a080028c5fe5"
        payload = json.dumps({
            "model": "opencode-go/glm-5.2", "max_tokens": 64,
            "metadata": {"user_id": json.dumps({"device_id": "d", "account_uuid": "a", "session_id": sid})},
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()
        self._set_opencode_response(_FakeUpstreamResponse(
            body=b'{"choices":[{"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}]}'))
        status, _, _ = self._request(
            "POST", "/v1/messages", body=payload,
            headers={"Content-Type": "application/json", "User-Agent": "claude-cli/2.1.259 (external, cli)"},
        )
        self.assertEqual(status, 200)
        h = self._oc_headers()
        self.assertEqual(h.get("x-opencode-session"), sid)
        self.assertEqual(h.get("x-opencode-client"), "claude-code")

    def test_opencode_session_anthropic_passthrough(self):
        sid = "11111111-2222-4333-8444-555555555555"
        payload = json.dumps({
            "model": "opencode-qwen3.7-max", "max_tokens": 64,
            "metadata": {"user_id": "user_abc_account_9f0e_session_%s" % sid},
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()
        self._set_opencode_response(_FakeUpstreamResponse(body=b'{"id":"msg_1"}'))
        status, _, _ = self._request(
            "POST", "/v1/messages", body=payload,
            headers={"Content-Type": "application/json", "x-api-key": "k"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(self._oc_headers().get("x-opencode-session"), sid)

    def test_opencode_session_passes_through_client_header(self):
        payload = json.dumps({
            "model": "opencode-go/glm-5.2",
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()
        self._set_opencode_response(_FakeUpstreamResponse(body=b'{"choices":[]}'))
        self._request(
            "POST", "/v1/chat/completions", body=payload,
            headers={"Content-Type": "application/json",
                     "x-opencode-session": "pi-sess-1", "x-opencode-client": "pi"},
        )
        h = self._oc_headers()
        self.assertEqual(h.get("x-opencode-session"), "pi-sess-1")
        self.assertEqual(h.get("x-opencode-client"), "pi")

    def test_opencode_session_from_codex_session_id_header(self):
        payload = json.dumps({"model": "opencode-go/gpt-5.6-luna", "input": "hi"}).encode()
        self._set_opencode_response(_FakeUpstreamResponse(body=b'{"id":"resp_1"}'))
        self._request(
            "POST", "/v1/responses", body=payload,
            headers={"Content-Type": "application/json", "session_id": "codex-42",
                     "User-Agent": "codex_cli_rs/0.149.1"},
        )
        h = self._oc_headers()
        self.assertEqual(h.get("x-opencode-session"), "codex-42")
        self.assertEqual(h.get("x-opencode-client"), "codex")

    def test_opencode_session_fingerprint_stable_across_turns(self):
        def _send(messages):
            payload = json.dumps({"model": "opencode-go/glm-5.2", "messages": messages}).encode()
            self._set_opencode_response(_FakeUpstreamResponse(body=b'{"choices":[]}'))
            self._request("POST", "/v1/chat/completions", body=payload,
                          headers={"Content-Type": "application/json", "User-Agent": "curl/8.7"})
            return self._oc_headers()

        first = [{"role": "system", "content": "s"}, {"role": "user", "content": "open"}]
        h1 = _send(first)
        h2 = _send(first + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "more"}])
        h3 = _send([{"role": "user", "content": "different opener"}])
        self.assertEqual(h1["x-opencode-session"], h2["x-opencode-session"])
        self.assertNotEqual(h1["x-opencode-session"], h3["x-opencode-session"])
        self.assertEqual(h1["x-opencode-client"], "highllama")
        uuid.UUID(h1["x-opencode-session"])  # well-formed


if __name__ == "__main__":
    unittest.main(verbosity=2)
