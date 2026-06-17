#!/usr/bin/env python3
"""Tests for proxy.py dataset logging helpers.

Stdlib unittest only — no pytest, no extra deps. Run with:
    python3 -m unittest localagent.test_proxy -v
"""
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main(verbosity=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
