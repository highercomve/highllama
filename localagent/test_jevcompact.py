#!/usr/bin/env python3
"""Tests for jevcompact.py and its proxy hook.

Stdlib unittest only. Run with:
    python3 -m unittest localagent.test_jevcompact -v
"""
import http.client
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ.setdefault("LLAMA_MODEL", "test-model")

import jevcompact as jc  # noqa: E402
import proxy as ap  # noqa: E402

CC_PROMPT = ("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\n\n"
             "Your task is to create a detailed summary of the conversation so far, "
             "paying close attention to the user's explicit requests.")
PI_SYSTEM = ("You are a context summarization assistant. Your task is to read a "
             "conversation between a user and an AI assistant.")
PI_PROMPT = ("<conversation>\n[User]: fix the login bug\n\n"
             "[Assistant tool calls]: read(path=\"auth.py\")\n\n"
             "[Tool result]: def login():\n\n    pass\n\n"
             "[Assistant]: Fixed it.\n</conversation>\n\n"
             "The messages above are a conversation to summarize.\n\n## Goal\n...")


def cc_body(stream=True):
    return {
        "model": "claude-opus-5",
        "stream": stream,
        "system": [{"type": "text", "text": "You are Claude Code."}],
        "messages": [
            {"role": "user", "content": "Fix the failing login test in auth.py"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "NOISE-LISTING a b c"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "auth.py"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "line 42: naive datetime"}]},
            {"role": "assistant", "content": "Found it: naive datetime on line 42."},
            {"role": "user", "content": "ok go on"},
            {"role": "assistant", "content": "Patched auth.py."},
            {"role": "user", "content": CC_PROMPT},
        ],
    }


class TestDetectExtract(unittest.TestCase):
    def test_claude_code(self):
        self.assertEqual(jc.detect(cc_body()), "claude-code")
        items, instr, system = jc.extract(cc_body(), "claude-code")
        self.assertEqual(len(items), 8)
        self.assertIn("Your task is to create", instr)
        self.assertEqual(items[2]["kind"], "tool")
        self.assertEqual(items[0]["kind"], "user")

    def test_opencode_openai_shape(self):
        body = {"model": "x", "messages": [
            {"role": "system", "content": "You are a context summarization agent. You are given..."},
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "content": "file body"},
            {"role": "user", "content": "Summarize using the template"},
        ]}
        self.assertEqual(jc.detect(body), "opencode")
        items, instr, _ = jc.extract(body, "opencode")
        self.assertEqual([i["kind"] for i in items], ["user", "assistant", "tool"])
        self.assertIn("[tool call] read", items[1]["text"])
        self.assertEqual(instr, "Summarize using the template")

    def test_pi(self):
        body = {"model": "x", "messages": [
            {"role": "system", "content": PI_SYSTEM},
            {"role": "user", "content": PI_PROMPT}]}
        self.assertEqual(jc.detect(body), "pi")
        items, instr, _ = jc.extract(body, "pi")
        self.assertEqual([i["kind"] for i in items], ["user", "assistant", "tool", "assistant"])
        self.assertIn("    pass", items[2]["text"])  # blank line inside a result kept together
        self.assertTrue(instr.startswith("The messages above"))

    def test_pi_responses_shape(self):
        body = {"model": "gpt-5.6-luna", "stream": True, "input": [
            {"role": "developer", "content": PI_SYSTEM},
            {"role": "user", "content": [{"type": "input_text", "text": PI_PROMPT}]}]}
        self.assertEqual(jc.detect(body), "pi")
        items, instr, system = jc.extract(body, "pi")
        self.assertEqual(len(items), 4)
        self.assertIn("context summarization assistant", system)
        self.assertIn("input", body)  # caller's body is not rewritten

    def test_responses_tool_items(self):
        body = {"instructions": "You are a context summarization agent.", "input": [
            {"role": "user", "content": "task"},
            {"type": "function_call", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "output": "file body"},
            {"role": "user", "content": "summarize"}]}
        self.assertEqual(jc.detect(body), "opencode")
        items, _, _ = jc.extract(body, "opencode")
        self.assertEqual([i["kind"] for i in items], ["user", "assistant", "tool"])

    def test_normal_requests_not_detected(self):
        self.assertIsNone(jc.detect({"messages": [{"role": "user", "content": "hi"}]}))
        body = cc_body()
        body["messages"][-1] = {"role": "user", "content": "keep going"}
        self.assertIsNone(jc.detect(body))
        self.assertIsNone(jc.detect({"messages": []}))


class TestSelect(unittest.TestCase):
    def test_keeps_users_tail_and_high_scores(self):
        items = [{"kind": "user", "text": "u0"}] + [
            {"kind": "tool", "text": "t%d" % i} for i in range(1, 12)]
        scores = [0.0] + [0.9 if i == 3 else 0.1 for i in range(1, 12)]
        keep = jc.select(items, scores, 10_000)
        self.assertEqual(keep, [0, 3, 6, 7, 8, 9, 10, 11])

    def test_budget_limits_optional_items(self):
        items = [{"kind": "tool", "text": "x" * 100} for _ in range(10)]
        keep = jc.select(items, [1.0] * 10, 650)
        self.assertEqual(keep, [4, 5, 6, 7, 8, 9])

    def test_render_marks_gaps(self):
        items = [{"kind": "tool", "text": "t%d" % i} for i in range(5)]
        out = jc.render(items, [1, 2])
        self.assertTrue(out.startswith("[... 1 messages omitted"))
        self.assertTrue(out.endswith("[... 2 messages omitted as not needed ...]"))


class TestFacts(unittest.TestCase):
    def test_files_only_from_tool_calls(self):
        items = [
            {"kind": "user", "text": "[User]: fix auth"},
            {"kind": "assistant", "text": '[Assistant tool calls]: read(path="src/auth.py")'},
            {"kind": "tool", "text": "[Tool result]: README.md install.sh auth.py"},
            {"kind": "assistant", "text": "[Assistant]: Fixed auth.py."},
        ]
        fs = jc.facts(items)
        self.assertEqual([f["fact"] for f in fs if f["kind"] == "file"],
                         ["The file `auth.py` was read, changed or discussed."])
        self.assertEqual([f["kind"] for f in fs if f["kind"] != "file"], ["request", "state"])


class _Fakes:
    """Fake llama-server (router mode) and fake Jev API."""

    def __init__(self):
        self.loaded = True
        self.jev_status = 200
        self.jev_calls = []
        self.local_calls = []
        fakes = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._send(200, {"data": [{"id": "chat", "status": {
                        "value": "loaded" if fakes.loaded else "unloaded",
                        "args": ["--ctx-size", "128k", "--parallel", "1"]}}]})
                self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v1/systemone":
                    fakes.jev_calls.append(body)
                    if fakes.jev_status != 200:
                        return self._send(fakes.jev_status, {"error": "boom"})
                    answers = {}
                    for qid, q in body["questions"].items():
                        noise = "NOISE" in q["instructions"]["item"]
                        answers[qid] = {"type": "noul", "noul": 0.05 if noise else 0.9}
                    return self._send(200, {"model": "jev-1.13.0", "answers": answers})
                if self.path == "/v1/chat/completions":
                    fakes.local_calls.append(body)
                    return self._send(200, {"choices": [{"finish_reason": "stop", "message": {
                        "content": "<summary>LOCAL SUMMARY</summary>"}}]})
                self._send(404, {})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]


class _PassthroughConn:
    """Records the raw bytes the proxy relays to Anthropic."""

    def __init__(self, sink):
        self.sink = sink

    def request(self, method, path, body=None, headers=None):
        self.sink.append(body)

    def getresponse(self):
        class R:
            status = 200
            lines = [b'{"relayed": true}']

            def getheader(self, name, default=None):
                return "application/json"

            def getheaders(self):
                return []

            def read(self, *a):
                return b'{"relayed": true}'

            def readline(self):
                return self.lines.pop(0) if self.lines else b""

            def read1(self, *a):
                return b""
        return R()

    def close(self):
        pass


class TestProxyHook(unittest.TestCase):
    def setUp(self):
        self.fakes = _Fakes()
        self.relayed = []
        self.proxy = ThreadingHTTPServer(("127.0.0.1", 0), ap.Handler)
        threading.Thread(target=self.proxy.serve_forever, daemon=True).start()
        self.patches = [
            mock.patch.dict(os.environ, {"LOCALAGENT_JEV_COMPACT": "1",
                                         "TYPESAFE_API_KEY": "test-key"}),
            mock.patch.object(jc, "JEV_BASE", self.fakes.base),
            mock.patch.object(ap, "LLAMA_BASE", self.fakes.base),
            mock.patch.object(ap, "anthropic_conn", lambda: _PassthroughConn(self.relayed)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.proxy.shutdown()
        self.fakes.server.shutdown()

    def _post(self, path, raw):
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy.server_address[1], timeout=10)
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read().decode()

    def test_fitting_conversation_skips_jev(self):
        status, text = self._post("/v1/messages", json.dumps(cc_body()))
        self.assertEqual(status, 200)
        self.assertIn("LOCAL SUMMARY", text)
        self.assertEqual(self.fakes.jev_calls, [])
        prompt = self.fakes.local_calls[0]["messages"][1]["content"]
        self.assertIn("NOISE-LISTING", prompt)  # everything fits, nothing dropped

    @mock.patch.object(jc, "KEEP_TAIL", 2)
    @mock.patch.object(jc, "fits", lambda items, budget: False)
    def test_claude_code_stream_served_locally(self):
        status, text = self._post("/v1/messages", json.dumps(cc_body()))
        self.assertEqual(status, 200)
        self.assertIn("event: message_start", text)
        self.assertIn("LOCAL SUMMARY", text)
        self.assertIn("event: message_stop", text)
        self.assertEqual(self.relayed, [])
        self.assertEqual(len(self.fakes.jev_calls), 1)
        prompt = self.fakes.local_calls[0]["messages"][1]["content"]
        self.assertNotIn("NOISE-LISTING", prompt)      # Jev dropped it
        self.assertIn("naive datetime", prompt)          # Jev kept it
        self.assertIn("Your task is to create", prompt)  # client's own instructions
        self.assertEqual(self.fakes.local_calls[0]["model"], "chat")

    def test_non_stream_anthropic(self):
        status, text = self._post("/v1/messages", json.dumps(cc_body(stream=False)))
        self.assertEqual(status, 200)
        data = json.loads(text)
        self.assertEqual(data["content"][0]["text"], "<summary>LOCAL SUMMARY</summary>")
        self.assertEqual(data["stop_reason"], "end_turn")

    def _assert_relayed_unchanged(self, raw):
        status, text = self._post("/v1/messages", raw)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text), {"relayed": True})
        self.assertEqual(self.relayed, [raw.encode()])
        self.assertEqual(self.fakes.local_calls, [])

    def test_local_model_unloaded_falls_back_untouched(self):
        self.fakes.loaded = False
        self._assert_relayed_unchanged(json.dumps(cc_body()))
        self.assertEqual(self.fakes.jev_calls, [])  # no Jev spend either

    @mock.patch.object(jc, "fits", lambda items, budget: False)
    def test_jev_failure_falls_back_untouched(self):
        self.fakes.jev_status = 500
        self._assert_relayed_unchanged(json.dumps(cc_body()))

    def test_llama_server_down_falls_back_untouched(self):
        with mock.patch.object(ap, "LLAMA_BASE", "http://127.0.0.1:9"):
            self._assert_relayed_unchanged(json.dumps(cc_body()))

    def test_disabled_falls_back_untouched(self):
        with mock.patch.dict(os.environ, {"LOCALAGENT_JEV_COMPACT": "0"}):
            self._assert_relayed_unchanged(json.dumps(cc_body()))
        self.assertEqual(self.fakes.jev_calls, [])

    def test_normal_request_untouched(self):
        body = cc_body()
        body["messages"][-1] = {"role": "user", "content": "keep going"}
        self._assert_relayed_unchanged(json.dumps(body))
        self.assertEqual(self.fakes.jev_calls, [])

    def test_pi_chat_completions(self):
        body = {"model": "opencode-go/kimi-k3", "messages": [
            {"role": "system", "content": PI_SYSTEM},
            {"role": "user", "content": PI_PROMPT}]}
        status, text = self._post("/v1/chat/completions", json.dumps(body))
        self.assertEqual(status, 200)
        data = json.loads(text)
        self.assertEqual(data["choices"][0]["message"]["content"],
                         "<summary>LOCAL SUMMARY</summary>")

        body["stream"] = True
        status, text = self._post("/v1/chat/completions", json.dumps(body))
        self.assertEqual(status, 200)
        self.assertIn("LOCAL SUMMARY", text)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))


class TestResponsesRoute(TestProxyHook):
    BODY = {"model": "opencode-go/gpt-5.6-luna", "input": [
        {"role": "developer", "content": PI_SYSTEM},
        {"role": "user", "content": [{"type": "input_text", "text": PI_PROMPT}]}]}

    def setUp(self):
        super().setUp()
        p = mock.patch.object(ap, "opencode_conn", lambda: _PassthroughConn(self.relayed))
        p.start()
        self.patches.append(p)

    def test_non_stream(self):
        status, text = self._post("/v1/responses", json.dumps(self.BODY))
        self.assertEqual(status, 200)
        data = json.loads(text)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["output"][0]["content"][0]["text"],
                         "<summary>LOCAL SUMMARY</summary>")
        self.assertEqual(self.relayed, [])

    def test_stream_event_sequence(self):
        status, text = self._post("/v1/responses", json.dumps(dict(self.BODY, stream=True)))
        self.assertEqual(status, 200)
        events = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ")]
        self.assertEqual([e["type"] for e in events], [
            "response.created", "response.output_item.added",
            "response.content_part.added", "response.output_text.delta",
            "response.output_text.done", "response.content_part.done",
            "response.output_item.done", "response.completed"])
        self.assertEqual(events[3]["delta"], "<summary>LOCAL SUMMARY</summary>")
        self.assertEqual([e["sequence_number"] for e in events], list(range(8)))

    def test_unloaded_relays_to_opencode(self):
        self.fakes.loaded = False
        status, text = self._post("/v1/responses", json.dumps(self.BODY))
        self.assertEqual(json.loads(text), {"relayed": True})
        sent = json.loads(self.relayed[0])
        self.assertEqual(sent["input"], self.BODY["input"])  # request content unchanged
        self.assertEqual(self.fakes.jev_calls, [])


if __name__ == "__main__":
    unittest.main()
