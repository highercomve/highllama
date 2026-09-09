#!/usr/bin/env python3
"""
Unit tests for localagent/compressor.py.
Stdlib unittest only — run with:
    PYTHONPATH=localagent python3 -m unittest localagent/test_compressor.py -v
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import compressor


class TestCompressor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_spill = compressor.SPILL_DIR
        compressor.SPILL_DIR = self.temp_dir

    def tearDown(self):
        compressor.SPILL_DIR = self.orig_spill
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_clean_terminal_noise(self):
        dirty = "\x1b[31mError:\x1b[0m Failed\r\nLine 2\r\n\n\n\nLine 3"
        cleaned = compressor.clean_terminal_noise(dirty)
        self.assertNotIn("\x1b[31m", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Error: Failed\nLine 2\n\nLine 3", cleaned)

    def test_minify_json(self):
        obj = {"name": "test", "items": [1, 2, 3], "nested": {"key": "val"}}
        formatted = json.dumps(obj, indent=4)
        minified = compressor.minify_json_if_applicable(formatted)
        self.assertEqual(minified, json.dumps(obj, separators=(",", ":")))
        # Non-JSON remains intact
        self.assertEqual(compressor.minify_json_if_applicable("plain text"), "plain text")

    def test_chunk_text(self):
        text = "\n".join(f"line {i}" for i in range(100))
        chunks = compressor.chunk_text(text, chunk_lines=35, overlap=5)
        self.assertTrue(len(chunks) >= 3)
        self.assertEqual(chunks[0][0], 1)
        self.assertEqual(chunks[0][1], 35)

    def test_cosine_similarity(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(compressor.cosine_similarity(v1, v2), 1.0)
        self.assertAlmostEqual(compressor.cosine_similarity(v1, v3), 0.0)

    def test_spill_content(self):
        content = "hello secret world"
        h, path = compressor.spill_content(content, spill_dir=self.temp_dir)
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), content)

    def test_prune_head_tail_fallback(self):
        long_content = "\n".join(f"data row {i}" for i in range(200))
        # With no query, should fallback to head/tail
        pruned, stats = compressor.prune_text(long_content, query="", max_lines=50, max_chars=500)
        self.assertTrue(stats["pruned"])
        self.assertEqual(stats["method"], "head_tail")
        self.assertIn("data row 0", pruned)
        self.assertIn("data row 199", pruned)
        self.assertIn("localagent truncated", pruned)
        self.assertTrue(os.path.exists(stats["spill_path"]))

    @mock.patch("compressor.fetch_embeddings")
    def test_prune_semantic_embeddings(self, mock_embed):
        # 100 lines: lines 0-40 generic, lines 40-70 rate limiting, lines 70-100 generic
        lines = [f"generic config line {i}" for i in range(40)]
        lines += [f"rate limiting logic item {i}" for i in range(30)]
        lines += [f"generic footer line {i}" for i in range(30)]
        text = "\n".join(lines)

        # Mock embeddings: query matches middle chunk
        def fake_embed(texts, embed_base=None, timeout=None):
            # texts[0] is query.
            # Return high similarity vector for chunk containing "rate limiting"
            res = []
            for t in texts:
                if "rate limiting" in t:
                    res.append([1.0, 0.0, 0.0])
                else:
                    res.append([0.0, 1.0, 0.0])
            return res

        mock_embed.side_effect = fake_embed

        pruned, stats = compressor.prune_text(text, query="rate limiting", max_lines=50, max_chars=500)
        self.assertTrue(stats["pruned"])
        self.assertEqual(stats["method"], "semantic_embedding")
        self.assertIn("semantic compression", pruned)
        self.assertIn("rate limiting logic item", pruned)

    def test_compress_anthropic_payload(self):
        big_tool_result = "\n".join(f"log line {i}: output details" for i in range(150))
        body = {
            "model": "claude-3-7-sonnet-20250219",
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "messages": [
                {"role": "user", "content": "Fix the rate limit error in server.py"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "server.py"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big_tool_result}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "I see the problem."}]},
                # Turn within preserve window (last 2 turns)
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}]},
            ],
        }

        compressed, stats = compressor.compress_anthropic_payload(body)
        self.assertTrue(stats["modified"])
        self.assertEqual(stats["pruned_blocks"], 1)
        # Thinking budget must remain 100% untouched by default
        self.assertFalse(stats.get("thinking_dampened", False))
        self.assertEqual(compressed["thinking"]["budget_tokens"], 8000)

        # Verify opt-in thinking dampening only occurs if explicitly enabled
        with mock.patch.object(compressor, "THINKING_DAMPEN", True):
            opt_in_compressed, opt_in_stats = compressor.compress_anthropic_payload(body)
            self.assertTrue(opt_in_stats.get("thinking_dampened"))
            self.assertEqual(opt_in_compressed["thinking"]["budget_tokens"], 1500)

        # Old tool result should be pruned (either via live semantic embeddings or head/tail fallback)
        old_result = compressed["messages"][2]["content"][0]["content"]
        self.assertTrue(
            "localagent semantic compression" in old_result or "localagent truncated" in old_result
        )

        # Preserved recent tool result should be untouched
        recent_result = compressed["messages"][4]["content"][0]["content"]
        self.assertEqual(recent_result, "ok")

    def test_gain_report(self):
        db_path = os.path.join(self.temp_dir, "test_compression.db")
        # Record sample events
        compressor.record_compression_event(
            endpoint="anthropic",
            source="Read",
            method="semantic_embedding",
            orig_chars=40000,
            final_chars=8000,
            duration_ms=150.0,
            db_path=db_path,
        )
        compressor.record_compression_event(
            endpoint="anthropic",
            source="Bash",
            method="head_tail",
            orig_chars=20000,
            final_chars=4000,
            duration_ms=5.0,
            db_path=db_path,
        )

        report = compressor.generate_gain_report(db_path=db_path)
        self.assertIn("LocalAgent Token Savings", report)
        self.assertIn("Efficiency meter:", report)
        self.assertIn("Read", report)
        self.assertIn("Bash", report)

        stats_json = compressor.get_compression_stats_json(db_path=db_path)
        self.assertEqual(stats_json["total_events"], 2)
        self.assertIn("By Provider", report)
        self.assertTrue(stats_json["by_provider"])
        self.assertIn("provider", stats_json["by_provider"][0])
        self.assertTrue(stats_json["tokens_saved"] > 0)


    def test_compress_gemini_payload(self):
        big = "\n".join(f"line {i} of tool output" for i in range(400))
        body = {
            "project": "p", "model": "gemini-3.8-flash",
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": "list the files"}]},
                    {"role": "model", "parts": [{"functionCall": {"name": "run_command", "args": {"cmd": "ls"}}}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "run_command", "response": {"output": big}}}]},
                    {"role": "user", "parts": [{"text": "now summarise"}]},
                    {"role": "model", "parts": [{"text": "ok"}]},
                    {"role": "user", "parts": [{"functionResponse": {"name": "run_command", "response": {"output": big}}}]},
                    {"role": "user", "parts": [{"text": "final question"}]},
                ]
            },
        }
        with mock.patch.object(compressor, "fetch_embeddings", return_value=None):
            out, stats = compressor.compress_gemini_payload(body)
        self.assertTrue(stats["modified"])
        self.assertEqual(stats["pruned_blocks"], 1)
        c = out["request"]["contents"]
        self.assertLess(len(c[2]["parts"][0]["functionResponse"]["response"]["output"]), len(big))
        # recent turns are preserved verbatim
        self.assertEqual(c[5]["parts"][0]["functionResponse"]["response"]["output"], big)
        # untouched when there is nothing large
        small = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        _, st = compressor.compress_gemini_payload(small)
        self.assertFalse(st["modified"])

if __name__ == "__main__":
    unittest.main()
