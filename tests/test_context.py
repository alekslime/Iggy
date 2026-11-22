"""
Tests for project instructions and context budgeting (seacode.context).

Run with:  python3 -m pytest tests/test_context.py -v
       or: python3 -m unittest tests.test_context -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.context as context_module


class InstructionTests(unittest.TestCase):

    def test_no_files_no_instructions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                context_module.load_project_instructions(tmpdir), ""
            )
            self.assertEqual(
                context_module.build_system_prompt("BASE", tmpdir), "BASE"
            )

    def test_loads_all_files_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "CLAUDE.md").write_text("claude rules")
            Path(tmpdir, "SEACODE.md").write_text("sea rules")
            Path(tmpdir, "AGENTS.md").write_text("agent rules")
            text = context_module.load_project_instructions(tmpdir)
            prompt = context_module.build_system_prompt("BASE", tmpdir)

        self.assertLess(text.index("sea rules"), text.index("agent rules"))
        self.assertLess(text.index("agent rules"), text.index("claude rules"))
        self.assertTrue(prompt.startswith("BASE"))
        self.assertIn("PROJECT INSTRUCTIONS", prompt)

    def test_instructions_capped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "SEACODE.md").write_text(
                "x" * (context_module.MAX_INSTRUCTION_CHARS + 5000)
            )
            text = context_module.load_project_instructions(tmpdir)
        self.assertIn("truncated", text)
        self.assertLess(
            len(text), context_module.MAX_INSTRUCTION_CHARS + 100
        )

    def test_fresh_messages_replaces_system_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "SEACODE.md").write_text("be nice")
            messages = [
                {"role": "system", "content": "OLD"},
                {"role": "user", "content": "hi"},
            ]
            fresh = context_module.fresh_messages(messages, "BASE", tmpdir)

        self.assertIn("be nice", fresh[0]["content"])
        self.assertEqual(fresh[1], {"role": "user", "content": "hi"})
        # In-place refresh: same list object, system content updated.
        self.assertIs(fresh, messages)
        self.assertIn("be nice", messages[0]["content"])


class TrimTests(unittest.TestCase):

    def test_short_transcript_unchanged_no_note(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        trimmed = context_module.trim_messages(messages, 1000)
        self.assertEqual(trimmed, messages)

    def test_trims_oldest_keeps_system_and_recent(self):
        messages = [{"role": "system", "content": "sys"}]
        for i in range(20):
            messages.append(
                {"role": "user", "content": f"msg-{i} " + "x" * 40}
            )
        trimmed = context_module.trim_messages(messages, 60)

        self.assertEqual(trimmed[0]["content"], "sys")
        self.assertIn("Context trimmed", trimmed[1]["content"])
        self.assertEqual(trimmed[-1]["content"], messages[-1]["content"])
        self.assertLess(len(trimmed), len(messages))
        # Original transcript untouched.
        self.assertEqual(len(messages), 21)

    def test_orphan_tool_messages_dropped(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a " + "x" * 200},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "c", "type": "function",
                    "function": {"name": "t", "arguments": {}},
                }],
            },
            {"role": "tool", "tool_call_id": "c", "content": "result"},
            {"role": "user", "content": "latest question"},
        ]
        trimmed = context_module.trim_messages(messages, 30)
        roles = [m["role"] for m in trimmed]
        if "tool" in roles:
            self.assertNotEqual(roles[roles.index("tool") - 1], "system")

    def test_oversized_final_message_truncated_not_dropped(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "y" * 10000},
        ]
        trimmed = context_module.trim_messages(messages, 100)
        self.assertEqual(trimmed[-1]["role"], "user")
        self.assertIn("truncated", trimmed[-1]["content"])
        self.assertLess(len(trimmed[-1]["content"]), 10000)


class LimitResolutionTests(unittest.TestCase):

    def setUp(self):
        context_module.reset_context_limit()

    def tearDown(self):
        context_module.reset_context_limit()

    def test_precedence(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_CONTEXT_TOKENS", None)
            self.assertEqual(
                context_module.resolve_context_tokens(),
                context_module.DEFAULT_CONTEXT_TOKENS,
            )
            self.assertEqual(
                context_module.resolve_context_tokens(
                    None, {"context_tokens": 5000}
                ),
                5000,
            )
        with patch.dict(
            os.environ, {"SEACODE_CONTEXT_TOKENS": "7000"}
        ):
            self.assertEqual(context_module.resolve_context_tokens(), 7000)
            self.assertEqual(
                context_module.resolve_context_tokens(9000), 9000
            )

    def test_bad_values(self):
        with self.assertRaises(ValueError):
            context_module.resolve_context_tokens(0)
        with self.assertRaises(ValueError):
            context_module.set_context_limit("nope")
        with patch.dict(
            os.environ, {"SEACODE_CONTEXT_TOKENS": "nope"}
        ):
            # Unparseable env falls back to the default, like max_tools.
            self.assertEqual(
                context_module.get_context_limit(),
                context_module.DEFAULT_CONTEXT_TOKENS,
            )


class AgentTrimIntegrationTests(unittest.TestCase):
    """run_agent sends the trimmed copy but keeps the full transcript."""

    def test_wire_messages_trimmed(self):
        seen = []

        def chat(model, messages, **kwargs):
            seen.append(list(messages))
            return SimpleNamespace(
                message=SimpleNamespace(content="All done.")
            )

        messages = [{"role": "system", "content": "sys"}]
        for i in range(30):
            messages.append(
                {"role": "user", "content": f"turn {i} " + "x" * 80}
            )
        messages.append({"role": "user", "content": "final question"})

        with patch.object(seacode_module.client, "chat", side_effect=chat):
            answer = seacode_module.run_agent(
                messages, max_context_tokens=50
            )

        self.assertEqual(answer, "All done.")
        self.assertLess(len(seen[0]), len(messages))
        self.assertEqual(seen[0][-1]["content"], "final question")
        # The stored transcript still has every message (+ the answer).
        self.assertGreaterEqual(len(messages), 32)


if __name__ == "__main__":
    unittest.main()
