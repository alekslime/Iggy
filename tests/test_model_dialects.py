"""
Tests for model-output dialects: models that ignore native tool
calling hand-write JSON in assorted shapes ({"name": ...},
{"function": ..., "parameters": ...}, {"tool_calls": [...]},
{"answer": ...}). normalize_action maps them onto the canonical
{"tool": ...} shape the loop consumes.

The {"name": ...} case is a regression test: qwen2.5-coder:3b via
Ollama emits exactly that (fenced), and before normalization the
loop reported it as 'Unknown tool requested: None' while the model
had made the correct call.

Run with:  python3 -m pytest tests/test_model_dialects.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.tools as tools_module


class NormalizeActionTests(unittest.TestCase):

    def test_canonical_passthrough(self):
        action = {"tool": "read_file", "arguments": {"path": "a"}}
        self.assertIs(seacode_module.normalize_action(action), action)

    def test_name_dialect(self):
        action = seacode_module.normalize_action(
            {"name": "replace_in_file",
             "arguments": {"path": "h.py", "old_text": "x",
                           "new_text": "y"}}
        )
        self.assertEqual(action["tool"], "replace_in_file")
        self.assertEqual(action["arguments"]["path"], "h.py")

    def test_function_parameters_dialect(self):
        action = seacode_module.normalize_action(
            {"function": "read_file", "parameters": {"path": "a.py"}}
        )
        self.assertEqual(
            action, {"tool": "read_file", "arguments": {"path": "a.py"}}
        )

    def test_tool_calls_dialect(self):
        action = seacode_module.normalize_action(
            {"tool_calls": [{"function": {
                "name": "list_files", "arguments": {"root": "."},
            }}]}
        )
        self.assertEqual(action["tool"], "list_files")
        self.assertEqual(action["arguments"], {"root": "."})

    def test_answer_dialect(self):
        action = seacode_module.normalize_action({"answer": "done!"})
        self.assertEqual(action, {"tool": "none", "answer": "done!"})

    def test_string_arguments_parsed(self):
        action = seacode_module.normalize_action(
            {"name": "read_file", "arguments": '{"path": "a.py"}'}
        )
        self.assertEqual(action["arguments"], {"path": "a.py"})

    def test_unrecognized_untouched(self):
        action = {"something": "else"}
        self.assertIs(seacode_module.normalize_action(action), action)

    def test_parse_tolerates_extra_data(self):
        action = seacode_module.parse_action(
            '{"tool": "list_files", "arguments": {}}\n'
            '{"tool": "read_file", "arguments": {}}'
        )
        self.assertEqual(action["tool"], "list_files")
        action = seacode_module.parse_action(
            '{"name": "list_files"} and some trailing prose'
        )
        self.assertEqual(action["name"], "list_files")


def fake_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


class DialectEndToEndTests(unittest.TestCase):
    """A scripted {"name": ...} model completes a real edit."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)
        for module in (tools_module, seacode_module):
            patcher = patch.object(
                module, "get_project_root",
                return_value=self.project_root,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        perm = patch.object(
            tools_module, "request_permission", return_value=True
        )
        perm.start()
        self.addCleanup(perm.stop)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_fenced_name_dialect_completes_edit(self):
        (self.project_root / "hello.py").write_text(
            'def greet():\n    return "helllo world"\n'
        )
        scripted = [
            # What qwen2.5-coder:3b actually emits (fenced, name-keyed).
            "```json\n"
            + json.dumps({
                "name": "read_file",
                "arguments": {"path": "hello.py"},
            })
            + "\n```",
            json.dumps({
                "name": "replace_in_file",
                "arguments": {
                    "path": "hello.py",
                    "old_text": "helllo",
                    "new_text": "hello",
                },
            }),
            "Fixed the typo.",
        ]
        responses = list(scripted)

        def chat(model, messages, **kwargs):
            if not responses:
                raise AssertionError("chat ran out of responses")
            return fake_response(responses.pop(0))

        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user",
             "content": "In hello.py, change `helllo` to `hello`."},
        ]

        with patch.object(seacode_module.client, "chat",
                          side_effect=chat):
            answer = seacode_module.run_agent(messages)

        self.assertEqual(answer, "Fixed the typo.")
        self.assertIn(
            '"hello world"',
            (self.project_root / "hello.py").read_text(),
        )

    def test_undo_after_verified_edit_is_blocked(self):
        """M7.13: observed live -- the model undid its own verified
        fix when told to 'continue working'."""
        (self.project_root / "hello.py").write_text(
            'def greet():\n    return "helllo world"\n'
        )

        def call(name, **arguments):
            return json.dumps({"tool": name, "arguments": arguments})

        scripted = [
            call("read_file", path="hello.py"),
            call("replace_in_file", path="hello.py",
                 old_text="helllo", new_text="hello"),
            call("undo_last_change", path="hello.py"),
            "Fixed the typo.",
        ]
        responses = list(scripted)

        def chat(model, messages, **kwargs):
            if not responses:
                raise AssertionError("chat ran out of responses")
            return fake_response(responses.pop(0))

        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user",
             "content": "In hello.py, change `helllo` to `hello`."},
        ]

        with patch.object(seacode_module.client, "chat",
                          side_effect=chat):
            answer = seacode_module.run_agent(messages)

        self.assertEqual(answer, "Fixed the typo.")
        self.assertIn(
            '"hello world"',
            (self.project_root / "hello.py").read_text(),
        )
        # The followup after the clean edit told it to finish.
        followups = [
            m["content"] for m in messages if m["role"] == "user"
        ]
        self.assertTrue(
            any("give your final answer now" in f for f in followups)
        )


    def test_repeated_identical_call_blocked(self):
        """M7.14: observed live -- read_file loop burned the budget."""
        (self.project_root / "hello.py").write_text("x = 1\n")

        def call(name, **arguments):
            return json.dumps({"tool": name, "arguments": arguments})

        scripted = [
            call("read_file", path="hello.py"),
            call("read_file", path="hello.py"),
            call("read_file", path="hello.py"),
            "It contains x = 1.",
        ]
        responses = list(scripted)

        def chat(model, messages, **kwargs):
            if not responses:
                raise AssertionError("chat ran out of responses")
            return fake_response(responses.pop(0))

        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": "What is in hello.py?"},
        ]

        with patch.object(seacode_module.client, "chat",
                          side_effect=chat):
            answer = seacode_module.run_agent(messages)

        self.assertEqual(answer, "It contains x = 1.")
        followups = [
            m["content"] for m in messages if m["role"] == "user"
        ]
        self.assertTrue(
            any("already called" in f for f in followups)
        )


    def test_embedded_action_in_prose_is_corrected_not_answered(self):
        """Narrated plans with embedded calls must not become a
        'final answer' claiming work that never ran."""
        (self.project_root / "hello.py").write_text("x = 1\n")
        plan = (
            "Step 1: I will read the file.\n"
            '{"name": "read_file", "arguments": {"path": "hello.py"}}\n'
            "Step 2: done. I fixed everything and ran the tests."
        )

        def call(name, **arguments):
            return json.dumps({"tool": name, "arguments": arguments})

        scripted = [plan, call("read_file", path="hello.py"),
                    "It contains x = 1."]
        responses = list(scripted)

        def chat(model, messages, **kwargs):
            if not responses:
                raise AssertionError("chat ran out of responses")
            return fake_response(responses.pop(0))

        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": "What is in hello.py?"},
        ]

        with patch.object(seacode_module.client, "chat",
                          side_effect=chat):
            answer = seacode_module.run_agent(messages)

        self.assertEqual(answer, "It contains x = 1.")
        followups = [
            m["content"] for m in messages if m["role"] == "user"
        ]
        self.assertTrue(
            any("Nothing in that plan" in f for f in followups)
        )


    def test_intent_narration_without_work_is_rejected(self):
        """M7.15: 'I need to inspect...' with zero tools is not an
        answer to an action request."""
        (self.project_root / "hello.py").write_text("x = 1\n")

        def call(name, **arguments):
            return json.dumps({"tool": name, "arguments": arguments})

        scripted = [
            "I need to inspect hello.py to understand it.",
            call("read_file", path="hello.py"),
            "It contains x = 1.",
        ]
        responses = list(scripted)

        def chat(model, messages, **kwargs):
            if not responses:
                raise AssertionError("chat ran out of responses")
            return fake_response(responses.pop(0))

        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": "Read hello.py and tell me "
                                        "what it contains."},
        ]

        with patch.object(seacode_module.client, "chat",
                          side_effect=chat):
            answer = seacode_module.run_agent(messages)

        self.assertEqual(answer, "It contains x = 1.")
        followups = [
            m["content"] for m in messages if m["role"] == "user"
        ]
        self.assertTrue(
            any("not actually done anything" in f for f in followups)
        )


if __name__ == "__main__":
    unittest.main()
