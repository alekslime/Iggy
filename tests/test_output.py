"""
Tests for the transcript output contract (seacode.py's print formats).

run_agent's stdout is shared UI: the REPL shows it directly and the
TUI parses these exact lines for styling, so the formats are pinned
here. Plain text throughout -- no ANSI codes (see seacode.py).

Scenarios covered:
  1. format_tool_call summarizes path/command/query targets and
     truncates long ones.
  2. print_tool_result marks, indents, and truncates output.
  3. A run prints the compact `● tool` line (not the old verbose
     block) and `✖` errors.
  4. The raw-response dump appears only with verbose=True.

Run with:  python3 -m pytest tests/test_output.py -v
       or: python3 -m unittest tests.test_output -v
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.tools as tools_module


def fake_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def tool_call(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def final_answer(answer):
    return json.dumps({"tool": "none", "answer": answer})


class ScriptedChat:

    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)

    def __call__(self, model, messages):
        if not self._responses:
            raise AssertionError(
                "ScriptedChat ran out of scripted responses."
            )
        return fake_response(self._responses.pop(0))


class FormatToolCallTests(unittest.TestCase):

    def test_path_target(self):
        self.assertEqual(
            seacode_module.format_tool_call(
                "read_file", {"path": "demo.txt"}
            ),
            "● read_file demo.txt",
        )

    def test_command_target(self):
        self.assertEqual(
            seacode_module.format_tool_call(
                "run_command", {"command": "pytest tests/ -q"}
            ),
            "● run_command pytest tests/ -q",
        )

    def test_query_target(self):
        self.assertEqual(
            seacode_module.format_tool_call(
                "search_files", {"query": "ollama"}
            ),
            "● search_files ollama",
        )

    def test_no_target(self):
        self.assertEqual(
            seacode_module.format_tool_call("git_status", {}),
            "● git_status",
        )
        self.assertEqual(
            seacode_module.format_tool_call("git_status", None),
            "● git_status",
        )

    def test_long_target_truncates(self):
        line = seacode_module.format_tool_call(
            "read_file", {"path": "x" * 100}
        )

        self.assertTrue(line.startswith("● read_file "))
        self.assertTrue(line.endswith("..."))
        self.assertLessEqual(len(line), len("● read_file ") + 70)


class PrintToolResultTests(unittest.TestCase):

    def printed(self, result):
        output = io.StringIO()

        with redirect_stdout(output):
            seacode_module.print_tool_result(result)

        return output.getvalue()

    def test_single_line(self):
        self.assertEqual(
            self.printed("hello"), "  ⎿ hello\n"
        )

    def test_multiline_indents_continuations(self):
        self.assertEqual(
            self.printed("one\ntwo"),
            "  ⎿ one\n    two\n",
        )

    def test_long_output_truncates(self):
        text = "\n".join(f"line {n}" for n in range(20))

        out = self.printed(text)

        self.assertIn("  ⎿ line 0\n", out)
        self.assertIn("    line 7\n", out)
        self.assertNotIn("line 8\n", out)
        self.assertIn("  … (+12 more lines)\n", out)

    def test_structures_print_as_json(self):
        out = self.printed([{"file": "a.txt"}])

        self.assertIn('"file": "a.txt"', out)
        self.assertTrue(out.startswith("  ⎿ "))


class RunPrintsTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        patchers = [
            patch.object(
                tools_module, "get_project_root", return_value=self.project_root
            ),
            patch.object(
                tools_module, "request_permission", return_value=True
            ),
        ]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)

    def run_captured(self, scripted, user_request, **kwargs):
        chat = ScriptedChat(scripted)
        output = io.StringIO()
        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]

        with (
            patch.object(
                seacode_module.client, "chat", side_effect=chat
            ),
            redirect_stdout(output),
        ):
            answer = seacode_module.run_agent(messages, **kwargs)

        return answer, output.getvalue()

    def test_compact_tool_line(self):
        (self.project_root / "a.txt").write_text("hello\n")

        _, output = self.run_captured(
            [
                tool_call("read_file", path="a.txt"),
                final_answer("It says hello."),
            ],
            "What is in a.txt?",
        )

        self.assertIn("● read_file a.txt", output)
        self.assertIn("  ⎿ hello", output)
        self.assertNotIn("Tool requested", output)
        self.assertNotIn("Arguments:", output)
        self.assertNotIn("Tool executed", output)

    def test_error_line(self):
        _, output = self.run_captured(
            [
                tool_call("delete_file", path="missing.txt"),
                final_answer("Gave up."),
                final_answer("Gave up."),
                final_answer("Gave up."),
            ],
            "Delete missing.txt.",
        )

        self.assertIn("✖ File does not exist: missing.txt", output)

    def test_raw_dump_only_when_verbose(self):
        scripted = lambda: [final_answer("Done.")]

        _, quiet = self.run_captured(scripted(), "Hi.")
        self.assertNotIn("Model raw response", quiet)

        _, loud = self.run_captured(scripted(), "Hi.", verbose=True)
        self.assertIn("Model raw response", loud)


class FormatDurationTests(unittest.TestCase):

    def test_seconds(self):
        self.assertEqual(seacode_module.format_duration(3.7), "3s")
        self.assertEqual(seacode_module.format_duration(0), "0s")

    def test_minutes(self):
        self.assertEqual(seacode_module.format_duration(61.7), "1m01s")
        self.assertEqual(seacode_module.format_duration(600), "10m00s")


class ThinkingSpinnerTests(unittest.TestCase):

    def test_disabled_off_tty(self):
        # pytest captures stdout: not a tty, so the spinner must stay
        # silent and never start a thread.
        spinner = seacode_module._ThinkingSpinner(True)

        self.assertFalse(spinner.enabled)

        output = io.StringIO()

        with redirect_stdout(output):
            with spinner:
                pass

        self.assertEqual(output.getvalue(), "")
        self.assertIsNone(spinner._thread)


if __name__ == "__main__":
    unittest.main()
