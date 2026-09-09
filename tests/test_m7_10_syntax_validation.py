"""
Mocked tests for M7.10 -- syntax validation.

Same testing philosophy as M7.6/M7.7/M7.9: no real Ollama server, no
real project directory (`agent.tools.get_project_root` is patched to a
temp dir), model responses are scripted.

Checked regardless of whether the file already had a syntax error
before the edit -- the bar is "nothing Iggy leaves behind is broken
Python," not "don't make an existing problem worse." This differs from
M7.6's destructive-change check, which deliberately compares before vs.
after.

Scenarios covered:
  1. Unit-level: `verify_python_syntax` -- valid code passes, a
     SyntaxError is caught and described, non-.py files and
     non-string content are skipped.
  2. Integration: an edit to an existing .py file leaves it syntactically
     broken -- reverted and retried, same shape as M7.6/M7.7.
  3. Integration: creating a brand-new .py file with broken syntax --
     the file is removed entirely (no prior content to fall back to),
     and the model is forced to retry.
  4. Integration: an edit to a .py file that already had a syntax error
     before the edit, and still does after -- still flagged (M7.10
     doesn't grandfather in pre-existing breakage).
  5. Regression: a valid edit to a .py file is never flagged, and a
     non-Python file (e.g. .txt) with content that wouldn't parse as
     Python is never checked at all.

Run with:  python3 -m pytest tests/test_m7_10_syntax_validation.py -v
       or: python3 -m unittest tests.test_m7_10_syntax_validation -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent.agent as agent_module
import agent.tools as tools_module


def fake_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def tool_call(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def final_answer(answer):
    return json.dumps({"tool": "none", "answer": answer})


class ScriptedChat:

    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls = []

    def __call__(self, model, messages):
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError(
                "ScriptedChat ran out of scripted responses -- the agent "
                "asked the model for another turn than the test expected."
            )
        return fake_response(self._responses.pop(0))


class VerifyPythonSyntaxUnitTests(unittest.TestCase):

    def test_valid_code_passes(self):
        self.assertIsNone(
            agent_module.verify_python_syntax(
                "app/utils.py", "def add(a, b):\n    return a + b\n"
            )
        )

    def test_syntax_error_is_caught_and_described(self):
        message = agent_module.verify_python_syntax(
            "app/utils.py", "def add(a, b:\n    return a + b\n"
        )

        self.assertIsNotNone(message)
        self.assertIn("app/utils.py", message)
        self.assertIn("does not parse", message)

    def test_non_python_file_is_skipped(self):
        self.assertIsNone(
            agent_module.verify_python_syntax(
                "notes/todo.txt", "def add(a, b:\n    this is not python\n"
            )
        )

    def test_non_string_content_is_skipped(self):
        self.assertIsNone(
            agent_module.verify_python_syntax("app/utils.py", None)
        )

    def test_non_string_path_is_skipped(self):
        self.assertIsNone(agent_module.verify_python_syntax(None, "x = 1"))


class M710IntegrationTests(unittest.TestCase):

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

    def make_file(self, relative_path, content):
        full_path = self.project_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def read_file(self, relative_path):
        return (self.project_root / relative_path).read_text(
            encoding="utf-8"
        )

    def file_exists(self, relative_path):
        return (self.project_root / relative_path).exists()

    def run_scripted(self, scripted_responses, user_request):
        chat = ScriptedChat(scripted_responses)
        with patch.object(agent_module.client, "chat", side_effect=chat):
            messages = [
                {"role": "system", "content": agent_module.SYSTEM_PROMPT},
                {"role": "user", "content": user_request},
            ]
            answer = agent_module.run_agent(messages)
        return answer, messages, chat

    # -------------------------------------------------------------
    # Edit to an existing file leaves it syntactically broken --
    # reverted and retried.
    # -------------------------------------------------------------

    def test_broken_edit_to_existing_file_is_reverted_and_retried(self):
        self.make_file(
            "app/utils.py", "def add(a, b):\n    return a + b\n"
        )

        user_request = "Rename the add function in app/utils.py to plus."

        scripted = [
            tool_call("read_file", path="app/utils.py"),
            tool_call(
                "replace_in_file",
                path="app/utils.py",
                old_text="def add(a, b):",
                new_text="def plus(a, b:",  # broken syntax
            ),
            final_answer("Done."),  # should be rejected
            tool_call(
                "replace_in_file",
                path="app/utils.py",
                old_text="def add(a, b):",
                new_text="def plus(a, b):",
            ),
            final_answer("Renamed add to plus."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Renamed add to plus.")
        self.assertEqual(
            self.read_file("app/utils.py"),
            "def plus(a, b):\n    return a + b\n",
        )

        rejection = chat.calls[2][-1]["content"]
        self.assertIn("does not parse", rejection)

    # -------------------------------------------------------------
    # Brand-new .py file created with broken syntax -- removed
    # entirely (no prior content to fall back to), retried.
    # -------------------------------------------------------------

    def test_broken_new_file_is_removed_and_retried(self):
        user_request = "Create app/greet.py with a greet function."

        scripted = [
            tool_call(
                "write_file",
                path="app/greet.py",
                content="def greet(name:\n    print(f'hi {name}')\n",
            ),
            final_answer("Created it."),  # should be rejected
            tool_call(
                "write_file",
                path="app/greet.py",
                content="def greet(name):\n    print(f'hi {name}')\n",
            ),
            final_answer("Created greet.py."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Created greet.py.")
        self.assertEqual(
            self.read_file("app/greet.py"),
            "def greet(name):\n    print(f'hi {name}')\n",
        )

        rejection = chat.calls[1][-1]["content"]
        self.assertIn("does not parse", rejection)
        self.assertIn("removed", rejection)

    # -------------------------------------------------------------
    # A file that already had a syntax error before the edit, and
    # still does after -- still flagged. M7.10 doesn't grandfather in
    # pre-existing breakage.
    # -------------------------------------------------------------

    def test_pre_existing_syntax_error_is_still_flagged(self):
        self.make_file(
            "app/broken.py", "def already_broken(:\n    pass\n"
        )

        user_request = "Add a docstring to app/broken.py."

        scripted = [
            tool_call("read_file", path="app/broken.py"),
            tool_call(
                "write_file",
                path="app/broken.py",
                content='"""Docstring."""\ndef already_broken(:\n    pass\n',
            ),
            final_answer("Added a docstring."),  # rejected: still broken
            tool_call(
                "write_file",
                path="app/broken.py",
                content='"""Docstring."""\n\n\ndef already_broken():\n    pass\n',
            ),
            final_answer("Added a docstring and fixed the syntax."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(
            answer, "Added a docstring and fixed the syntax."
        )
        self.assertNotIn("already_broken(:", self.read_file("app/broken.py"))

    # -------------------------------------------------------------
    # Regression: a valid edit is never flagged.
    # -------------------------------------------------------------

    def test_valid_edit_is_never_flagged(self):
        self.make_file(
            "app/utils.py", "def add(a, b):\n    return a + b\n"
        )

        user_request = "Rename add to plus in app/utils.py."

        scripted = [
            tool_call("read_file", path="app/utils.py"),
            tool_call(
                "replace_in_file",
                path="app/utils.py",
                old_text="def add(a, b):",
                new_text="def plus(a, b):",
            ),
            final_answer("Renamed add to plus."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Renamed add to plus.")
        self.assertEqual(len(chat.calls), 3)

    # -------------------------------------------------------------
    # Regression: a non-Python file is never syntax-checked, even with
    # content that wouldn't parse as Python.
    # -------------------------------------------------------------

    def test_non_python_file_is_never_checked(self):
        self.make_file("notes/todo.txt", "buy milk\n")

        user_request = "Add 'call mom' to notes/todo.txt."

        scripted = [
            tool_call("read_file", path="notes/todo.txt"),
            tool_call(
                "write_file",
                path="notes/todo.txt",
                content="buy milk\ncall mom (this isn't python at all!\n",
            ),
            final_answer("Added the note."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Added the note.")
        self.assertEqual(len(chat.calls), 3)


if __name__ == "__main__":
    unittest.main()
