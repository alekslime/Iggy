"""
Mocked tests for M7.11 -- incomplete multi-target edit detection.

Same testing philosophy as M7.6/M7.7/M7.9/M7.10: no real Ollama server,
no real project directory (`agent.tools.get_project_root` is patched to
a temp dir), model responses are scripted.

This is a deliberately narrow slice of "incomplete multi-step": it only
fires when the request names two or more files by name (reusing M7.7's
extract_mentioned_filenames) and at least one hasn't received a
successful, verified modification by the time a final answer is
attempted. Multi-step tasks not anchored to named files aren't covered.

Scenarios covered:
  1. Unit-level: `check_incomplete_multi_target_edit` -- flags a
     missing file, accepts when both are done, and stays quiet when
     fewer than two files are named or nothing was modified yet.
  2. Integration: request names two files, model only edits one and
     tries to stop -- rejected and forced to continue, accepted once
     both are done.
  3. Integration: request names two files, but the model explicitly
     explains in its final answer that the second doesn't need
     changes -- still forced to continue in this narrow implementation
     (the check doesn't parse the answer text), but accepted once the
     retry cap is reached, without ever silently succeeding on the
     first attempt.
  4. Regression: single-file request is never flagged, and a pure
     information request that mentions two files but modifies neither
     is never flagged either.

Run with:  python3 -m pytest tests/test_m7_11_incomplete_multi_target.py -v
       or: python3 -m unittest tests.test_m7_11_incomplete_multi_target -v
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


class CheckIncompleteMultiTargetUnitTests(unittest.TestCase):

    def test_flags_missing_file(self):
        message = agent_module.check_incomplete_multi_target_edit(
            "Rename `foo` to `bar` in app/api.py and app/client.py.",
            {"app/api.py"},
        )

        self.assertIsNotNone(message)
        self.assertIn("app/client.py", message)

    def test_accepts_when_both_done(self):
        message = agent_module.check_incomplete_multi_target_edit(
            "Rename `foo` to `bar` in app/api.py and app/client.py.",
            {"app/api.py", "app/client.py"},
        )

        self.assertIsNone(message)

    def test_ignores_single_file_request(self):
        message = agent_module.check_incomplete_multi_target_edit(
            "Fix a typo in app/api.py.",
            set(),
        )

        self.assertIsNone(message)

    def test_ignores_when_nothing_modified_yet(self):
        message = agent_module.check_incomplete_multi_target_edit(
            "Rename `foo` to `bar` in app/api.py and app/client.py.",
            set(),
        )

        self.assertIsNone(message)


class M711IntegrationTests(unittest.TestCase):

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
    # Two files named, only one edited -- rejected, forced to
    # continue, accepted once both are done.
    # -------------------------------------------------------------

    def test_partial_multi_file_edit_is_rejected_then_completed(self):
        self.make_file("app/api.py", "def foo():\n    pass\n")
        self.make_file("app/client.py", "def foo():\n    pass\n")

        user_request = (
            "Rename the foo function to bar in app/api.py and "
            "app/client.py."
        )

        scripted = [
            tool_call("read_file", path="app/api.py"),
            tool_call(
                "replace_in_file",
                path="app/api.py",
                old_text="def foo():",
                new_text="def bar():",
            ),
            final_answer("Renamed foo to bar."),  # rejected: client.py untouched
            tool_call("read_file", path="app/client.py"),
            tool_call(
                "replace_in_file",
                path="app/client.py",
                old_text="def foo():",
                new_text="def bar():",
            ),
            final_answer("Renamed foo to bar in both files."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Renamed foo to bar in both files.")
        self.assertIn("def bar():", self.read_file("app/api.py"))
        self.assertIn("def bar():", self.read_file("app/client.py"))

        rejection = chat.calls[3][-1]["content"]
        self.assertIn("app/client.py", rejection)

    # -------------------------------------------------------------
    # Retry cap: model insists the second file doesn't need changes;
    # forced retries happen (this narrow check doesn't parse the
    # answer text), then accepted once the cap is reached.
    # -------------------------------------------------------------

    def test_retry_cap_eventually_accepts_explained_partial_edit(self):
        self.make_file("app/api.py", "def foo():\n    pass\n")
        self.make_file("app/client.py", "def foo():\n    pass\n")

        user_request = (
            "Rename the foo function to bar in app/api.py and "
            "app/client.py."
        )

        scripted = [
            tool_call("read_file", path="app/api.py"),
            tool_call(
                "replace_in_file",
                path="app/api.py",
                old_text="def foo():",
                new_text="def bar():",
            ),
            final_answer("Renamed in api.py."),  # retry 1/2
            final_answer(
                "Renamed in api.py; client.py doesn't call foo so no "
                "change needed there."
            ),  # retry 2/2
            final_answer("Renamed in api.py only."),  # accepted: cap reached
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Renamed in api.py only.")
        self.assertIn("def bar():", self.read_file("app/api.py"))
        # client.py was legitimately left alone in this scenario.
        self.assertIn("def foo():", self.read_file("app/client.py"))

    # -------------------------------------------------------------
    # Regression: single-file request never flagged.
    # -------------------------------------------------------------

    def test_single_file_request_is_never_flagged(self):
        self.make_file("app/api.py", "def foo():\n    pass\n")

        user_request = "Rename the foo function to bar in app/api.py."

        scripted = [
            tool_call("read_file", path="app/api.py"),
            tool_call(
                "replace_in_file",
                path="app/api.py",
                old_text="def foo():",
                new_text="def bar():",
            ),
            final_answer("Renamed foo to bar."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Renamed foo to bar.")
        self.assertEqual(len(chat.calls), 3)

    # -------------------------------------------------------------
    # Regression: pure info request mentioning two files, no
    # modification at all -- never flagged.
    # -------------------------------------------------------------

    def test_info_request_mentioning_two_files_is_never_flagged(self):
        self.make_file("app/api.py", "def foo():\n    pass\n")
        self.make_file("app/client.py", "def foo():\n    pass\n")

        user_request = "How do app/api.py and app/client.py interact?"

        scripted = [
            tool_call("read_file", path="app/api.py"),
            tool_call("read_file", path="app/client.py"),
            final_answer("They both define a similarly-named function."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(
            answer, "They both define a similarly-named function."
        )
        self.assertEqual(len(chat.calls), 3)


if __name__ == "__main__":
    unittest.main()
