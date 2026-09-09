"""
Mocked tests for M7.9 -- unread-file edit guard.

Same testing philosophy as M7.6/M7.7/M7.8: no real Ollama server, no
real project directory (`agent.tools.get_project_root` is patched to a
temp dir), model responses are scripted.

Unlike M7.6/M7.7/M7.8, this is a pre-execution guard, not a post-hoc
verification: the model's blind write_file/replace_in_file call never
reaches the filesystem in the first place, so there's nothing to revert
and no separate retry-cap state -- the guard simply re-fires on every
attempt until the model actually reads or searches the file (bounded,
same as everything else, by MAX_TOOL_CALLS).

Scenarios covered:
  1. Unit-level: `check_unread_file_edit` blocks an existing,
     never-seen file; allows a new file; allows an already-seen file;
     ignores non-modification tools and unusable arguments.
  2. Integration: model calls `replace_in_file` on an existing file
     with no prior read/search this run -- blocked before the tool
     executes, then succeeds once it reads first.
  3. Integration: model calls `write_file` on an existing file with no
     prior read/search -- also blocked (guard applies to both
     modification tools, not just replace_in_file).
  4. Integration: `search_files` counting as "seen" for any path it
     returns a match in, without a separate `read_file` call.
  5. Integration: creating a brand-new file with `write_file` is never
     blocked, and a same-run follow-up `replace_in_file` on that same
     freshly-created file is not blocked either (a successful
     modification counts as having seen the file).
  6. Regression: a file already read earlier in the same run is not
     re-flagged on a later modification attempt.

Run with:  python3 -m pytest tests/test_m7_9_unread_file.py -v
       or: python3 -m unittest tests.test_m7_9_unread_file -v
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


class CheckUnreadFileEditUnitTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        patcher = patch.object(
            tools_module, "get_project_root", return_value=self.project_root
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_file(self, relative_path, content):
        full_path = self.project_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def test_blocks_existing_never_seen_file(self):
        self.make_file("app/config.py", "DEBUG = True\n")

        message = agent_module.check_unread_file_edit(
            "replace_in_file",
            {"path": "app/config.py", "old_text": "x", "new_text": "y"},
            set(),
        )

        self.assertIsNotNone(message)
        self.assertIn("app/config.py", message)
        self.assertIn("hasn't been read or searched", message)

    def test_allows_already_seen_file(self):
        self.make_file("app/config.py", "DEBUG = True\n")

        message = agent_module.check_unread_file_edit(
            "replace_in_file",
            {"path": "app/config.py", "old_text": "x", "new_text": "y"},
            {"app/config.py"},
        )

        self.assertIsNone(message)

    def test_allows_new_file(self):
        message = agent_module.check_unread_file_edit(
            "write_file",
            {"path": "app/new_module.py", "content": "print('hi')"},
            set(),
        )

        self.assertIsNone(message)

    def test_ignores_non_modification_tools(self):
        message = agent_module.check_unread_file_edit(
            "read_file",
            {"path": "app/config.py"},
            set(),
        )

        self.assertIsNone(message)

    def test_ignores_unusable_arguments(self):
        self.assertIsNone(
            agent_module.check_unread_file_edit(
                "write_file", "not-a-dict", set()
            )
        )
        self.assertIsNone(
            agent_module.check_unread_file_edit(
                "write_file", {"path": ""}, set()
            )
        )
        self.assertIsNone(
            agent_module.check_unread_file_edit(
                "write_file", {"path": 123}, set()
            )
        )


class M79IntegrationTests(unittest.TestCase):

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
    # Blind replace_in_file on an existing, never-read file -- blocked,
    # then succeeds once the model reads first.
    # -------------------------------------------------------------

    def test_blind_replace_in_file_is_blocked_then_recovers(self):
        self.make_file("app/config.py", "DEBUG = True\n")

        user_request = "Set DEBUG to False in app/config.py."

        scripted = [
            # No read_file/search_files call at all -- straight to a
            # blind edit.
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            # Complies with the guard's instruction.
            tool_call("read_file", path="app/config.py"),
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            final_answer("Updated DEBUG to False."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Updated DEBUG to False.")

        # The blocking message was injected as the second model turn's
        # trailing user message.
        second_call_messages = chat.calls[1]
        guard_message = second_call_messages[-1]["content"]
        self.assertIn("hasn't been read or searched", guard_message)

        # The file must reflect the eventual successful edit.
        self.assertEqual(
            (self.project_root / "app/config.py").read_text(),
            "DEBUG = False\n",
        )

    # -------------------------------------------------------------
    # Blind write_file on an existing file is blocked too -- not just
    # replace_in_file.
    # -------------------------------------------------------------

    def test_blind_write_file_is_blocked(self):
        self.make_file("app/config.py", "DEBUG = True\n")

        user_request = "Rewrite the entire app/config.py from scratch."

        scripted = [
            tool_call(
                "write_file",
                path="app/config.py",
                content="DEBUG = False\n",
            ),
            tool_call("read_file", path="app/config.py"),
            tool_call(
                "write_file",
                path="app/config.py",
                content="DEBUG = False\n",
            ),
            final_answer("Rewrote the file."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Rewrote the file.")

        second_call_messages = chat.calls[1]
        guard_message = second_call_messages[-1]["content"]
        self.assertIn("hasn't been read or searched", guard_message)

    # -------------------------------------------------------------
    # search_files counts as "seen" for any path it matches in, even
    # without a separate read_file call.
    # -------------------------------------------------------------

    def test_search_files_match_counts_as_seen(self):
        self.make_file("app/config.py", "DEBUG = True\n")

        user_request = "Set DEBUG to False in app/config.py."

        scripted = [
            tool_call("search_files", query="DEBUG"),
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            final_answer("Updated DEBUG to False."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Updated DEBUG to False.")
        # No guard message anywhere in the transcript -- the edit went
        # straight through after the search hit.
        self.assertEqual(len(chat.calls), 3)

    # -------------------------------------------------------------
    # Creating a brand-new file is never blocked, and a same-run
    # follow-up edit to that same file isn't blocked either.
    # -------------------------------------------------------------

    def test_new_file_then_same_run_edit_is_not_blocked(self):
        user_request = "Create app/new_module.py, then fix a typo in it."

        scripted = [
            tool_call(
                "write_file",
                path="app/new_module.py",
                content="def helo():\n    pass\n",
            ),
            tool_call(
                "replace_in_file",
                path="app/new_module.py",
                old_text="def helo():",
                new_text="def hello():",
            ),
            final_answer("Created the file and fixed the typo."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Created the file and fixed the typo.")
        self.assertEqual(len(chat.calls), 3)
        self.assertEqual(
            (self.project_root / "app/new_module.py").read_text(),
            "def hello():\n    pass\n",
        )

    # -------------------------------------------------------------
    # A file already read earlier in the run is not re-flagged later.
    # -------------------------------------------------------------

    def test_previously_read_file_is_not_reflagged(self):
        self.make_file("app/config.py", "DEBUG = True\nVERBOSE = False\n")

        user_request = "Set DEBUG to False and VERBOSE to True in app/config.py."

        scripted = [
            tool_call("read_file", path="app/config.py"),
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="VERBOSE = False",
                new_text="VERBOSE = True",
            ),
            final_answer("Updated both settings."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Updated both settings.")
        self.assertEqual(len(chat.calls), 4)


if __name__ == "__main__":
    unittest.main()
