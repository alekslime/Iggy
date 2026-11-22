"""
Tests for tools.delete_file and its integration with the agent loop.

delete_file is a modification tool, so it inherits the M7
guardrails: the M7.9 unread-file guard blocks blind deletions, M7.3
forces recovery after a failed deletion, and every deletion is
backed up first for undo_last_change.

Scenarios covered:
  1. Unit-level: delete removes the file and backs up its content;
     denied/empty/missing/outside-project/backup-dir cases.
  2. Unit-level: classify_modification_result recognizes delete
     outcomes.
  3. Integration: model reads a file, deletes it, and finishes.
  4. Integration: a blind delete is blocked until the model reads
     first (M7.9).
  5. Integration: a failed delete forces recovery instead of
     letting the agent quit (M7.3).

Run with:  python3 -m pytest tests/test_delete_file.py -v
       or: python3 -m unittest tests.test_delete_file -v
"""

import json
import tempfile
import unittest
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
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError(
                "ScriptedChat ran out of scripted responses -- the agent "
                "asked the model for another turn than the test expected."
            )
        return fake_response(self._responses.pop(0))


class DeleteFileUnitTests(unittest.TestCase):

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

    def test_delete_removes_file_and_backs_up(self):
        self.make_file("scratch.txt", "bye\n")

        result = tools_module.delete_file("scratch.txt")

        self.assertIn("deleted successfully", result)
        self.assertFalse((self.project_root / "scratch.txt").exists())

        entries = tools_module.list_backups("scratch.txt")
        self.assertEqual(len(entries), 1)
        self.assertEqual(
            (self.project_root / entries[0]["backup"]).read_text(),
            "bye\n",
        )

    def test_denied_delete_keeps_file(self):
        self.make_file("scratch.txt", "bye\n")

        with patch.object(
            tools_module, "request_permission", return_value=False
        ):
            result = tools_module.delete_file("scratch.txt")

        self.assertEqual(result, "File modification denied by user.")
        self.assertTrue((self.project_root / "scratch.txt").exists())
        self.assertEqual(tools_module.list_backups(), [])

    def test_delete_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            tools_module.delete_file("nope.txt")

    def test_delete_empty_path_raises(self):
        with self.assertRaises(ValueError):
            tools_module.delete_file("   ")

    def test_delete_outside_project_raises(self):
        with self.assertRaises(PermissionError):
            tools_module.delete_file("../outside.txt")

    def test_delete_backup_dir_raises(self):
        with self.assertRaises(PermissionError):
            tools_module.delete_file(".seacode_backups/some.bak")

    def test_delete_is_registered_as_agent_tool(self):
        self.assertIs(
            tools_module.TOOLS["delete_file"],
            tools_module.delete_file,
        )


class ClassifyDeleteTests(unittest.TestCase):

    def test_success(self):
        self.assertEqual(
            seacode_module.classify_modification_result(
                "delete_file", "File deleted successfully: a.txt"
            ),
            "success",
        )

    def test_denied(self):
        self.assertEqual(
            seacode_module.classify_modification_result(
                "delete_file", "File modification denied by user."
            ),
            "denied",
        )

    def test_failed(self):
        self.assertEqual(
            seacode_module.classify_modification_result(
                "delete_file", {"error": "boom"}
            ),
            "failed",
        )
        self.assertEqual(
            seacode_module.classify_modification_result(
                "delete_file", "Delete failed: nope."
            ),
            "failed",
        )


class DeleteFileIntegrationTests(unittest.TestCase):

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
        with patch.object(seacode_module.client, "chat", side_effect=chat):
            messages = [
                {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
                {"role": "user", "content": user_request},
            ]
            answer = seacode_module.run_agent(messages)
        return answer, messages, chat

    def test_read_then_delete_finishes(self):
        self.make_file("scratch.txt", "bye\n")

        scripted = [
            tool_call("read_file", path="scratch.txt"),
            tool_call("delete_file", path="scratch.txt"),
            final_answer("Deleted scratch.txt."),
        ]

        answer, _, _ = self.run_scripted(
            scripted, "Delete scratch.txt."
        )

        self.assertEqual(answer, "Deleted scratch.txt.")
        self.assertFalse((self.project_root / "scratch.txt").exists())
        # Backed up first, so undo can bring it back.
        self.assertEqual(len(tools_module.list_backups()), 1)

    def test_blind_delete_is_blocked_until_read(self):
        self.make_file("scratch.txt", "bye\n")

        scripted = [
            tool_call("delete_file", path="scratch.txt"),
            tool_call("read_file", path="scratch.txt"),
            tool_call("delete_file", path="scratch.txt"),
            final_answer("Deleted scratch.txt."),
        ]

        answer, _, chat = self.run_scripted(
            scripted, "Delete scratch.txt."
        )

        self.assertEqual(answer, "Deleted scratch.txt.")
        guard_message = chat.calls[1][-1]["content"]
        self.assertIn("hasn't been read or searched", guard_message)
        self.assertFalse((self.project_root / "scratch.txt").exists())

    def test_failed_delete_forces_recovery(self):
        scripted = [
            tool_call("delete_file", path="missing.txt"),
            final_answer("Could not delete it."),
            final_answer("Still cannot delete it."),
            final_answer("Gave up after retrying."),
        ]

        answer, _, chat = self.run_scripted(
            scripted, "Delete missing.txt."
        )

        # The first two final answers are rejected (forced retries);
        # the third is accepted once the retry cap is reached.
        self.assertEqual(answer, "Gave up after retrying.")
        retry_message = chat.calls[2][-1]["content"]
        self.assertIn("may not give up yet", retry_message)


if __name__ == "__main__":
    unittest.main()
