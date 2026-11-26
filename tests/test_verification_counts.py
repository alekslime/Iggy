"""
Tests for which tool calls count as verifying a modification
(seacode.verify.counts_as_verification, used by the M7.8 guardrail).

A read_file / search_files / run_command call that errored, was blocked
by a hook, or was denied by the user checked nothing, so it must not
clear the "unverified modification" flag -- otherwise a model could run
a doomed command and then claim "I tested it". A command that ran and
failed (a red test run) is still a real check.

Run with:  python3 -m pytest tests/test_verification_counts.py -v
       or: python3 -m unittest tests.test_verification_counts -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.tools as tools_module
from seacode.verify import counts_as_verification


def tool_call(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def final_answer(answer):
    return json.dumps({"tool": "none", "answer": answer})


class CountsAsVerificationTests(unittest.TestCase):

    def test_successful_verification_tools_count(self):
        self.assertTrue(counts_as_verification("read_file", "text"))
        self.assertTrue(counts_as_verification("run_command", "ok"))
        self.assertTrue(counts_as_verification("run_command", ""))
        self.assertTrue(counts_as_verification("search_files", []))
        self.assertTrue(counts_as_verification("search_files", [{"file": "a"}]))

    def test_errors_do_not_count(self):
        error = {"error": "File does not exist: x"}

        for tool in ("read_file", "run_command", "search_files"):
            self.assertFalse(counts_as_verification(tool, error), tool)

    def test_denied_command_does_not_count(self):
        self.assertFalse(counts_as_verification(
            "run_command", tools_module.COMMAND_DENIED_MESSAGE
        ))

    def test_failing_command_output_still_counts(self):
        self.assertTrue(counts_as_verification(
            "run_command", "FAILED tests/test_a.py::test_x\n1 failed"
        ))

    def test_other_tools_never_count(self):
        for tool in ("write_file", "list_files", "git_status", "bogus"):
            self.assertFalse(counts_as_verification(tool, "ok"), tool)


class ScriptedChat:

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError("ScriptedChat ran out of responses")
        return SimpleNamespace(
            message=SimpleNamespace(content=self._responses.pop(0))
        )


class FailedVerificationIntegrationTests(unittest.TestCase):

    EDIT = [
        tool_call("read_file", path="config.py"),
        tool_call(
            "replace_in_file", path="config.py",
            old_text="DEBUG = True", new_text="DEBUG = False",
        ),
    ]
    CLAIM = "Done, I verified the change works correctly."
    HONEST = "Done, I updated DEBUG to False."

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "config.py").write_text("DEBUG = True\n")

        for p in (
            patch.object(
                tools_module, "get_project_root", return_value=self.root
            ),
            patch.object(
                seacode_module, "get_project_root", return_value=self.root
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def run_scripted(self, steps, allow_commands=True):
        def permission(action, diff=None):
            return allow_commands or not action.startswith("run:")

        chat = ScriptedChat(steps)

        with patch.object(
            tools_module, "request_permission", side_effect=permission
        ), patch.object(seacode_module.client, "chat", side_effect=chat):
            messages = [
                {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
                {"role": "user", "content": "Set DEBUG to False."},
            ]
            answer = seacode_module.run_agent(messages)

        return answer, chat

    def assertClaimRejected(self, verifying_step, allow_commands=True):
        answer, chat = self.run_scripted(
            [*self.EDIT, verifying_step,
             final_answer(self.CLAIM), final_answer(self.HONEST)],
            allow_commands=allow_commands,
        )

        self.assertEqual(answer, self.HONEST)
        self.assertIn(
            "claims the change was verified",
            chat.calls[4][-1]["content"],
        )

    def test_failed_read_does_not_verify(self):
        self.assertClaimRejected(tool_call("read_file", path="missing.py"))

    def test_disallowed_command_does_not_verify(self):
        self.assertClaimRejected(tool_call("run_command", command="rm a.txt"))

    def test_command_with_outside_argument_does_not_verify(self):
        self.assertClaimRejected(
            tool_call("run_command", command="cat ../../etc/passwd")
        )

    def test_denied_command_does_not_verify(self):
        self.assertClaimRejected(
            tool_call("run_command", command="python config.py"),
            allow_commands=False,
        )

    def test_command_that_ran_does_verify(self):
        answer, chat = self.run_scripted([
            *self.EDIT,
            tool_call("run_command", command="cat config.py"),
            final_answer(self.CLAIM),
        ])

        self.assertEqual(answer, self.CLAIM)
        self.assertEqual(len(chat.calls), 4)

    def test_command_that_ran_and_failed_still_verifies(self):
        answer, chat = self.run_scripted([
            *self.EDIT,
            tool_call("run_command", command="cat nonexistent.txt"),
            final_answer(self.CLAIM),
        ])

        self.assertEqual(answer, self.CLAIM)
        self.assertEqual(len(chat.calls), 4)


if __name__ == "__main__":
    unittest.main()
