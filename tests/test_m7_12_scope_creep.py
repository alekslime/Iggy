"""
Mocked tests for M7.12 -- scope creep detection.

Same testing philosophy as the rest of M7.x: no real Ollama server, no
real project directory (`agent.tools.get_project_root` is patched to a
temp dir), model responses are scripted.

This is the fuzziest check in the M7.9-12 group, and deliberately the
most conservative: it only fires when the request contains an explicit
scope-limiting phrase (SCOPE_LIMIT_KEYWORDS, e.g. "only change
`app/config.py`") AND names at least one specific file. Without an
explicit signal, touching more than one file is completely normal
(a bug fix pulling in a related import, an accompanying test update)
and is never flagged.

Scenarios covered:
  1. Unit-level: `check_scope_creep` -- flags an extra file when scope
     language + a named file are both present; stays quiet without
     scope language, without a named file, or when nothing extra was
     touched.
  2. Integration: the one case M7.7's wrong-file redirect can't
     already catch on its own -- an extra edit lands on a file that
     already exists, made *before* the scope-limited request's named
     (not-yet-created) file exists in the project, so M7.7 has no
     project file to redirect to. M7.12 still catches it at
     final-answer time, forces retries, and accepts once the retry cap
     is reached.
  3. Regression: multiple files touched with no filename named in the
     request at all (so neither M7.7 nor M7.12 has anything to check
     against) is never flagged.

Note: in most everyday scope-creep scenarios (a scope-limited request
naming a file that already exists, with an extra edit landing on some
other existing file), M7.7's wrong-file redirect already reverts the
extra edit on its own -- it doesn't care about scope language, only
about whether the edited file matches something the request named.
M7.12 exists for the residual case above, where M7.7's redirect
mechanism has no project file to point to yet.

Run with:  python3 -m pytest tests/test_m7_12_scope_creep.py -v
       or: python3 -m unittest tests.test_m7_12_scope_creep -v
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


class CheckScopeCreepUnitTests(unittest.TestCase):

    def test_flags_extra_file_with_scope_language(self):
        message = agent_module.check_scope_creep(
            "Only change app/config.py, don't touch anything else.",
            {"app/config.py", "app/settings.py"},
        )

        self.assertIsNotNone(message)
        self.assertIn("app/settings.py", message)

    def test_no_scope_language_is_never_flagged(self):
        message = agent_module.check_scope_creep(
            "Update app/config.py.",
            {"app/config.py", "app/settings.py"},
        )

        self.assertIsNone(message)

    def test_no_named_file_is_never_flagged(self):
        message = agent_module.check_scope_creep(
            "Only fix the bug, don't touch anything else.",
            {"app/config.py", "app/settings.py"},
        )

        self.assertIsNone(message)

    def test_no_extra_file_is_not_flagged(self):
        message = agent_module.check_scope_creep(
            "Only change app/config.py, don't touch anything else.",
            {"app/config.py"},
        )

        self.assertIsNone(message)

    def test_nothing_modified_yet_is_not_flagged(self):
        message = agent_module.check_scope_creep(
            "Only change app/config.py, don't touch anything else.",
            set(),
        )

        self.assertIsNone(message)


class M712IntegrationTests(unittest.TestCase):

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
    # The one scenario M7.7's wrong-file redirect can't already catch:
    # the scope-limited request's named file doesn't exist yet (it's
    # being created), so M7.7 has no project file to redirect an
    # extra edit to -- until M7.12 catches it at final-answer time.
    # -------------------------------------------------------------

    def test_extra_file_before_named_file_exists_is_detected(self):
        self.make_file("app/other.py", "TIMEOUT = 10\n")

        user_request = (
            "Only create app/newmod.py with a greet function. "
            "Don't touch anything else."
        )

        scripted = [
            # An extra, unrequested edit to a file that already
            # exists -- made *before* app/newmod.py exists, so M7.7
            # has no project file matching the mention to redirect to.
            tool_call("read_file", path="app/other.py"),
            tool_call(
                "replace_in_file",
                path="app/other.py",
                old_text="TIMEOUT = 10",
                new_text="TIMEOUT = 20",
            ),
            # Now create the actually-requested file.
            tool_call(
                "write_file",
                path="app/newmod.py",
                content="def greet(name):\n    print(f'hi {name}')\n",
            ),
            final_answer("Created newmod.py."),  # retry 1/2: extra file
            final_answer(
                "Created newmod.py; also bumped TIMEOUT in other.py."
            ),  # retry 2/2
            final_answer("Created newmod.py and updated other.py."),  # cap
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(
            answer, "Created newmod.py and updated other.py."
        )

        user_messages = [
            m["content"] for m in chat.calls[-1] if m["role"] == "user"
        ]
        scope_warning = next(
            (m for m in user_messages if "asked for the change to be limited" in m),
            None,
        )
        self.assertIsNotNone(scope_warning)
        self.assertIn("app/other.py", scope_warning)

    # -------------------------------------------------------------
    # Regression: multiple files touched, no explicit scope-limiting
    # language and no filename named at all in the request -- never
    # flagged (by either M7.7 or M7.12).
    # -------------------------------------------------------------

    def test_no_filename_named_multi_file_edit_is_never_flagged(self):
        self.make_file("app/config.py", "DEBUG = True\n")
        self.make_file("app/settings.py", "TIMEOUT = 10\n")

        user_request = "Fix the timeout bug."

        scripted = [
            tool_call("read_file", path="app/config.py"),
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            tool_call("read_file", path="app/settings.py"),
            tool_call(
                "replace_in_file",
                path="app/settings.py",
                old_text="TIMEOUT = 10",
                new_text="TIMEOUT = 20",
            ),
            final_answer("Fixed it in both files."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Fixed it in both files.")
        self.assertEqual(len(chat.calls), 5)


if __name__ == "__main__":
    unittest.main()
