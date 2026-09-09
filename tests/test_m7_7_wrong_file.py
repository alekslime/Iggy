"""
Mocked tests for M7.7 -- wrong-file edit detection.

Same testing philosophy as M7.6: no real Ollama server, no real project
directory (`agent.tools.get_project_root` is patched to a temp dir), and
the model is a `ScriptedChat` that plays back a fixed response list.

Scenarios covered:
  1. Unit-level: `extract_mentioned_filenames` pulls filename-like
     tokens out of a request and ignores unrelated "word.word" text.
  2. Unit-level: `verify_target_file` -- clean match, clear mismatch
     with one redirect candidate, and ambiguous match/mismatch with
     multiple candidates.
  3. Integration: the request clearly names one file, but the model
     edits a different one that happens to contain similar text (so
     the tool call succeeds and M7.6's content checks pass too) --
     M7.7 should catch it, revert, and force a retry onto the named
     file.
  4. Integration: the request names only a bare filename that's
     ambiguous across the project (two files share that name). Every
     attempt is unresolvable, so this exercises the retry cap:
     2 forced retries, then the model's next "success" claim is
     accepted -- but every ambiguous edit still gets reverted, so the
     files are never left silently modified even though the run
     "succeeds".
  5. Regression: an edit that clearly matches what was named (full
     path, or a bare name with only one project match) is NOT flagged,
     and a request with no filename at all is NOT checked (M7.7 must
     not fire when it has no evidence).

Run with:  python3 -m pytest tests/test_m7_7_wrong_file.py -v
       or: python3 -m unittest tests.test_m7_7_wrong_file -v
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
    """Same scripted-response stand-in used by the M7.6 tests."""

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


# ---------------------------------------------------------------------
# Unit tests: extract_mentioned_filenames / verify_target_file directly,
# no agent loop or fake model involved.
# ---------------------------------------------------------------------


class ExtractMentionedFilenamesTests(unittest.TestCase):

    def test_finds_bare_and_path_filenames(self):
        text = "Fix the bug in utils.py, it's called from agent/tools.py"
        found = agent_module.extract_mentioned_filenames(text)
        self.assertIn("utils.py", found)
        self.assertIn("agent/tools.py", found)

    def test_ignores_unrelated_word_dot_word(self):
        text = "e.g. version 2.0 should work fine, see p.s. below"
        found = agent_module.extract_mentioned_filenames(text)
        self.assertEqual(found, set())

    def test_strips_surrounding_quotes_and_leading_dotslash(self):
        text = 'Update `./agent/tools.py` and "config.json" please'
        found = agent_module.extract_mentioned_filenames(text)
        self.assertIn("agent/tools.py", found)
        self.assertIn("config.json", found)

    def test_no_filenames_returns_empty_set(self):
        text = "Please fix the bug where the counter resets incorrectly."
        self.assertEqual(agent_module.extract_mentioned_filenames(text), set())


class VerifyTargetFileTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        patcher = patch.object(
            tools_module, "get_project_root", return_value=self.project_root
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_file(self, relative_path, content="placeholder"):
        full_path = self.project_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def test_no_mentioned_filenames_skips_check(self):
        self.make_file("app/config.py")
        result = agent_module.verify_target_file(
            "app/settings.py", "Fix the bug where things break."
        )
        self.assertIsNone(result)

    def test_full_path_match_is_not_flagged(self):
        self.make_file("app/config.py")
        result = agent_module.verify_target_file(
            "app/config.py", "Update `app/config.py` to fix the DEBUG flag."
        )
        self.assertIsNone(result)

    def test_bare_name_match_with_single_candidate_is_not_flagged(self):
        self.make_file("app/config.py")
        result = agent_module.verify_target_file(
            "app/config.py", "Update config.py to fix the DEBUG flag."
        )
        self.assertIsNone(result)

    def test_bare_name_match_with_multiple_candidates_is_ambiguous(self):
        self.make_file("app/config.py")
        self.make_file("lib/config.py")
        result = agent_module.verify_target_file(
            "lib/config.py", "Update config.py to fix the DEBUG flag."
        )
        self.assertIsNotNone(result)
        self.assertIn("app/config.py", result)
        self.assertIn("lib/config.py", result)

    def test_clear_mismatch_with_single_candidate_redirects(self):
        self.make_file("app/config.py")
        self.make_file("app/settings.py")
        result = agent_module.verify_target_file(
            "app/settings.py", "Update config.py to fix the DEBUG flag."
        )
        self.assertIsNotNone(result)
        self.assertIn("app/config.py", result)

    def test_mismatch_with_no_project_candidate_is_not_flagged(self):
        # Mentioned filename doesn't exist anywhere in the project --
        # not enough evidence to call this wrong (e.g. a new file).
        self.make_file("app/settings.py")
        result = agent_module.verify_target_file(
            "app/settings.py", "Create a new file called config.py."
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------
# Integration tests: full run_agent loop with a scripted fake model.
# ---------------------------------------------------------------------


class M77IntegrationTests(unittest.TestCase):

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
        return (self.project_root / relative_path).read_text(encoding="utf-8")

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
    # Scenario: clear wrong-file edit, single redirect candidate.
    # The wrong file coincidentally contains matching text, so the
    # tool call succeeds and M7.6's content checks pass -- only M7.7
    # catches this.
    # -------------------------------------------------------------

    def test_clear_wrong_file_edit_is_reverted_and_retried(self):
        self.make_file("app/config.py", "DEBUG = True\nNAME = 'app'\n")
        self.make_file("app/settings.py", "DEBUG = True\nTIMEOUT = 10\n")

        user_request = "Set DEBUG to False in `app/config.py`."

        scripted = [
            tool_call("read_file", path="app/config.py"),
            # Wrong file -- but the replacement text also exists here,
            # so the tool call itself succeeds cleanly. (Also read
            # first, so this exercises M7.7's wrong-file detection
            # specifically, not M7.9's unread-file guard.)
            tool_call("read_file", path="app/settings.py"),
            tool_call(
                "replace_in_file",
                path="app/settings.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            final_answer("Done."),  # should be rejected
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            final_answer("Updated DEBUG in app/config.py."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Updated DEBUG in app/config.py.")

        # The wrong file was reverted...
        self.assertEqual(
            self.read_file("app/settings.py"), "DEBUG = True\nTIMEOUT = 10\n"
        )
        # ...and the right one ended up correctly modified.
        self.assertEqual(
            self.read_file("app/config.py"), "DEBUG = False\nNAME = 'app'\n"
        )

        # Confirm the forced-retry message actually named the mistake.
        third_call_messages = chat.calls[3]
        guard_message = third_call_messages[-1]["content"]
        self.assertIn("which file was modified", guard_message)
        self.assertIn("app/config.py", guard_message)

    # -------------------------------------------------------------
    # Scenario: unresolvable ambiguity. Retries exhaust at the cap,
    # the model's next claim of success is accepted -- but every
    # attempt still gets reverted, so nothing is silently left wrong.
    # -------------------------------------------------------------

    def test_ambiguous_target_exhausts_retries_but_always_reverts(self):
        self.make_file("app/config.py", "TIMEOUT = 10\n")
        self.make_file("lib/config.py", "TIMEOUT = 10\n")

        user_request = "Change the TIMEOUT value in config.py to 30."

        scripted = [
            tool_call("read_file", path="lib/config.py"),
            tool_call(
                "replace_in_file",
                path="lib/config.py",
                old_text="TIMEOUT = 10",
                new_text="TIMEOUT = 30",
            ),
            final_answer("Done."),  # rejected, retry 1/2
            tool_call("read_file", path="app/config.py"),
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="TIMEOUT = 10",
                new_text="TIMEOUT = 30",
            ),
            final_answer("Done."),  # rejected, retry 2/2
            tool_call(
                "replace_in_file",
                path="lib/config.py",
                old_text="TIMEOUT = 10",
                new_text="TIMEOUT = 30",
            ),
            final_answer("Done, I updated config.py."),  # accepted: cap reached
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Done, I updated config.py.")

        # Every attempt was ambiguous and should have been reverted --
        # neither file should actually carry the change, even though
        # the run "succeeded".
        self.assertEqual(self.read_file("app/config.py"), "TIMEOUT = 10\n")
        self.assertEqual(self.read_file("lib/config.py"), "TIMEOUT = 10\n")

    # -------------------------------------------------------------
    # Regression: a correctly-targeted edit is never flagged, and
    # runs exactly like it would without M7.7 present.
    # -------------------------------------------------------------

    def test_correct_single_file_edit_is_not_flagged(self):
        self.make_file("app/config.py", "DEBUG = True\n")

        user_request = "Set DEBUG to False in app/config.py."

        scripted = [
            tool_call("read_file", path="app/config.py"),
            tool_call(
                "replace_in_file",
                path="app/config.py",
                old_text="DEBUG = True",
                new_text="DEBUG = False",
            ),
            final_answer("Updated DEBUG in app/config.py."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Updated DEBUG in app/config.py.")
        self.assertEqual(self.read_file("app/config.py"), "DEBUG = False\n")
        # No forced-retry turn should have been injected.
        self.assertEqual(len(chat.calls), 3)


if __name__ == "__main__":
    unittest.main()
