"""
Mocked tests for M7.8 -- false verification claim detection.

Same testing philosophy as M7.6/M7.7: no real Ollama server, no real
project directory (`agent.tools.get_project_root` is patched to a temp
dir), model responses are scripted.

Scenarios covered:
  1. Unit-level: `claims_verification` recognizes the keyword set and
     doesn't fire on plain, non-claiming answers.
  2. Integration: model modifies a file, then immediately claims in its
     final answer that it "verified" the change without calling any
     verification-capable tool -- should be rejected and forced to
     either actually verify or rephrase.
  3. Integration: model modifies a file, then actually re-reads it
     (or runs a command) before claiming verification -- should be
     accepted immediately, no forced retry.
  4. Integration: model modifies a file and gives a final answer that
     does NOT claim verification at all ("I made the change.") --
     should be accepted immediately; M7.8 only catches false claims,
     not unverified changes in general.
  5. Regression: a pure information request (no modification at all)
     that happens to say "confirmed" is not flagged -- the check only
     applies after a successful modification.
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


class ClaimsVerificationTests(unittest.TestCase):

    def test_recognizes_common_claim_phrasings(self):
        self.assertTrue(
            agent_module.claims_verification("I verified the change works.")
        )
        self.assertTrue(
            agent_module.claims_verification("Done -- tests pass now.")
        )
        self.assertTrue(
            agent_module.claims_verification(
                "I ran the tests and confirmed the fix."
            )
        )

    def test_plain_answer_is_not_a_claim(self):
        self.assertFalse(
            agent_module.claims_verification("I updated the DEBUG flag.")
        )
        self.assertFalse(agent_module.claims_verification(""))
        self.assertFalse(agent_module.claims_verification(None))


class M78IntegrationTests(unittest.TestCase):

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
    # False claim with no verification tool call at all -- rejected.
    # -------------------------------------------------------------

    def test_false_verification_claim_is_rejected_and_retried(self):
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
            # Claims verification with zero tool calls since the edit.
            final_answer("Done, I verified the change works correctly."),
            # After correction, drops the false claim.
            final_answer("Done, I updated DEBUG to False."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Done, I updated DEBUG to False.")

        # Confirm a correction was actually injected.
        fourth_call_messages = chat.calls[3]
        correction = fourth_call_messages[-1]["content"]
        self.assertIn("claims the change was verified", correction)

    # -------------------------------------------------------------
    # Model actually re-reads the file before claiming verification --
    # accepted immediately, no forced retry.
    # -------------------------------------------------------------

    def test_real_verification_before_claim_is_accepted(self):
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
            # Actually re-reads the file after the edit.
            tool_call("read_file", path="app/config.py"),
            final_answer("Done, I verified the change works correctly."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Done, I verified the change works correctly.")
        # No correction turn should have been injected -- 4 model calls,
        # one per scripted response, none repeated.
        self.assertEqual(len(chat.calls), 4)

    # -------------------------------------------------------------
    # Model doesn't claim verification at all -- never flagged.
    # -------------------------------------------------------------

    def test_no_verification_claim_is_never_flagged(self):
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
            final_answer("I updated DEBUG to False in app/config.py."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "I updated DEBUG to False in app/config.py.")
        self.assertEqual(len(chat.calls), 3)

    # -------------------------------------------------------------
    # Pure info request that happens to use "confirmed" -- not flagged,
    # since no modification happened in this run at all.
    # -------------------------------------------------------------

    def test_info_request_with_claim_language_is_not_flagged(self):
        self.make_file("app/config.py", "DEBUG = True\n")

        user_request = "What is the current value of DEBUG in app/config.py?"

        scripted = [
            tool_call("read_file", path="app/config.py"),
            final_answer("I confirmed DEBUG is currently set to True."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "I confirmed DEBUG is currently set to True.")
        self.assertEqual(len(chat.calls), 2)

    # -------------------------------------------------------------
    # Retry cap: after MAX_VERIFICATION_CLAIM_RETRY_ATTEMPTS forced
    # retries, a repeated false claim is finally accepted as-is.
    # -------------------------------------------------------------

    def test_retry_cap_eventually_accepts_the_claim(self):
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
            final_answer("Verified: the change works."),  # retry 1/2
            final_answer("Confirmed it works."),  # retry 2/2
            final_answer("Tests pass, verified."),  # accepted: cap reached
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Tests pass, verified.")


if __name__ == "__main__":
    unittest.main()
