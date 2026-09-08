"""
Mocked tests for M7.6 -- semantic edit verification.

These tests never talk to a real Ollama server and never touch the real
project directory: `agent.tools.get_project_root` is patched to a
temporary directory for the duration of each test, and the Ollama client
is replaced with a small fake that plays back a scripted sequence of
"model" responses.

Scenarios covered:
  1. The exact failure from the M7.5 testing report: "change comment X
     to Y" -- the model's first instinct is a full write_file rewrite
     that also destroys unrelated content. The pre-execution guard
     should block that call before it ever touches disk and steer the
     model to replace_in_file.
  2. A backstop case where the request doesn't give the guard an exact
     quoted target to work with, so a destructive write_file slips
     through -- post-execution verification should catch it, revert the
     file, and force a retry instead of letting the run report success.
  3. A legitimate, explicitly-requested full-file rewrite should NOT be
     blocked or flagged, and should complete normally.
  4. Existing M7.3 (tool failure recovery) and M7.5 (invalid action
     recovery) behavior still works unmodified.

Run with:  python3 -m pytest tests/test_m7_6_semantic_verification.py -v
       or: python3 -m unittest tests.test_m7_6_semantic_verification -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent.agent as agent_module
import agent.tools as tools_module


# The actual current content of agent/permissions.py in this project.
ORIGINAL_PERMISSIONS_CONTENT = (
    'def request_permission(action: str) -> bool:\n'
    '    """Ask the user for permission to perform an action.\n'
    '\n'
    '    Prints the requested action and prompts the user for a y/N\n'
    '    confirmation, returning True only on an explicit "y".\n'
    '    """\n'
    '\n'
    '    print("\\nIggy needs your permission to:")\n'
    '    print(f"  {action}")\n'
    '\n'
    '    response = input("\\nProceed? [y/N]: ").strip().lower()\n'
    '\n'
    '    return response == "y"\n'
)

# The exact destructive rewrite from the bug report: it DOES contain the
# new comment text, but drops the docstring, the print statements, and
# even the function name/signature -- which is what the destructive-
# change check (not the "target text missing" check) is meant to catch.
DESTRUCTIVE_REWRITE_CONTENT = (
    "# Ask the user to approve an action.\n"
    "def ask_permission(self, action):\n"
    "    approval = input(f'Approve {action}? (y/n): ') == 'y'\n"
    "    return approval\n"
)


def fake_response(content: str):
    """Build an object shaped like ollama's chat() response, i.e.
    supporting `response.message.content`.
    """
    return SimpleNamespace(message=SimpleNamespace(content=content))


def tool_call(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def final_answer(answer):
    return json.dumps({"tool": "none", "answer": answer})


class ScriptedChat:
    """Stand-in for ollama.Client().chat that plays back a fixed list of
    canned model responses, one per call, regardless of what messages
    are passed in. Records the message list at each call for inspection.
    """

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


class M76TestCase(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        self.permissions_path = self.project_root / "agent" / "permissions.py"
        self.permissions_path.parent.mkdir(parents=True, exist_ok=True)
        self.permissions_path.write_text(
            ORIGINAL_PERMISSIONS_CONTENT, encoding="utf-8"
        )

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

    def run_scripted(self, scripted_responses, user_request):
        chat = ScriptedChat(scripted_responses)
        with patch.object(agent_module.client, "chat", side_effect=chat):
            messages = [
                {"role": "system", "content": agent_module.SYSTEM_PROMPT},
                {"role": "user", "content": user_request},
            ]
            answer = agent_module.run_agent(messages)
        return answer, messages, chat

    def current_permissions_content(self):
        return self.permissions_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Scenario 1: exact bug-report failure. The guard should intercept
    # the bad write_file before it ever reaches disk.
    # ------------------------------------------------------------------

    def test_guard_blocks_destructive_write_file_and_model_recovers(self):
        user_request = (
            "Change the comment in `agent/permissions.py` from "
            "`Ask the user for permission to perform an action.` to "
            "`Ask the user to approve an action.`"
        )

        scripted = [
            # 1. Model reads the file first (normal behavior).
            tool_call("read_file", path="agent/permissions.py"),
            # 2. Model's first instinct: a full, destructive write_file.
            #    This should be INTERCEPTED by the guard -- never executed.
            tool_call(
                "write_file",
                path="agent/permissions.py",
                content=DESTRUCTIVE_REWRITE_CONTENT,
            ),
            # 3. After the corrective message, model does the right thing.
            tool_call(
                "replace_in_file",
                path="agent/permissions.py",
                old_text="Ask the user for permission to perform an action.",
                new_text="Ask the user to approve an action.",
            ),
            # 4. Model reports completion.
            final_answer("Updated the comment as requested."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Updated the comment as requested.")

        final_content = self.current_permissions_content()

        # The rest of the function must still be intact -- only the
        # docstring's first line changed.
        self.assertIn("Ask the user to approve an action.", final_content)
        self.assertIn(
            "def request_permission(action: str) -> bool:", final_content
        )
        self.assertIn('print("\\nIggy needs your permission to:")', final_content)
        self.assertIn('response = input("\\nProceed? [y/N]: ").strip().lower()', final_content)
        self.assertIn('return response == "y"', final_content)
        self.assertNotIn(
            "Ask the user for permission to perform an action.",
            final_content,
        )

        # Confirm the guard's corrective message was actually injected
        # (i.e. the third scripted turn's messages contain it), proving
        # write_file was intercepted rather than silently allowed.
        third_call_messages = chat.calls[2]
        guard_message = third_call_messages[-1]["content"]
        self.assertIn("replace_in_file", guard_message)
        self.assertIn("not a full rewrite", guard_message)

    # ------------------------------------------------------------------
    # Scenario 2: backstop. No quoted "from X to Y" in the request, so
    # the pre-execution guard has nothing to key off and a destructive
    # write_file slips through. Post-execution verification must catch
    # it, revert the file, and force a retry.
    # ------------------------------------------------------------------

    def test_post_hoc_verification_catches_and_reverts_destructive_edit(self):
        user_request = (
            "Update the comment in agent/permissions.py so it talks "
            "about approving the action instead of asking permission."
        )

        scripted = [
            tool_call("read_file", path="agent/permissions.py"),
            # Destructive full rewrite, no explicit quoted target in the
            # request, so the pre-execution guard can't intercept this.
            tool_call(
                "write_file",
                path="agent/permissions.py",
                content=DESTRUCTIVE_REWRITE_CONTENT,
            ),
            # Model tries to claim victory immediately -- must be
            # rejected because pending_semantic_recovery is set.
            final_answer("Done."),
            # Forced retry: model makes the correct, small edit instead.
            tool_call(
                "replace_in_file",
                path="agent/permissions.py",
                old_text="Ask the user for permission to perform an action.",
                new_text="Ask the user to approve the action.",
            ),
            final_answer("Updated the comment."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Updated the comment.")

        final_content = self.current_permissions_content()
        self.assertIn("Ask the user to approve the action.", final_content)
        self.assertIn(
            "def request_permission(action: str) -> bool:", final_content
        )
        self.assertIn('return response == "y"', final_content)

        # The premature "Done." must have been rejected: find the user
        # message injected right after that assistant turn and confirm
        # it explains the verification failure and reversion.
        rejection_messages = chat.calls[3]
        rejection_note = rejection_messages[-1]["content"]
        self.assertIn("did not pass verification", rejection_note)
        self.assertIn("reverted", rejection_note)

    def test_destructive_write_file_is_reverted_before_retry(self):
        """More granular check: immediately after the bad write_file
        call, before the model gets another turn, the file on disk must
        already be back to its original content (not left corrupted).
        """
        user_request = (
            "Update the comment in agent/permissions.py so it talks "
            "about approving the action instead of asking permission."
        )

        scripted = [
            tool_call(
                "write_file",
                path="agent/permissions.py",
                content=DESTRUCTIVE_REWRITE_CONTENT,
            ),
        ]

        chat = ScriptedChat(scripted)

        with patch.object(agent_module.client, "chat", side_effect=chat):
            messages = [
                {"role": "system", "content": agent_module.SYSTEM_PROMPT},
                {"role": "user", "content": user_request},
            ]

            # Only let the loop run one iteration by using up the only
            # scripted response and catching the "ran out" assertion is
            # unnecessary -- instead check disk state right after via a
            # thin wrapper. Simplest: just run and inspect disk content
            # once ScriptedChat raises (it will, on the 2nd call), then
            # confirm content was reverted despite the run not finishing.
            with self.assertRaises(AssertionError):
                agent_module.run_agent(messages)

        # The bad write_file DID execute (tool call happened) but M7.6
        # verification should have reverted it immediately afterwards.
        self.assertEqual(
            self.current_permissions_content(),
            ORIGINAL_PERMISSIONS_CONTENT,
        )

    # ------------------------------------------------------------------
    # Scenario 3: legitimate full rewrite, explicitly requested. Must
    # NOT be blocked or flagged.
    # ------------------------------------------------------------------

    def test_legitimate_full_rewrite_is_not_blocked(self):
        user_request = (
            "Rewrite the entire agent/permissions.py file from scratch "
            "to use getpass instead of input for a hidden prompt."
        )

        new_content = (
            "import getpass\n\n\n"
            "def ask_permission(self, action):\n"
            "    response = getpass.getpass(f'Approve {action}? (y/n): ')\n"
            "    return response.strip().lower() == 'y'\n"
        )

        scripted = [
            tool_call("read_file", path="agent/permissions.py"),
            tool_call(
                "write_file",
                path="agent/permissions.py",
                content=new_content,
            ),
            final_answer("Rewrote permissions.py to use getpass."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Rewrote permissions.py to use getpass.")
        self.assertEqual(self.current_permissions_content(), new_content)

        # Make sure the write_file call was never intercepted: the
        # message right after it should be the normal tool-result
        # followup, not a verification-failure notice.
        followup_messages = chat.calls[2]
        followup = followup_messages[-1]["content"]
        self.assertIn("Tool `write_file` returned", followup)
        self.assertNotIn("verification found a problem", followup)

    # ------------------------------------------------------------------
    # Scenario 4: existing M7.3 / M7.5 behavior is untouched.
    # ------------------------------------------------------------------

    def test_m7_3_tool_failure_recovery_still_works(self):
        """replace_in_file failing (text not found) should still force a
        retry via the pre-existing M7.3 mechanism, unaffected by M7.6.
        """
        user_request = "Fix a typo in agent/permissions.py."

        scripted = [
            tool_call(
                "replace_in_file",
                path="agent/permissions.py",
                old_text="this text does not exist in the file",
                new_text="replacement",
            ),
            final_answer("Done."),
            tool_call(
                "replace_in_file",
                path="agent/permissions.py",
                old_text="Ask the user for permission to perform an action.",
                new_text="Ask the user for permission to perform the action.",
            ),
            final_answer("Fixed the typo."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Fixed the typo.")
        self.assertIn(
            "Ask the user for permission to perform the action.",
            self.current_permissions_content(),
        )

        rejection_messages = chat.calls[2]
        rejection_note = rejection_messages[-1]["content"]
        self.assertIn("recoverable tool failure", rejection_note)

    def test_m7_5_invalid_json_recovery_still_works(self):
        user_request = "List the project files."

        scripted = [
            "this is not valid json at all",
            tool_call("list_files", root="."),
            final_answer("Here are the files."),
        ]

        answer, messages, chat = self.run_scripted(scripted, user_request)

        self.assertEqual(answer, "Here are the files.")

        rejection_messages = chat.calls[1]
        rejection_note = rejection_messages[-1]["content"]
        self.assertIn("could not be parsed as JSON", rejection_note)


if __name__ == "__main__":
    unittest.main()
