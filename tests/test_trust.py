"""
Tests for the permission trust policy (seacode.permissions).

Levels: ask (prompt everything), edits (auto-approve file changes),
all (auto-approve everything). Precedence is set_trust_level() >
SEACODE_TRUST > seacode.toml `trust` > ask.

Run with:  python3 -m pytest tests/test_trust.py -v
       or: python3 -m unittest tests.test_trust -v
"""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from seacode import permissions as permissions_module


EDIT_ACTIONS = [
    "modify file: a.txt",
    "delete file: a.txt",
    "restore file from backup: a.txt",
]

COMMAND_ACTIONS = ["run: ls"]


class TrustLevelTests(unittest.TestCase):

    def setUp(self):
        self.addCleanup(permissions_module.reset_trust_level)

    def test_default_is_ask(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_TRUST", None)
            self.assertEqual(
                permissions_module.get_trust_level(), "ask"
            )

    def test_explicit_wins_over_env_and_config(self):
        permissions_module.set_trust_level("all")

        with patch.dict(os.environ, {"SEACODE_TRUST": "edits"}):
            self.assertEqual(
                permissions_module.get_trust_level({"trust": "ask"}),
                "all",
            )

    def test_env_wins_over_config(self):
        with patch.dict(os.environ, {"SEACODE_TRUST": "edits"}):
            self.assertEqual(
                permissions_module.get_trust_level({"trust": "all"}),
                "edits",
            )

    def test_config_value_used(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_TRUST", None)
            self.assertEqual(
                permissions_module.get_trust_level({"trust": "edits"}),
                "edits",
            )

    def test_invalid_explicit_raises(self):
        with self.assertRaises(ValueError):
            permissions_module.set_trust_level("bogus")

    def test_invalid_env_raises(self):
        with patch.dict(os.environ, {"SEACODE_TRUST": "bogus"}):
            with self.assertRaises(ValueError):
                permissions_module.get_trust_level()

    def test_invalid_config_raises(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_TRUST", None)
            with self.assertRaises(ValueError):
                permissions_module.get_trust_level({"trust": "bogus"})


class ActionKindTests(unittest.TestCase):

    def test_edit_actions(self):
        for action in EDIT_ACTIONS:
            self.assertEqual(
                permissions_module.action_kind(action), "edit"
            )

    def test_command_actions(self):
        for action in COMMAND_ACTIONS:
            self.assertEqual(
                permissions_module.action_kind(action), "command"
            )

    def test_unknown_action_is_other(self):
        self.assertEqual(
            permissions_module.action_kind("do something weird"),
            "other",
        )


class AutoApproveTests(unittest.TestCase):

    def setUp(self):
        self.addCleanup(permissions_module.reset_trust_level)

    def test_ask_approves_nothing(self):
        permissions_module.set_trust_level("ask")

        for action in EDIT_ACTIONS + COMMAND_ACTIONS + ["weird"]:
            self.assertFalse(
                permissions_module.should_auto_approve(action)
            )

    def test_edits_approves_only_edits(self):
        permissions_module.set_trust_level("edits")

        for action in EDIT_ACTIONS:
            self.assertTrue(
                permissions_module.should_auto_approve(action)
            )

        for action in COMMAND_ACTIONS + ["weird"]:
            self.assertFalse(
                permissions_module.should_auto_approve(action)
            )

    def test_all_approves_everything(self):
        permissions_module.set_trust_level("all")

        for action in EDIT_ACTIONS + COMMAND_ACTIONS + ["weird"]:
            self.assertTrue(
                permissions_module.should_auto_approve(action)
            )


class RequestPermissionTests(unittest.TestCase):

    def setUp(self):
        self.addCleanup(permissions_module.reset_trust_level)

    def test_auto_approve_never_prompts(self):
        permissions_module.set_trust_level("all")
        output = io.StringIO()

        with (
            patch(
                "builtins.input",
                side_effect=AssertionError("must not prompt"),
            ),
            redirect_stdout(output),
        ):
            approved = permissions_module.request_permission("run: ls")

        self.assertTrue(approved)
        self.assertIn("✓ run: ls (auto-approved, trust=all)", output.getvalue())

    def test_ask_prompts_and_honors_answer(self):
        permissions_module.set_trust_level("ask")

        with patch("builtins.input", return_value="y"):
            self.assertTrue(
                permissions_module.request_permission("run: ls")
            )

        with patch("builtins.input", return_value="n"):
            self.assertFalse(
                permissions_module.request_permission("run: ls")
            )


if __name__ == "__main__":
    unittest.main()
