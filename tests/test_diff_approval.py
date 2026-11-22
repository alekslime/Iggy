"""
Tests for diff-before-approval: modification tools compute their
proposed unified diff and pass it to request_permission, which shows
it before the y/N prompt (and on auto-approval).

Run with:  python3 -m pytest tests/test_diff_approval.py -v
       or: python3 -m unittest tests.test_diff_approval -v
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import seacode.tools as tools_module
from seacode import permissions as permissions_module


class MakeDiffTests(unittest.TestCase):

    def test_changed_lines(self):
        diff = tools_module.make_diff("a.py", "x = 1\n", "x = 2\n")
        self.assertIn("--- a/a.py", diff)
        self.assertIn("-x = 1", diff)
        self.assertIn("+x = 2", diff)

    def test_no_change_empty(self):
        self.assertEqual(tools_module.make_diff("a.py", "same\n", "same\n"), "")


class ToolDiffCaptureTests(unittest.TestCase):
    """Each gated modification passes a diff describing its change."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)
        root_patcher = patch.object(
            tools_module, "get_project_root",
            return_value=self.project_root,
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)
        self.captured = {}

        def fake_permission(action, diff=None):
            self.captured["action"] = action
            self.captured["diff"] = diff
            return True

        perm_patcher = patch.object(
            tools_module, "request_permission",
            side_effect=fake_permission,
        )
        perm_patcher.start()
        self.addCleanup(perm_patcher.stop)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_replace_shows_one_line_change(self):
        (self.project_root / "a.py").write_text("x = 1\ny = 2\n")
        result = tools_module.replace_in_file("a.py", "x = 1", "x = 9")
        self.assertIn("successfully", result)
        self.assertIn("-x = 1", self.captured["diff"])
        self.assertIn("+x = 9", self.captured["diff"])
        self.assertNotIn("y = 2", self.captured["diff"].replace(" y = 2", ""))

    def test_write_existing_shows_diff(self):
        (self.project_root / "a.txt").write_text("old\n")
        tools_module.write_file("a.txt", "new\n")
        self.assertIn("-old", self.captured["diff"])
        self.assertIn("+new", self.captured["diff"])

    def test_write_new_file_shows_additions(self):
        tools_module.write_file("b.txt", "fresh\n")
        self.assertIn("+fresh", self.captured["diff"])

    def test_delete_shows_removals(self):
        (self.project_root / "a.txt").write_text("gone\n")
        tools_module.delete_file("a.txt")
        self.assertIn("-gone", self.captured["diff"])

    def test_undo_shows_restoration_diff(self):
        path = self.project_root / "a.txt"
        path.write_text("v1\n")
        tools_module.write_file("a.txt", "v2\n")
        self.captured.clear()
        result = tools_module.undo_last_change("a.txt")
        self.assertIn("Restored", result)
        self.assertIn("-v2", self.captured["diff"])
        self.assertIn("+v1", self.captured["diff"])
        self.assertEqual(path.read_text(), "v1\n")


class PermissionPrintingTests(unittest.TestCase):

    def tearDown(self):
        permissions_module.reset_trust_level()

    def test_diff_printed_before_prompt(self):
        diff = "--- a/a.py\n+++ b/a.py\n-x = 1\n+x = 2\n"
        with patch("builtins.input", return_value="y") as mock_input:
            out = io.StringIO()
            with redirect_stdout(out):
                approved = permissions_module.request_permission(
                    "modify file: a.py", diff=diff
                )
        self.assertTrue(approved)
        text = out.getvalue()
        self.assertLess(text.index("+x = 2"), text.index("Proceed?")
                        if "Proceed?" in text else len(text))
        mock_input.assert_called_once()

    def test_diff_truncated(self):
        diff = "\n".join(f"+line{i}" for i in range(200))
        with patch("builtins.input", return_value="n"):
            out = io.StringIO()
            with redirect_stdout(out):
                approved = permissions_module.request_permission(
                    "modify file: a.py", diff=diff
                )
        self.assertFalse(approved)
        self.assertIn("more diff lines", out.getvalue())

    def test_no_diff_still_prompts(self):
        with patch("builtins.input", return_value="y"):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertTrue(
                    permissions_module.request_permission("run: ls")
                )
        self.assertIn("? run: ls", out.getvalue())


if __name__ == "__main__":
    unittest.main()
