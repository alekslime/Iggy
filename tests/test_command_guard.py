"""
Tests for run_command's argument guard (seacode.tools.validate_command_args).

The allowlist only checks the command *name*; the guard additionally
refuses arguments that reach outside the project (../, absolute paths,
~, --opt=/path, symlinks that escape) and options that run other
programs (find -exec, git -c, ...). It runs before the permission
prompt, so a refused command never asks the user anything.

Run with:  python3 -m pytest tests/test_command_guard.py -v
       or: python3 -m unittest tests.test_command_guard -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seacode import tools as tools_module


class GuardTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        base = Path(self._tmp.name).resolve()
        self.root = base / "project"
        self.outside = base / "outside"
        self.root.mkdir()
        self.outside.mkdir()

        (self.root / "notes.txt").write_text("hello\n")
        (self.outside / "secret.txt").write_text("secret\n")

        env = patch.dict(os.environ, {"SEACODE_DIR": str(self.root)})
        env.start()
        self.addCleanup(env.stop)

    def check(self, *args):
        tools_module.validate_command_args(list(args), self.root)

    def assertBlocked(self, *args):
        with self.assertRaises(PermissionError):
            self.check(*args)


class PathArgumentTests(GuardTestCase):

    def test_parent_traversal_blocked(self):
        self.assertBlocked("cat", "../outside/secret.txt")

    def test_absolute_path_outside_blocked(self):
        self.assertBlocked("cat", str(self.outside / "secret.txt"))
        self.assertBlocked("grep", "-r", "secret", "/")

    def test_home_shortcut_blocked(self):
        self.assertBlocked("cat", "~/.ssh/id_rsa")

    def test_long_option_value_blocked(self):
        self.assertBlocked("git", "--git-dir=/etc")
        self.assertBlocked("grep", "--file=/etc/passwd", "x")

    def test_short_option_attached_value_blocked(self):
        self.assertBlocked("grep", "-f/etc/passwd", "x")
        self.assertBlocked("head", "-c5", "../outside/secret.txt")

    def test_git_dash_C_value_blocked(self):
        self.assertBlocked("git", "-C", "..", "status")

    def test_symlink_escape_blocked(self):
        link = self.root / "sneaky"
        try:
            link.symlink_to(self.outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")

        self.assertBlocked("cat", "sneaky/secret.txt")

    def test_paths_inside_project_allowed(self):
        self.check("cat", "notes.txt")
        self.check("cat", "./notes.txt")
        self.check("ls", ".")
        self.check("cat", str(self.root / "notes.txt"))
        self.check("python", "script.py")

    def test_plain_flags_and_patterns_allowed(self):
        self.check("grep", "-rn", "def foo(", ".")
        self.check("grep", "a..b", "notes.txt")
        self.check("pytest", "tests/", "-q", "-x")
        self.check("git", "log", "HEAD~1..HEAD", "--oneline")
        self.check("head", "-n", "5", "notes.txt")


class FindTests(GuardTestCase):

    def test_exec_and_write_flags_blocked(self):
        for flag in ("-exec", "-execdir", "-ok", "-okdir", "-delete",
                     "-fprint", "-fls"):
            with self.subTest(flag=flag):
                self.assertBlocked("find", ".", flag, "x")

    def test_plain_find_allowed(self):
        self.check("find", ".", "-name", "*.py", "-type", "f")

    def test_find_outside_root_blocked(self):
        self.assertBlocked("find", "/", "-name", "id_rsa")


class GitTests(GuardTestCase):

    def test_global_config_flags_blocked(self):
        self.assertBlocked("git", "-c", "core.pager=sh", "log")
        self.assertBlocked("git", "-ccore.pager=sh", "log")
        self.assertBlocked("git", "--config-env=x=Y", "log")
        self.assertBlocked("git", "--exec-path=/tmp", "log")

    def test_helper_program_flags_blocked(self):
        self.assertBlocked("git", "diff", "--ext-diff")
        self.assertBlocked("git", "show", "--textconv", "HEAD")
        self.assertBlocked("git", "fetch", "--upload-pack=evil")

    def test_dash_c_after_subcommand_is_just_a_flag(self):
        self.check("git", "log", "-c")
        self.check("git", "show", "-c", "HEAD")

    def test_common_git_commands_allowed(self):
        self.check("git", "status", "--short")
        self.check("git", "diff", "--stat")
        self.check("git", "log", "--oneline", "-5")
        self.check("git", "-C", ".", "status")


class RunCommandIntegrationTests(GuardTestCase):

    def test_blocked_command_never_prompts(self):
        with patch.object(
            tools_module, "request_permission", return_value=True
        ) as prompt:
            with self.assertRaises(PermissionError):
                tools_module.run_command("cat ../outside/secret.txt")

        prompt.assert_not_called()

    def test_allowed_command_still_runs(self):
        with patch.object(
            tools_module, "request_permission", return_value=True
        ):
            output = tools_module.run_command("cat notes.txt")

        self.assertEqual(output, "hello")

    def test_denied_by_user_still_works(self):
        with patch.object(
            tools_module, "request_permission", return_value=False
        ):
            output = tools_module.run_command("cat notes.txt")

        self.assertEqual(output, "Command denied by user.")


if __name__ == "__main__":
    unittest.main()
