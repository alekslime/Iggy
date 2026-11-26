"""
Tests for run_command's output cap, timeout handling and stdin.

Output is capped (head + tail kept, middle dropped with a note) so a
chatty command can't flood a small model's context. A timeout raises a
clear TimeoutError carrying whatever output was captured. stdin is
closed so a command waiting for input can't hang or steal the TUI's
terminal.

Run with:  python3 -m pytest tests/test_command_output.py -v
       or: python3 -m unittest tests.test_command_output -v
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seacode import tools as tools_module


class TruncateOutputTests(unittest.TestCase):

    def test_short_text_unchanged(self):
        self.assertEqual(tools_module.truncate_output("hello", 100), "hello")

    def test_text_at_limit_unchanged(self):
        text = "x" * 100
        self.assertEqual(tools_module.truncate_output(text, 100), text)

    def test_long_text_keeps_head_and_tail(self):
        text = "HEAD" + "m" * 1000 + "TAIL"
        result = tools_module.truncate_output(text, 100)

        self.assertTrue(result.startswith("HEAD"))
        self.assertTrue(result.endswith("TAIL"))
        self.assertIn("characters omitted", result)
        self.assertLess(len(result), 200)

    def test_omitted_count_is_accurate(self):
        text = "a" * 1000
        result = tools_module.truncate_output(text, 100)

        # 60 head + 40 tail kept, 900 dropped.
        marker = "\n[... 900 characters omitted ...]\n"
        self.assertEqual(result, "a" * 60 + marker + "a" * 40)

    def test_default_limit_applies(self):
        text = "z" * (tools_module.MAX_COMMAND_OUTPUT_CHARS * 5)
        result = tools_module.truncate_output(text)

        self.assertLess(
            len(result), tools_module.MAX_COMMAND_OUTPUT_CHARS + 100
        )


class RunCommandLimitsTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        env = patch.dict(os.environ, {"SEACODE_DIR": self._tmp.name})
        env.start()
        self.addCleanup(env.stop)

        perm = patch.object(
            tools_module, "request_permission", return_value=True
        )
        perm.start()
        self.addCleanup(perm.stop)

    def test_large_output_is_capped(self):
        big = "line\n" * 50_000 + "SUMMARY: 3 passed"
        fake = subprocess.CompletedProcess(["pytest"], 0, big, "")

        with patch.object(tools_module.subprocess, "run", return_value=fake):
            result = tools_module.run_command("pytest -q")

        self.assertLess(
            len(result), tools_module.MAX_COMMAND_OUTPUT_CHARS + 100
        )
        self.assertIn("characters omitted", result)
        self.assertTrue(result.endswith("SUMMARY: 3 passed"))

    def test_small_output_untouched(self):
        fake = subprocess.CompletedProcess(["ls"], 0, "a.txt\n", "warn\n")

        with patch.object(tools_module.subprocess, "run", return_value=fake):
            result = tools_module.run_command("ls")

        self.assertEqual(result, "a.txt\nwarn")

    def test_timeout_raises_clear_error_with_partial_output(self):
        error = subprocess.TimeoutExpired(
            ["pytest"], 30, output=b"collected 3 items\n", stderr=b"warn\n"
        )

        with patch.object(tools_module.subprocess, "run", side_effect=error):
            with self.assertRaises(TimeoutError) as ctx:
                tools_module.run_command("pytest")

        message = str(ctx.exception)
        self.assertIn("timed out after 30s", message)
        self.assertIn("killed", message)
        self.assertIn("collected 3 items", message)

    def test_timeout_without_output(self):
        error = subprocess.TimeoutExpired(["pytest"], 30)

        with patch.object(tools_module.subprocess, "run", side_effect=error):
            with self.assertRaises(TimeoutError) as ctx:
                tools_module.run_command("pytest")

        self.assertNotIn("Partial output", str(ctx.exception))

    def test_timeout_partial_output_is_capped(self):
        error = subprocess.TimeoutExpired(
            ["pytest"], 30, output=b"x" * 100_000
        )

        with patch.object(tools_module.subprocess, "run", side_effect=error):
            with self.assertRaises(TimeoutError) as ctx:
                tools_module.run_command("pytest")

        self.assertLess(
            len(str(ctx.exception)),
            tools_module.MAX_COMMAND_OUTPUT_CHARS + 300,
        )

    def test_stdin_is_closed(self):
        fake = subprocess.CompletedProcess(["ls"], 0, "", "")

        with patch.object(
            tools_module.subprocess, "run", return_value=fake
        ) as run:
            tools_module.run_command("ls")

        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)


@unittest.skipUnless(shutil.which("python"), "python not on PATH")
class RealSubprocessTests(unittest.TestCase):
    """End to end against a real process (no mocking of subprocess)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        env = patch.dict(os.environ, {"SEACODE_DIR": self._tmp.name})
        env.start()
        self.addCleanup(env.stop)

        perm = patch.object(
            tools_module, "request_permission", return_value=True
        )
        perm.start()
        self.addCleanup(perm.stop)

    def test_real_timeout_kills_and_reports(self):
        script = Path(self._tmp.name) / "slow.py"
        script.write_text(
            "import time\nprint('started', flush=True)\ntime.sleep(10)\n"
        )

        with patch.object(tools_module, "COMMAND_TIMEOUT_SECONDS", 1):
            with self.assertRaises(TimeoutError) as ctx:
                tools_module.run_command("python slow.py")

        self.assertIn("timed out after 1s", str(ctx.exception))
        self.assertIn("started", str(ctx.exception))

    def test_command_reading_stdin_does_not_hang(self):
        script = Path(self._tmp.name) / "readin.py"
        script.write_text(
            "import sys\nprint(repr(sys.stdin.read()))\n"
        )

        result = tools_module.run_command("python readin.py")

        self.assertEqual(result, "''")

    def test_real_large_output_capped(self):
        script = Path(self._tmp.name) / "noisy.py"
        script.write_text("print('x' * 100000)\nprint('END')\n")

        result = tools_module.run_command("python noisy.py")

        self.assertLess(
            len(result), tools_module.MAX_COMMAND_OUTPUT_CHARS + 100
        )
        self.assertTrue(result.endswith("END"))


if __name__ == "__main__":
    unittest.main()
