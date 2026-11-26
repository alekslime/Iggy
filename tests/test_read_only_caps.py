"""
Tests for the caps on what the model is shown from the read-only
tools (seacode.tools.cap_tool_result, applied in agent.execute_tool)
and for git_status's timeout.

The tools themselves still return everything -- the M7 checks and the
TUI depend on that -- so these tests check both layers.

Run with:  python3 -m pytest tests/test_read_only_caps.py -v
       or: python3 -m unittest tests.test_read_only_caps -v
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.tools as tools_module
from seacode.tools import cap_tool_result


class ListFilesCapTests(unittest.TestCase):

    def test_under_cap_unchanged(self):
        files = [f"f{i}.txt" for i in range(10)]
        self.assertEqual(cap_tool_result("list_files", files), files)

    def test_over_cap_truncated_with_note(self):
        n = tools_module.MAX_LIST_FILES_ENTRIES + 25
        files = [f"f{i}.txt" for i in range(n)]

        result = cap_tool_result("list_files", files)

        self.assertEqual(len(result), tools_module.MAX_LIST_FILES_ENTRIES + 1)
        self.assertEqual(result[0], "f0.txt")
        self.assertIn("25 more files not shown", result[-1])

    def test_original_list_not_mutated(self):
        files = [f"f{i}" for i in range(tools_module.MAX_LIST_FILES_ENTRIES + 5)]
        before = list(files)
        cap_tool_result("list_files", files)
        self.assertEqual(files, before)


class SearchFilesCapTests(unittest.TestCase):

    def match(self, i, text="x"):
        return {"file": f"a{i}.py", "line": i, "text": text}

    def test_under_cap_unchanged(self):
        matches = [self.match(i) for i in range(5)]
        self.assertEqual(cap_tool_result("search_files", matches), matches)

    def test_over_cap_truncated_with_note(self):
        n = tools_module.MAX_SEARCH_MATCHES + 7
        result = cap_tool_result(
            "search_files", [self.match(i) for i in range(n)]
        )

        self.assertEqual(len(result), tools_module.MAX_SEARCH_MATCHES + 1)
        self.assertIn("7 more matches not shown", result[-1]["note"])
        self.assertNotIn("file", result[-1])  # M7.9 skips it

    def test_long_lines_shortened(self):
        long_text = "y" * 5000
        result = cap_tool_result("search_files", [self.match(1, long_text)])

        self.assertLess(len(result[0]["text"]), 400)
        self.assertEqual(result[0]["file"], "a1.py")

    def test_original_matches_not_mutated(self):
        original = self.match(1, "y" * 5000)
        cap_tool_result("search_files", [original])
        self.assertEqual(len(original["text"]), 5000)


class ReadFileCapTests(unittest.TestCase):

    def test_under_cap_unchanged(self):
        self.assertEqual(cap_tool_result("read_file", "hello\n"), "hello\n")

    def test_exactly_at_cap_unchanged(self):
        text = "x" * tools_module.MAX_READ_FILE_CHARS
        self.assertEqual(cap_tool_result("read_file", text), text)

    def test_over_cap_keeps_start_and_warns_about_rewrite(self):
        text = "A" * tools_module.MAX_READ_FILE_CHARS + "B" * 123

        result = cap_tool_result("read_file", text)

        self.assertTrue(result.startswith("A" * 100))
        self.assertNotIn("B", result.split("[...")[0])
        self.assertIn("123 more characters", result)
        self.assertIn("do NOT rewrite", result)


class GitStatusCapTests(unittest.TestCase):

    def test_under_cap_unchanged(self):
        self.assertEqual(cap_tool_result("git_status", " M a.py"), " M a.py")

    def test_over_cap_truncated(self):
        n = tools_module.MAX_GIT_STATUS_LINES + 40
        text = "\n".join(f"?? f{i}" for i in range(n))

        result = cap_tool_result("git_status", text)

        self.assertEqual(
            len(result.splitlines()), tools_module.MAX_GIT_STATUS_LINES + 1
        )
        self.assertIn("40 more entries not shown", result)


class PassThroughTests(unittest.TestCase):

    def test_errors_and_other_tools_unchanged(self):
        error = {"error": "boom"}
        self.assertIs(cap_tool_result("read_file", error), error)
        self.assertEqual(cap_tool_result("run_command", "x" * 99999),
                         "x" * 99999)
        self.assertEqual(cap_tool_result("write_file", "ok"), "ok")


class ExecuteToolIntegrationTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

        patcher = patch.object(
            tools_module, "get_project_root", return_value=self.root
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_model_sees_capped_but_tool_returns_everything(self):
        big = "line\n" * (tools_module.MAX_READ_FILE_CHARS)
        (self.root / "big.txt").write_text(big)

        direct = tools_module.read_file("big.txt")
        shown = seacode_module.execute_tool(
            "read_file", {"path": "big.txt"}
        )

        self.assertEqual(direct, big)
        self.assertLess(len(shown), len(big))
        self.assertIn("more characters not shown", shown)

    def test_many_files_capped_for_model_only(self):
        for i in range(tools_module.MAX_LIST_FILES_ENTRIES + 10):
            (self.root / f"f{i}.txt").write_text("")

        direct = tools_module.list_files()
        shown = seacode_module.execute_tool("list_files", {})

        self.assertEqual(len(direct), tools_module.MAX_LIST_FILES_ENTRIES + 10)
        self.assertEqual(len(shown), tools_module.MAX_LIST_FILES_ENTRIES + 1)


class GitStatusTimeoutTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            tools_module, "get_project_root",
            return_value=Path(self._tmp.name),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_timeout_raises_clear_error(self):
        with patch.object(
            tools_module.subprocess, "run",
            side_effect=subprocess.TimeoutExpired("git", 30),
        ):
            with self.assertRaises(TimeoutError) as ctx:
                tools_module.git_status()

        self.assertIn("timed out", str(ctx.exception))

    def test_passes_timeout_and_closed_stdin(self):
        completed = subprocess.CompletedProcess([], 0, stdout=" M a\n", stderr="")

        with patch.object(
            tools_module.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(tools_module.git_status(), "M a")

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["timeout"], tools_module.COMMAND_TIMEOUT_SECONDS)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
