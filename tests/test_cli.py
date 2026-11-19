"""
Tests for the `iggy` command-line interface (agent.cli).

Covers argument parsing for every command and the `undo` command's
behavior against a temp project. The REPL and TUI runners are not
executed here (both are interactive loops); parsing resolves them to
the right handler.

Run with:  python3 -m pytest tests/test_cli.py -v
       or: python3 -m unittest tests.test_cli -v
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent.agent as agent_module
import agent.cli as cli_module
import agent.tools as tools_module


class ParserTests(unittest.TestCase):

    def parse(self, argv):
        return cli_module.build_parser().parse_args(argv)

    def test_bare_invocation_is_repl(self):
        args = self.parse([])
        self.assertIsNone(args.command)
        self.assertIs(args.func, cli_module.run_repl)
        self.assertIsNone(args.model)

    def test_bare_invocation_accepts_shared_flags(self):
        args = self.parse(
            ["--model", "m", "--max-tools", "5", "--dir", "/tmp/x"]
        )
        self.assertIs(args.func, cli_module.run_repl)
        self.assertEqual(args.model, "m")
        self.assertEqual(args.max_tools, 5)
        self.assertEqual(args.dir, "/tmp/x")

    def test_repl_subcommand(self):
        args = self.parse(["repl", "--model", "m"])
        self.assertEqual(args.command, "repl")
        self.assertIs(args.func, cli_module.run_repl)
        self.assertEqual(args.model, "m")

    def test_tui_subcommand(self):
        args = self.parse(["tui", "--host", "http://x:11434"])
        self.assertEqual(args.command, "tui")
        self.assertIs(args.func, cli_module.run_tui)
        self.assertEqual(args.host, "http://x:11434")

    def test_undo_subcommand(self):
        args = self.parse(["undo", "notes.txt"])
        self.assertEqual(args.command, "undo")
        self.assertIs(args.func, cli_module.run_undo)
        self.assertEqual(args.path, "notes.txt")
        self.assertFalse(args.list)

    def test_undo_list_flag(self):
        args = self.parse(["undo", "--list"])
        self.assertTrue(args.list)
        self.assertIsNone(args.path)


class ApplyCommonOptionsTests(unittest.TestCase):

    def test_resolves_model_and_max_tools(self):
        args = SimpleNamespace(
            model="cli-model", host=None, max_tools=4, dir=None
        )

        with patch.dict(os.environ):
            os.environ.pop("IGGY_DIR", None)
            model, max_tools = cli_module.apply_common_options(args)

        self.assertEqual(model, "cli-model")
        self.assertEqual(max_tools, 4)
        self.assertNotIn("IGGY_DIR", os.environ)

    def test_dir_is_exported_as_iggy_dir(self):
        args = SimpleNamespace(
            model=None, host=None, max_tools=None, dir="/tmp/proj"
        )

        with patch.dict(os.environ):
            cli_module.apply_common_options(args)
            self.assertEqual(os.environ["IGGY_DIR"], "/tmp/proj")

    def test_host_rebuilds_client(self):
        args = SimpleNamespace(
            model=None,
            host="http://example:11434",
            max_tools=None,
            dir=None,
        )

        original_client = agent_module.client
        try:
            with patch("ollama.Client") as mock_client:
                cli_module.apply_common_options(args)
                mock_client.assert_called_once_with(
                    host="http://example:11434"
                )
                self.assertIs(
                    agent_module.client, mock_client.return_value
                )
        finally:
            agent_module.client = original_client


class RunUndoTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        permission_patcher = patch.object(
            tools_module, "request_permission", return_value=True
        )
        permission_patcher.start()
        self.addCleanup(permission_patcher.stop)

    def use_temp_project(self):
        """Scope the tools to this test's temp dir (all tests except
        the --dir one, which needs the real IGGY_DIR lookup)."""

        root_patcher = patch.object(
            tools_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

    def run_undo(self, path=None, list_backups=False, dir=None):
        args = SimpleNamespace(path=path, list=list_backups, dir=dir)
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli_module.run_undo(args)
        return exit_code, output.getvalue()

    def test_list_with_no_backups(self):
        self.use_temp_project()
        exit_code, output = self.run_undo(list_backups=True)

        self.assertEqual(exit_code, 0)
        self.assertIn("No backups found", output)

    def test_list_shows_backups(self):
        self.use_temp_project()
        (self.project_root / "n.txt").write_text("v1\n")
        tools_module.write_file("n.txt", "v2\n")

        exit_code, output = self.run_undo(list_backups=True)

        self.assertEqual(exit_code, 0)
        self.assertIn("n.txt", output)

    def test_restore(self):
        self.use_temp_project()
        (self.project_root / "n.txt").write_text("v1\n")
        tools_module.write_file("n.txt", "v2\n")

        exit_code, output = self.run_undo(path="n.txt")

        self.assertEqual(exit_code, 0)
        self.assertIn("Restored n.txt", output)
        self.assertEqual(
            (self.project_root / "n.txt").read_text(), "v1\n"
        )

    def test_nothing_to_undo(self):
        self.use_temp_project()
        exit_code, output = self.run_undo()

        self.assertEqual(exit_code, 0)
        self.assertIn("Nothing to undo", output)

    def test_dir_flag_scopes_to_that_project(self):
        other = self.project_root / "other"
        other.mkdir()
        (other / "n.txt").write_text("v1\n")

        with patch.dict(os.environ, {"IGGY_DIR": str(other)}):
            tools_module.write_file("n.txt", "v2\n")

        with patch.dict(os.environ):
            exit_code, output = self.run_undo(
                list_backups=True, dir=str(other)
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(str(other), output)
        self.assertIn("n.txt", output)


if __name__ == "__main__":
    unittest.main()
