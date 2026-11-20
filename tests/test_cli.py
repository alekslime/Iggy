"""
Tests for the `seacode` command-line interface (seacode.cli).

Covers argument parsing for every command and the `undo`,
`sessions`, and `init` commands' behavior against a temp project.
The REPL and TUI runners are not executed here (both are interactive
loops); parsing resolves them to the right handler.

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

import seacode.agent as seacode_module
import seacode.cli as cli_module
import seacode.tools as tools_module
from seacode import permissions as permissions_module
from seacode.sessions import save_session


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

    def test_sessions_subcommand(self):
        args = self.parse(["sessions"])
        self.assertEqual(args.command, "sessions")
        self.assertIs(args.func, cli_module.run_sessions)

    def test_init_subcommand(self):
        args = self.parse(["init", "--force"])
        self.assertEqual(args.command, "init")
        self.assertIs(args.func, cli_module.run_init)
        self.assertTrue(args.force)

    def test_trust_and_yes_flags(self):
        args = self.parse(["repl", "--trust", "edits"])
        self.assertEqual(args.trust, "edits")
        self.assertFalse(args.yes)

        args = self.parse(["tui", "--yes"])
        self.assertTrue(args.yes)
        self.assertIsNone(args.trust)

    def test_resume_flag(self):
        args = self.parse(["repl", "--resume"])
        self.assertEqual(args.resume, "latest")

        args = self.parse(["repl", "--resume", "20261119"])
        self.assertEqual(args.resume, "20261119")

        args = self.parse(["repl"])
        self.assertIsNone(args.resume)

    def test_no_stream_flag(self):
        self.assertFalse(self.parse(["repl"]).no_stream)
        self.assertTrue(self.parse(["repl", "--no-stream"]).no_stream)

    def test_verbose_flag(self):
        self.assertFalse(self.parse(["repl"]).verbose)
        self.assertTrue(self.parse(["repl", "--verbose"]).verbose)

    def test_no_anim_flag(self):
        self.assertFalse(self.parse(["tui"]).no_anim)
        self.assertTrue(self.parse(["tui", "--no-anim"]).no_anim)

    def test_bare_invocation_accepts_new_flags(self):
        args = self.parse(["--verbose", "--resume", "--yes"])
        self.assertTrue(args.verbose)
        self.assertEqual(args.resume, "latest")
        self.assertTrue(args.yes)


class ApplyCommonOptionsTests(unittest.TestCase):

    def full_args(self, **overrides):
        values = {
            "model": None,
            "host": None,
            "max_tools": None,
            "trust": None,
            "yes": False,
            "dir": None,
            "resume": None,
            "no_stream": False,
            "verbose": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def setUp(self):
        self.addCleanup(permissions_module.reset_trust_level)
        self.addCleanup(tools_module.set_extra_commands, [])

    def test_resolves_model_and_max_tools(self):
        args = self.full_args(model="cli-model", max_tools=4)

        with patch.dict(os.environ):
            os.environ.pop("SEACODE_DIR", None)
            (
                model,
                max_tools,
                config,
                stream,
                verbose,
            ) = cli_module.apply_common_options(args)

        self.assertEqual(model, "cli-model")
        self.assertEqual(max_tools, 4)
        self.assertEqual(config, {})
        self.assertTrue(stream)
        self.assertFalse(verbose)
        self.assertNotIn("SEACODE_DIR", os.environ)

    def test_no_stream_disables_streaming(self):
        args = self.full_args(no_stream=True)

        with patch.dict(os.environ):
            result = cli_module.apply_common_options(args)

        self.assertFalse(result[3])

    def test_dir_is_exported_as_seacode_dir(self):
        args = self.full_args(dir="/tmp/proj")

        with patch.dict(os.environ):
            cli_module.apply_common_options(args)
            self.assertEqual(os.environ["SEACODE_DIR"], "/tmp/proj")

    def test_config_file_is_loaded_and_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "seacode.toml").write_text(
                "[seacode]\n"
                'model = "config-model"\n'
                'trust = "edits"\n'
                'extra_commands = ["make"]\n',
                encoding="utf-8",
            )
            args = self.full_args(dir=tmpdir)

            with patch.dict(os.environ):
                os.environ.pop("SEACODE_MODEL", None)
                os.environ.pop("SEACODE_TRUST", None)
                (
                    model,
                    _,
                    config,
                    _,
                    _,
                ) = cli_module.apply_common_options(args)

        self.assertEqual(model, "config-model")
        self.assertEqual(config["trust"], "edits")
        self.assertEqual(
            permissions_module.get_trust_level(), "edits"
        )
        self.assertTrue(tools_module.is_command_allowed("make"))

    def test_trust_flag_sets_policy(self):
        args = self.full_args(trust="all")

        with patch.dict(os.environ):
            cli_module.apply_common_options(args)

        self.assertEqual(
            permissions_module.get_trust_level(), "all"
        )

    def test_yes_flag_sets_trust_all(self):
        args = self.full_args(yes=True)

        with patch.dict(os.environ):
            cli_module.apply_common_options(args)

        self.assertEqual(
            permissions_module.get_trust_level(), "all"
        )

    def test_invalid_config_trust_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "seacode.toml").write_text(
                '[seacode]\ntrust = "bogus"\n', encoding="utf-8"
            )
            args = self.full_args(dir=tmpdir)

            with (
                patch.dict(os.environ),
                self.assertRaises(ValueError),
            ):
                cli_module.apply_common_options(args)

    def test_host_rebuilds_client(self):
        args = SimpleNamespace(
            model=None,
            host="http://example:11434",
            max_tools=None,
            dir=None,
        )

        original_client = seacode_module.client
        try:
            with patch("ollama.Client") as mock_client:
                cli_module.apply_common_options(args)
                mock_client.assert_called_once_with(
                    host="http://example:11434"
                )
                self.assertIs(
                    seacode_module.client, mock_client.return_value
                )
        finally:
            seacode_module.client = original_client


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
        the --dir one, which needs the real SEACODE_DIR lookup)."""

        root_patcher = patch.object(
            cli_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

        # run_undo reaches the project both directly and through the
        # tools' own module-global lookup.
        tools_patcher = patch.object(
            tools_module, "get_project_root", return_value=self.project_root
        )
        tools_patcher.start()
        self.addCleanup(tools_patcher.stop)

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

        with patch.dict(os.environ, {"SEACODE_DIR": str(other)}):
            tools_module.write_file("n.txt", "v2\n")

        with patch.dict(os.environ):
            exit_code, output = self.run_undo(
                list_backups=True, dir=str(other)
            )

        self.assertEqual(exit_code, 0)
        self.assertIn(str(other), output)
        self.assertIn("n.txt", output)


class RunSessionsTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        root_patcher = patch.object(
            cli_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

    def test_no_sessions(self):
        args = SimpleNamespace(dir=None)
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = cli_module.run_sessions(args)

        self.assertEqual(exit_code, 0)
        self.assertIn("No saved sessions", output.getvalue())

    def test_lists_sessions(self):
        session_id = save_session(
            self.project_root,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Fix the bug"},
            ],
            "test-model",
        )

        args = SimpleNamespace(dir=None)
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = cli_module.run_sessions(args)

        self.assertEqual(exit_code, 0)
        self.assertIn(session_id, output.getvalue())
        self.assertIn("Fix the bug", output.getvalue())


class RunInitTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        root_patcher = patch.object(
            cli_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

    def test_writes_starter_config(self):
        args = SimpleNamespace(dir=None, force=False)
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = cli_module.run_init(args)

        self.assertEqual(exit_code, 0)
        self.assertTrue(
            (self.project_root / "seacode.toml").is_file()
        )

    def test_refuses_overwrite_without_force(self):
        (self.project_root / "seacode.toml").write_text(
            '[seacode]\nmodel = "mine"\n', encoding="utf-8"
        )
        args = SimpleNamespace(dir=None, force=False)
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = cli_module.run_init(args)

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            (self.project_root / "seacode.toml").read_text(
                encoding="utf-8"
            ),
            '[seacode]\nmodel = "mine"\n',
        )

    def test_force_overwrites(self):
        (self.project_root / "seacode.toml").write_text(
            '[seacode]\nmodel = "mine"\n', encoding="utf-8"
        )
        args = SimpleNamespace(dir=None, force=True)
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = cli_module.run_init(args)

        self.assertEqual(exit_code, 0)
        self.assertNotIn(
            '"mine"',
            (self.project_root / "seacode.toml").read_text(
                encoding="utf-8"
            ),
        )


class LoadResumeSessionTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        root_patcher = patch.object(
            cli_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

    def test_no_flag_returns_none(self):
        self.assertIsNone(
            cli_module.load_resume_session(SimpleNamespace(resume=None))
        )

    def test_loads_latest(self):
        first = save_session(
            self.project_root,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
            ],
            "m",
        )
        save_session(
            self.project_root,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "second"},
            ],
            "m",
            session_id=first + "-9",
        )

        output = io.StringIO()

        with redirect_stdout(output):
            session = cli_module.load_resume_session(
                SimpleNamespace(resume="latest")
            )

        # Same-second ids sort after the plain timestamp id.
        self.assertEqual(session["id"], first + "-9")

    def test_missing_session_exits(self):
        output = io.StringIO()

        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                cli_module.load_resume_session(
                    SimpleNamespace(resume="latest")
                )

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Cannot resume", output.getvalue())


if __name__ == "__main__":
    unittest.main()
