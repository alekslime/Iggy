"""
Tests for project configuration (seacode.config) and the
config-extended command allowlist (tools.set_extra_commands).

Run with:  python3 -m pytest tests/test_project_config.py -v
       or: python3 -m unittest tests.test_project_config -v
"""

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import seacode.config as config_module
import seacode.tools as tools_module


class LoadProjectConfigTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

    def test_missing_file_is_empty_config(self):
        self.assertEqual(
            config_module.load_project_config(self.project_root), {}
        )

    def test_valid_keys(self):
        (self.project_root / "seacode.toml").write_text(
            "[seacode]\n"
            'model = "m"\n'
            'host = "http://x:11434"\n'
            "max_tools = 25\n"
            'trust = "edits"\n'
            'extra_commands = ["make", "npm"]\n',
            encoding="utf-8",
        )

        self.assertEqual(
            config_module.load_project_config(self.project_root),
            {
                "model": "m",
                "host": "http://x:11434",
                "max_tools": 25,
                "trust": "edits",
                "extra_commands": ["make", "npm"],
            },
        )

    def test_unknown_keys_ignored(self):
        (self.project_root / "seacode.toml").write_text(
            "[seacode]\nmodel = \"m\"\nfuture_key = true\n",
            encoding="utf-8",
        )

        self.assertEqual(
            config_module.load_project_config(self.project_root),
            {"model": "m"},
        )

    def test_missing_table_is_empty_config(self):
        (self.project_root / "seacode.toml").write_text(
            "[other]\nmodel = \"m\"\n", encoding="utf-8"
        )

        self.assertEqual(
            config_module.load_project_config(self.project_root), {}
        )

    def test_invalid_toml_warns_and_is_empty(self):
        (self.project_root / "seacode.toml").write_text(
            "[seacode\nmodel = ", encoding="utf-8"
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            config = config_module.load_project_config(
                self.project_root
            )

        self.assertEqual(config, {})
        self.assertIn("Warning", stderr.getvalue())

    def test_non_table_warns_and_is_empty(self):
        (self.project_root / "seacode.toml").write_text(
            'seacode = "nope"\n', encoding="utf-8"
        )
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            config = config_module.load_project_config(
                self.project_root
            )

        self.assertEqual(config, {})
        self.assertIn("Warning", stderr.getvalue())


class WriteDefaultConfigTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

    def test_writes_parseable_starter(self):
        path = config_module.write_default_config(self.project_root)

        self.assertEqual(
            path, self.project_root / "seacode.toml"
        )
        # All starter keys are commented out, so it loads empty.
        self.assertEqual(
            config_module.load_project_config(self.project_root), {}
        )

    def test_refuses_existing_without_force(self):
        target = self.project_root / "seacode.toml"
        target.write_text('[seacode]\nmodel = "m"\n', encoding="utf-8")

        self.assertIsNone(
            config_module.write_default_config(self.project_root)
        )
        self.assertIn('"m"', target.read_text(encoding="utf-8"))

    def test_force_overwrites(self):
        target = self.project_root / "seacode.toml"
        target.write_text('[seacode]\nmodel = "m"\n', encoding="utf-8")

        path = config_module.write_default_config(
            self.project_root, force=True
        )

        self.assertEqual(path, target)
        self.assertNotIn('"m"', target.read_text(encoding="utf-8"))


class ExtraCommandsTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        root_patcher = patch.object(
            tools_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

        permission_patcher = patch.object(
            tools_module, "request_permission", return_value=True
        )
        permission_patcher.start()
        self.addCleanup(permission_patcher.stop)

        self.addCleanup(tools_module.set_extra_commands, [])

    def test_builtin_allowlist_unchanged(self):
        self.assertTrue(tools_module.is_command_allowed("git"))
        self.assertFalse(tools_module.is_command_allowed("make"))

    def test_extras_are_allowed(self):
        tools_module.set_extra_commands(["make", "npm"])

        self.assertTrue(tools_module.is_command_allowed("make"))
        self.assertTrue(tools_module.is_command_allowed("npm"))
        self.assertFalse(tools_module.is_command_allowed("cargo"))

    def test_none_resets(self):
        tools_module.set_extra_commands(["make"])
        tools_module.set_extra_commands(None)

        self.assertFalse(tools_module.is_command_allowed("make"))

    def test_invalid_entries_raise(self):
        for bad in ("", "  ", "rm -rf", "/bin/ls", 123, None):
            with self.subTest(entry=bad):
                with self.assertRaises(ValueError):
                    tools_module.set_extra_commands([bad])

    def test_run_command_honors_extras(self):
        tools_module.set_extra_commands(["true"])

        self.assertEqual(tools_module.run_command("true"), "")

        with self.assertRaises(PermissionError):
            tools_module.run_command("definitely-not-allowed")


if __name__ == "__main__":
    unittest.main()
