"""
Tests that SeaCode's own working directories stay invisible to the file
tools: backups and sessions are never listed, read via search, or
stats-counted as project files.

Run with:  python3 -m pytest tests/test_project_files.py -v
       or: python3 -m unittest tests.test_project_files -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import seacode.tools as tools_module


class IgnoredDirectoriesTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        root_patcher = patch.object(
            tools_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

    def test_list_files_skips_seacode_dirs(self):
        (self.project_root / "real.py").write_text("x = 1\n")
        backups = self.project_root / ".seacode_backups" / "real.py"
        backups.mkdir(parents=True)
        (backups / "20261120-000000-000000.bak").write_text("old\n")
        sessions = self.project_root / ".seacode_sessions"
        sessions.mkdir()
        (sessions / "20261120-000000.json").write_text("{}\n")

        self.assertEqual(
            tools_module.list_files("."), ["real.py"]
        )

    def test_search_files_skips_seacode_dirs(self):
        (self.project_root / "real.py").write_text("needle here\n")
        sessions = self.project_root / ".seacode_sessions"
        sessions.mkdir()
        (sessions / "s.json").write_text("needle in session\n")

        matches = tools_module.search_files("needle")

        self.assertEqual(
            [(m["file"], m["line"]) for m in matches],
            [("real.py", 1)],
        )


if __name__ == "__main__":
    unittest.main()
