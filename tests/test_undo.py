"""
Tests for backup listing and undo (tools.list_backups,
tools.undo_last_change, and the `seacode undo` command's backend).

Same philosophy as the M7.x suites: no real project directory
(`seacode.tools.get_project_root` is patched to a temp dir) and
permission prompts are scripted.

Scenarios covered:
  1. list_backups returns entries oldest-first, for the whole
     project or filtered to one path.
  2. undo_last_change restores the previous content and backs up the
     current content first, so an undo can itself be undone.
  3. Undo recreates a file that was deleted after its last backup.
  4. Undo with no backups (project-wide or for one path) reports
     that instead of failing.
  5. A denied undo leaves the file untouched.
  6. Undo is registered as an agent tool the model can call.

Run with:  python3 -m pytest tests/test_undo.py -v
       or: python3 -m unittest tests.test_undo -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import seacode.tools as tools_module


class UndoTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

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

    def make_file(self, relative_path, content):
        full_path = self.project_root / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def read_project_file(self, relative_path):
        return (self.project_root / relative_path).read_text(
            encoding="utf-8"
        )

    def write_versions(self, relative_path, *versions):
        for version in versions:
            tools_module.write_file(relative_path, version)

    def test_list_backups_oldest_first(self):
        self.write_versions("notes.txt", "v1\n", "v2\n", "v3\n")

        entries = tools_module.list_backups("notes.txt")

        # First write creates the file; the next two back up v1, v2.
        self.assertEqual(len(entries), 2)
        self.assertLess(
            entries[0]["timestamp"], entries[1]["timestamp"]
        )
        for entry in entries:
            self.assertEqual(entry["path"], "notes.txt")
            self.assertTrue(
                (self.project_root / entry["backup"]).is_file()
            )

    def test_list_backups_all_and_filtered(self):
        self.write_versions("a.txt", "a1\n", "a2\n")
        self.write_versions("sub/b.txt", "b1\n", "b2\n")

        all_entries = tools_module.list_backups()
        self.assertEqual(len(all_entries), 2)

        filtered = tools_module.list_backups("sub/b.txt")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["path"], "sub/b.txt")

    def test_list_backups_empty_project(self):
        self.assertEqual(tools_module.list_backups(), [])

    def test_list_backups_rejects_outside_path(self):
        with self.assertRaises(PermissionError):
            tools_module.list_backups("../outside.txt")

    def test_undo_restores_previous_content(self):
        self.write_versions("notes.txt", "v1\n", "v2\n", "v3\n")

        result = tools_module.undo_last_change("notes.txt")

        self.assertIn("Restored notes.txt", result)
        self.assertEqual(
            self.read_project_file("notes.txt"), "v2\n"
        )

    def test_undo_backs_up_current_content_first(self):
        self.write_versions("notes.txt", "v1\n", "v2\n")

        tools_module.undo_last_change("notes.txt")

        # v1 was backed up by the second write; the undo itself backed
        # up v2 before restoring v1 -- so v2 is recoverable too.
        entries = tools_module.list_backups("notes.txt")
        self.assertEqual(len(entries), 2)

        tools_module.undo_last_change("notes.txt")
        self.assertEqual(
            self.read_project_file("notes.txt"), "v2\n"
        )

    def test_undo_without_path_picks_latest_backup(self):
        self.write_versions("old.txt", "o1\n", "o2\n")
        self.write_versions("new.txt", "n1\n", "n2\n")

        result = tools_module.undo_last_change()

        self.assertIn("Restored new.txt", result)
        self.assertEqual(self.read_project_file("new.txt"), "n1\n")
        self.assertEqual(self.read_project_file("old.txt"), "o2\n")

    def test_undo_recreates_deleted_file(self):
        self.write_versions("gone.txt", "v1\n", "v2\n")
        tools_module.delete_file("gone.txt")
        self.assertFalse((self.project_root / "gone.txt").exists())

        result = tools_module.undo_last_change("gone.txt")

        self.assertIn("Restored gone.txt", result)
        self.assertEqual(
            self.read_project_file("gone.txt"), "v2\n"
        )

    def test_undo_with_no_backups(self):
        self.make_file("plain.txt", "untouched\n")

        self.assertIn(
            "no backups", tools_module.undo_last_change()
        )
        self.assertIn(
            "no backups", tools_module.undo_last_change("plain.txt")
        )
        # Nothing changed.
        self.assertEqual(
            self.read_project_file("plain.txt"), "untouched\n"
        )

    def test_denied_undo_leaves_file_untouched(self):
        self.write_versions("notes.txt", "v1\n", "v2\n")

        with patch.object(
            tools_module, "request_permission", return_value=False
        ):
            result = tools_module.undo_last_change("notes.txt")

        self.assertEqual(result, "File modification denied by user.")
        self.assertEqual(
            self.read_project_file("notes.txt"), "v2\n"
        )
        self.assertEqual(len(tools_module.list_backups()), 1)

    def test_undo_is_registered_as_agent_tool(self):
        self.assertIs(
            tools_module.TOOLS["undo_last_change"],
            tools_module.undo_last_change,
        )


if __name__ == "__main__":
    unittest.main()
