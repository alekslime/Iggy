"""Tests for real-compiler diagnostics after Python edits."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import seacode.tools as tools_module
import seacode.verify as verify_module


class DiagnosticsTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)
        patcher = patch.object(
            tools_module, "get_project_root",
            return_value=self.project_root,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # verify imported get_project_root by name; patch its binding.
        verify_patcher = patch.object(
            verify_module, "get_project_root",
            return_value=self.project_root,
        )
        verify_patcher.start()
        self.addCleanup(verify_patcher.stop)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_valid_file_passes(self):
        (self.project_root / "ok.py").write_text("x = 1\n")
        self.assertIsNone(
            verify_module.verify_python_diagnostics("ok.py")
        )

    def test_broken_file_reports(self):
        (self.project_root / "bad.py").write_text("def broken(:\n")
        message = verify_module.verify_python_diagnostics("bad.py")
        self.assertIsNotNone(message)
        self.assertIn("does not compile", message)

    def test_non_python_skipped(self):
        self.assertIsNone(
            verify_module.verify_python_diagnostics("notes.txt")
        )

    def test_missing_file_never_blocks(self):
        # py_compile fails on a missing file, which would report --
        # but the agent only calls this after a successful edit, so
        # the file exists. Direct callers still get a message rather
        # than an exception; assert it doesn't raise.
        try:
            verify_module.verify_python_diagnostics("ghost.py")
        except Exception as e:  # pragma: no cover
            self.fail(f"diagnostics raised: {e}")


if __name__ == "__main__":
    unittest.main()
