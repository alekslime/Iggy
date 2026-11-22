"""
Tests for conversation persistence (seacode.sessions).

Sessions are plain JSON under a temp project root -- no agent loop,
no Ollama involved.

Scenarios covered:
  1. save_session starts a new timestamped session and re-saving
     with the id keeps `created` while bumping `updated`.
  2. generate_session_id suffixes on same-second collisions.
  3. load_session resolves latest, exact ids, and unique prefixes;
     reports missing/ambiguous/corrupt sessions.
  4. list_sessions orders newest-first with previews and turn
     counts, skipping corrupt files.
  5. No temp files are left behind after a save.

Run with:  python3 -m pytest tests/test_sessions.py -v
       or: python3 -m unittest tests.test_sessions -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import seacode.sessions as sessions_module


def sample_messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Fix the bug in app.py"},
        {"role": "assistant", "content": '{"tool": "none"}'},
    ]


class SaveSessionTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

    def test_save_creates_session_file(self):
        session_id = sessions_module.save_session(
            self.project_root, sample_messages(), "model-x"
        )

        session_file = (
            self.project_root / ".seacode_sessions" / f"{session_id}.json"
        )
        self.assertTrue(session_file.is_file())

        payload = json.loads(session_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "model-x")
        self.assertEqual(payload["messages"], sample_messages())
        self.assertIn("created", payload)
        self.assertIn("updated", payload)

    def test_resave_keeps_created(self):
        session_id = sessions_module.save_session(
            self.project_root, sample_messages(), "model-x"
        )
        session_file = (
            self.project_root / ".seacode_sessions" / f"{session_id}.json"
        )
        created = json.loads(
            session_file.read_text(encoding="utf-8")
        )["created"]

        again = sessions_module.save_session(
            self.project_root,
            sample_messages() + [{"role": "user", "content": "more"}],
            "model-x",
            session_id=session_id,
        )

        self.assertEqual(again, session_id)
        payload = json.loads(session_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["created"], created)
        self.assertEqual(len(payload["messages"]), 4)

    def test_no_temp_files_left_behind(self):
        sessions_module.save_session(
            self.project_root, sample_messages(), "model-x"
        )

        leftovers = list(
            (self.project_root / ".seacode_sessions").glob(".tmp-*")
        )
        self.assertEqual(leftovers, [])

    def test_generate_session_id_suffixes_on_collision(self):
        first = sessions_module.generate_session_id(set())
        second = sessions_module.generate_session_id({first})

        self.assertNotEqual(first, second)
        self.assertTrue(second.startswith(first))

        third = sessions_module.generate_session_id({first, second})
        self.assertNotIn(third, {first, second})


class LoadSessionTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

    def test_no_sessions_raises(self):
        with self.assertRaises(FileNotFoundError):
            sessions_module.load_session(self.project_root)

    def test_loads_latest_by_default(self):
        first = sessions_module.save_session(
            self.project_root, sample_messages(), "m"
        )
        second = sessions_module.save_session(
            self.project_root, sample_messages(), "m",
            session_id=first + "-2",
        )

        for selector in (None, "latest"):
            session = sessions_module.load_session(
                self.project_root, selector
            )
            # Same-second ids sort after the plain timestamp id (or
            # the second save landed in a later second -- either way
            # the explicitly later session wins).
            self.assertEqual(session["id"], second)
            self.assertEqual(session["messages"], sample_messages())

    def test_loads_exact_and_prefix(self):
        session_id = sessions_module.save_session(
            self.project_root, sample_messages(), "model-x"
        )

        exact = sessions_module.load_session(
            self.project_root, session_id
        )
        self.assertEqual(exact["id"], session_id)
        self.assertEqual(exact["model"], "model-x")

        prefixed = sessions_module.load_session(
            self.project_root, session_id[:8]
        )
        self.assertEqual(prefixed["id"], session_id)

    def test_unknown_prefix_raises(self):
        sessions_module.save_session(
            self.project_root, sample_messages(), "m"
        )

        with self.assertRaises(FileNotFoundError):
            sessions_module.load_session(self.project_root, "99999999")

    def test_ambiguous_prefix_raises(self):
        first = sessions_module.save_session(
            self.project_root, sample_messages(), "m"
        )
        sessions_module.save_session(
            self.project_root, sample_messages(), "m",
            session_id=first + "-2",
        )

        with self.assertRaises(ValueError) as raised:
            sessions_module.load_session(
                self.project_root, first[:8]
            )

        self.assertIn("ambiguous", str(raised.exception))

    def test_corrupt_file_raises(self):
        session_id = sessions_module.save_session(
            self.project_root, sample_messages(), "m"
        )
        (self.project_root / ".seacode_sessions" / f"{session_id}.json"
         ).write_text("not json", encoding="utf-8")

        with self.assertRaises(ValueError):
            sessions_module.load_session(self.project_root, session_id)

    def test_bad_messages_shape_raises(self):
        session_id = sessions_module.save_session(
            self.project_root, sample_messages(), "m"
        )
        session_file = (
            self.project_root / ".seacode_sessions" / f"{session_id}.json"
        )
        payload = json.loads(session_file.read_text(encoding="utf-8"))
        payload["messages"] = [{"role": "user"}]
        session_file.write_text(
            json.dumps(payload), encoding="utf-8"
        )

        with self.assertRaises(ValueError):
            sessions_module.load_session(self.project_root, session_id)


class ListSessionsTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

    def test_missing_dir_lists_empty(self):
        self.assertEqual(
            sessions_module.list_sessions(self.project_root), []
        )

    def test_entries_newest_first_with_preview(self):
        first = sessions_module.save_session(
            self.project_root, sample_messages(), "model-a",
            turns=1,
        )
        sessions_module.save_session(
            self.project_root,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Second question here"},
                {"role": "user", "content": "follow-up"},
            ],
            "model-b",
            session_id=first + "-2",
            turns=2,
        )

        entries = sessions_module.list_sessions(self.project_root)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["id"], first + "-2")
        self.assertEqual(entries[0]["turns"], 2)
        self.assertEqual(
            entries[0]["preview"], "Second question here"
        )
        self.assertEqual(entries[1]["id"], first)
        self.assertEqual(entries[1]["turns"], 1)

    def test_turns_fall_back_to_user_count(self):
        sessions_module.save_session(
            self.project_root, sample_messages(), "m"
        )

        entries = sessions_module.list_sessions(self.project_root)

        self.assertEqual(entries[0]["turns"], 1)

    def test_load_returns_turns(self):
        session_id = sessions_module.save_session(
            self.project_root, sample_messages(), "m", turns=4
        )

        session = sessions_module.load_session(
            self.project_root, session_id
        )

        self.assertEqual(session["turns"], 4)

    def test_long_preview_truncates(self):
        sessions_module.save_session(
            self.project_root,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x" * 100},
            ],
            "m",
        )

        entries = sessions_module.list_sessions(self.project_root)

        self.assertEqual(len(entries[0]["preview"]), 60)
        self.assertTrue(entries[0]["preview"].endswith("..."))

    def test_corrupt_files_are_skipped(self):
        good = sessions_module.save_session(
            self.project_root, sample_messages(), "m"
        )
        (self.project_root / ".seacode_sessions" / "bogus.json"
         ).write_text("not json", encoding="utf-8")

        entries = sessions_module.list_sessions(self.project_root)

        self.assertEqual([entry["id"] for entry in entries], [good])


class RelativeTimeTests(unittest.TestCase):

    NOW = sessions_module.datetime(2026, 11, 22, 12, 0, 0,
                                     tzinfo=sessions_module.timezone.utc)

    def rel(self, iso):
        return sessions_module.relative_time(iso, now=self.NOW)

    def test_ages(self):
        self.assertEqual(self.rel("2026-11-22T11:59:30+00:00"), "just now")
        self.assertEqual(self.rel("2026-11-22T11:45:00+00:00"), "15m ago")
        self.assertEqual(self.rel("2026-11-22T09:00:00+00:00"), "3h ago")
        self.assertEqual(self.rel("2026-11-20T12:00:00+00:00"), "2d ago")
        self.assertEqual(self.rel("2026-11-01T12:00:00+00:00"), "2026-11-01")

    def test_bad_input(self):
        self.assertEqual(sessions_module.relative_time(""), "")
        self.assertEqual(
            sessions_module.relative_time("garbage"), "garbage"[:10]
        )


if __name__ == "__main__":
    unittest.main()
