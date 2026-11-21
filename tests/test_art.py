"""
Tests for the ASCII mascot (seacode.art).

Locks the transcription: line count, signature details, and
formatting rules (no tabs, no trailing whitespace, sane width) so
an edit can't silently corrupt the picture.

Run with:  python3 -m pytest tests/test_art.py -v
       or: python3 -m unittest tests.test_art -v
"""

import unittest

from seacode.art import MASCOT, dither_rule, side_column


class MascotTests(unittest.TestCase):

    def test_line_count(self):
        self.assertEqual(len(MASCOT.splitlines()), 25)

    def test_signature_details(self):
        self.assertIn("(9)", MASCOT)
        self.assertIn(r"\:.|", MASCOT)
        self.assertIn(".-.", MASCOT)
        self.assertIn("`-..:::-'`", MASCOT)

    def test_clean_formatting(self):
        for line in MASCOT.splitlines():
            self.assertNotIn("\t", line)
            self.assertEqual(line, line.rstrip())
            self.assertLessEqual(len(line), 60)


class DitherRuleTests(unittest.TestCase):

    def test_exact_width(self):
        self.assertEqual(len(dither_rule(64)), 64)
        self.assertEqual(dither_rule(1), " ")

    def test_symmetric_fade(self):
        rule = dither_rule(64)

        self.assertEqual(rule, rule[::-1])
        # Dense block edges fading to a clear middle.
        self.assertEqual(rule[0], "▓")
        self.assertEqual(rule[-1], "▓")
        self.assertEqual(rule[28:36], " " * 8)

    def test_only_shade_glyphs(self):
        rule = dither_rule(64)

        for char in rule:
            self.assertIn(char, " ░▒▓")

    def test_phase_shifts_shimmer(self):
        first = dither_rule(64, 0)

        self.assertEqual(dither_rule(64, 0), first)
        self.assertNotEqual(dither_rule(64, 1), first)
        # Period 8: a full cycle returns to the same frame.
        self.assertEqual(dither_rule(64, 8), first)

    def test_phases_stay_symmetric(self):
        for phase in range(8):
            rule = dither_rule(64, phase)
            self.assertEqual(rule, rule[::-1])


class SideColumnTests(unittest.TestCase):

    def test_exact_dimensions(self):
        frame = side_column(30, 25, phase=0).splitlines()

        self.assertEqual(len(frame), 25)

        for row in frame:
            self.assertLessEqual(len(row), 30)

    def test_deterministic_per_phase(self):
        self.assertEqual(
            side_column(30, 25, phase=4),
            side_column(30, 25, phase=4),
        )
        self.assertNotEqual(
            side_column(30, 25, phase=4),
            side_column(30, 25, phase=5),
        )

    def test_seed_and_direction_vary_output(self):
        base = side_column(30, 25, phase=0, seed=0, direction=1)

        self.assertNotEqual(
            side_column(30, 25, phase=0, seed=3, direction=1), base
        )
        self.assertNotEqual(
            side_column(30, 25, phase=0, seed=0, direction=-1), base
        )

    def test_tiny_sizes_dont_crash(self):
        frame = side_column(1, 1, phase=0).splitlines()

        self.assertEqual(len(frame), 1)

    def test_fish_swim_across(self):
        frames = [
            side_column(30, 25, phase=phase) for phase in (0, 20, 40)
        ]

        self.assertEqual(len(set(frames)), 3)


if __name__ == "__main__":
    unittest.main()
