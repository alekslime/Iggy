"""
Tests for the ASCII mascot (seacode.art).

Locks the transcription: line count, signature details, and
formatting rules (no tabs, no trailing whitespace, sane width) so
an edit can't silently corrupt the picture.

Run with:  python3 -m pytest tests/test_art.py -v
       or: python3 -m unittest tests.test_art -v
"""

import unittest

from seacode.art import (
    CREATURE,
    MASCOT,
    MOSAIC_FADE_FRAMES,
    MOSAIC_GLYPHS,
    MOSAIC_MODE_FRAMES,
    MOSAIC_MODES,
    blend_hex,
    dither_rule,
    mosaic_frame,
    mosaic_grid,
    mosaic_palette,
    sea_runs,
    side_column,
    value_noise,
    wave_rule,
)


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


class WaveRuleTests(unittest.TestCase):

    def test_exact_width_and_glyphs(self):
        rule = wave_rule(64)
        self.assertEqual(len(rule), 64)
        for char in rule:
            self.assertIn(char, " ∼≈~")

    def test_edges_fade_to_space(self):
        rule = wave_rule(64)
        self.assertEqual(rule[0], " ")
        self.assertEqual(rule[-1], " ")
        self.assertIn("≈", rule)

    def test_phase_drifts(self):
        self.assertNotEqual(wave_rule(64, 0), wave_rule(64, 1))
        self.assertEqual(wave_rule(64, 0), wave_rule(64, 0))


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


class ValueNoiseTests(unittest.TestCase):

    def test_range_and_determinism(self):
        for i in range(200):
            v = value_noise(i * 0.37, i * 0.11, seed=3)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
            self.assertEqual(v, value_noise(i * 0.37, i * 0.11, seed=3))

    def test_smooth_between_neighbours(self):
        a = value_noise(5.00, 2.0)
        b = value_noise(5.01, 2.0)

        self.assertLess(abs(a - b), 0.05)


class MosaicGridTests(unittest.TestCase):

    def test_exact_dimensions_incl_odd_width(self):
        for width in (1, 2, 7, 30):
            grid = mosaic_grid(width, 9, phase=5)

            self.assertEqual(len(grid), 9)
            for row in grid:
                self.assertEqual(len(row), width)

    def test_levels_in_range(self):
        for phase in (0, 33, 79, 120):
            for row in mosaic_grid(30, 12, phase):
                for level in row:
                    self.assertIn(level, range(len(MOSAIC_GLYPHS)))

    def test_tiles_are_two_wide(self):
        for row in mosaic_grid(30, 12, phase=17):
            for x in range(0, 30, 2):
                self.assertEqual(row[x], row[x + 1])

    def test_deterministic_and_moving(self):
        self.assertEqual(
            mosaic_grid(30, 12, 9, 2), mosaic_grid(30, 12, 9, 2)
        )
        self.assertNotEqual(
            mosaic_grid(30, 12, 9, 2), mosaic_grid(30, 12, 10, 2)
        )
        self.assertNotEqual(
            mosaic_grid(30, 12, 9, 0), mosaic_grid(30, 12, 9, 1)
        )

    def test_every_pattern_moves_and_uses_range(self):
        for mode in range(MOSAIC_MODES):
            base = mode * MOSAIC_MODE_FRAMES + 10
            a = mosaic_grid(40, 16, base)
            b = mosaic_grid(40, 16, base + 3)

            self.assertNotEqual(a, b)
            levels = {level for row in a for level in row}
            self.assertGreaterEqual(len(levels), 3)

    def test_patterns_differ_from_each_other(self):
        frames = {
            str(mosaic_grid(40, 16, m * MOSAIC_MODE_FRAMES + 10))
            for m in range(MOSAIC_MODES)
        }

        self.assertEqual(len(frames), MOSAIC_MODES)

    def test_crossfade_is_continuous_across_pattern_boundary(self):
        # No hard cut: frame-to-frame change around each switch stays
        # within the range the patterns show on their own. A cut would
        # flip most of the grid at once.
        def changed(phase):
            a = mosaic_grid(40, 16, phase)
            b = mosaic_grid(40, 16, phase + 1)

            return sum(
                x != y
                for ra, rb in zip(a, b)
                for x, y in zip(ra, rb)
            )

        everywhere = max(
            changed(p) for p in range(MOSAIC_MODES * MOSAIC_MODE_FRAMES)
        )
        cells = 40 * 16

        for slot in range(1, MOSAIC_MODES):
            boundary = slot * MOSAIC_MODE_FRAMES

            for phase in range(
                boundary - MOSAIC_FADE_FRAMES, boundary + 2
            ):
                self.assertLessEqual(changed(phase), everywhere)
                self.assertLess(changed(phase), cells * 0.6)

    def test_fade_clears_the_inner_edge(self):
        for fade, edge_x in ((1, slice(-4, None)), (-1, slice(0, 4))):
            grid = mosaic_grid(40, 16, phase=30, fade=fade)
            edge = sum(level for row in grid for level in row[edge_x])
            total = sum(level for row in grid for level in row)

            self.assertGreater(total, 0)
            self.assertEqual(edge, 0)

    def test_frame_text_matches_grid(self):
        rows = mosaic_frame(20, 6, 4).split("\n")

        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertLessEqual(len(row), 20)
            for char in row:
                self.assertIn(char, MOSAIC_GLYPHS)


class SharedCanvasTests(unittest.TestCase):

    def test_stacked_strips_match_one_tall_grid(self):
        # Three strips of different heights/widths, all on one canvas,
        # must show exactly the cells of a single grid: no seams.
        for phase in (5, 90, 170):
            whole = mosaic_grid(40, 30, phase, seed=4)
            canvas = (40, 30)

            top = mosaic_grid(40, 10, phase, 4, 0, (0, 0), canvas)
            mid = mosaic_grid(31, 6, phase, 4, 0, (0, 10), canvas)
            low = mosaic_grid(23, 14, phase, 4, 0, (0, 16), canvas)

            self.assertEqual(top, whole[0:10])
            self.assertEqual(mid, [r[:31] for r in whole[10:16]])
            self.assertEqual(low, [r[:23] for r in whole[16:30]])

    def test_right_anchored_strips_align_odd_offsets(self):
        # An odd column offset must not shift the 2-wide tile grid.
        whole = mosaic_grid(41, 12, 33, seed=1)
        right = mosaic_grid(14, 12, 33, 1, 0, (27, 0), (41, 12))

        self.assertEqual(right, [r[27:] for r in whole])

    def test_defaults_unchanged(self):
        self.assertEqual(
            mosaic_grid(30, 12, 9, 2),
            mosaic_grid(30, 12, 9, 2, 0, (0, 0), None),
        )

    def test_sea_runs_field_seed_leaves_creatures_alone(self):
        a = sea_runs(30, 25, 12, seed=0, direction=1)
        b = sea_runs(30, 25, 12, seed=0, direction=1, field_seed=9)

        def creatures(rows):
            return [
                [(len(t), k) for t, k in runs if k == CREATURE]
                for runs in rows
            ]

        self.assertEqual(creatures(a), creatures(b))
        self.assertNotEqual(a, b)


class SeaRunsTests(unittest.TestCase):

    def flatten(self, rows):
        return ["".join(text for text, _ in runs) for runs in rows]

    def test_dimensions_and_run_merging(self):
        rows = sea_runs(30, 25, phase=3)

        self.assertEqual(len(rows), 25)
        for runs in rows:
            self.assertEqual(sum(len(t) for t, _ in runs), 30)
            for (_, k1), (_, k2) in zip(runs, runs[1:]):
                self.assertNotEqual(k1, k2)

    def test_creatures_overlay_the_mosaic(self):
        overlay = side_column(30, 25, phase=12, seed=0, direction=1)
        rows = sea_runs(30, 25, phase=12, seed=0, direction=1)

        for y, line in enumerate(self.flatten(rows)):
            src = overlay.split("\n")[y]

            for x, char in enumerate(src):
                if char != " ":
                    self.assertEqual(line[x], char)

        kinds = {k for runs in rows for _, k in runs}
        self.assertIn(CREATURE, kinds)

    def test_animates_and_mirrors_direction(self):
        self.assertNotEqual(
            sea_runs(30, 25, 4), sea_runs(30, 25, 5)
        )
        self.assertNotEqual(
            sea_runs(30, 25, 4, 0, 1), sea_runs(30, 25, 4, 0, -1)
        )

    def test_tiny_sizes_dont_crash(self):
        self.assertEqual(len(sea_runs(1, 1, 0)), 1)


class PaletteTests(unittest.TestCase):

    def test_blend_endpoints_and_midpoint(self):
        self.assertEqual(blend_hex("#000000", "#ffffff", 0), "#000000")
        self.assertEqual(blend_hex("#000000", "#ffffff", 1), "#ffffff")
        self.assertEqual(blend_hex("#000000", "#ffffff", 0.5), "#808080")

    def test_palette_shape(self):
        palette = mosaic_palette("#0b0e14", "#7dd3fc")

        self.assertEqual(len(palette), len(MOSAIC_GLYPHS) + 1)
        self.assertEqual(palette[0], "#0b0e14")
        for color in palette:
            self.assertRegex(color, r"^#[0-9a-f]{6}$")


if __name__ == "__main__":
    unittest.main()
