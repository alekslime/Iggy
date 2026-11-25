"""
Headless tests for the TUI (seacode.tui).

Constructing the app without running it is enough to cover model
selection: the configured model must reach the status pills and the
agent turns. Slash-command parsing is pure and covered directly.
Interactive behavior is intentionally not covered here.

Run with:  python3 -m pytest tests/test_tui.py -v
       or: python3 -m unittest tests.test_tui -v
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.tools as tools_module
from seacode.sessions import save_session
from seacode.tui import (
    SeaCodeApp,
    classify_line,
    context_meter,
    mascot_pulse,
    max_recents_for_height,
    parse_slash_command,
    splash_mode,
)
from textual.containers import Horizontal
from textual.widgets import Input, RichLog, Static

# Mosaic strips: beside the mascot, and beside the input stack below.
SEA_STRIPS = (
    "#sea-left", "#sea-right", "#sea-bottom-left", "#sea-bottom-right",
)


class TuiModelTests(unittest.TestCase):

    def test_default_model_matches_agent(self):
        app = SeaCodeApp()

        self.assertEqual(app._model, seacode_module.MODEL)
        self.assertIsNone(app._max_tool_calls)
        self.assertTrue(app._stream)
        self.assertIsNone(app._session_id)

    def test_custom_model_reaches_status_pills(self):
        app = SeaCodeApp(model="custom-model:tag", max_tool_calls=25)

        self.assertEqual(app._model, "custom-model:tag")
        self.assertEqual(app._max_tool_calls, 25)
        self.assertIn("custom-model:tag", app._status_pills().plain)

    def test_resume_state_kept(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        app = SeaCodeApp(
            initial_messages=messages, session_id="20261119-120000"
        )

        self.assertIs(app.messages, messages)
        self.assertEqual(app._session_id, "20261119-120000")


class ParseSlashCommandTests(unittest.TestCase):

    def test_bare_command(self):
        self.assertEqual(
            parse_slash_command("/help"), ("help", "")
        )

    def test_command_with_arg(self):
        self.assertEqual(
            parse_slash_command("/undo  notes.txt "),
            ("undo", "notes.txt"),
        )

    def test_name_is_lowercased(self):
        self.assertEqual(
            parse_slash_command("/MODEL llama3"),
            ("model", "llama3"),
        )

    def test_not_a_command(self):
        self.assertIsNone(parse_slash_command("hello"))
        self.assertIsNone(parse_slash_command(""))
        self.assertIsNone(parse_slash_command("/"))
        self.assertIsNone(parse_slash_command("// comment"))


class ClassifyLineTests(unittest.TestCase):
    """Feed styling mirrors run_agent's transcript formats."""

    def test_tool_call_is_tool(self):
        self.assertEqual(
            classify_line("● read_file demo.txt"), "tool"
        )

    def test_result_lines_are_dim(self):
        self.assertEqual(classify_line("  ⎿ hello"), "dim")
        self.assertEqual(
            classify_line("  … (+12 more lines)"), "dim"
        )

    def test_error_line_is_error(self):
        self.assertEqual(
            classify_line("  ✖ File does not exist: x."),
            "error",
        )

    def test_success_line_is_success(self):
        self.assertEqual(
            classify_line(
                "Replacement applied successfully: demo.txt"
            ),
            "success",
        )

    def test_guard_lines_are_warning(self):
        self.assertEqual(
            classify_line(
                "Blocking modification: target file hasn't been read."
            ),
            "warning",
        )
        self.assertEqual(
            classify_line(
                "Rejecting premature final answer: recovery pending."
            ),
            "warning",
        )


class SplashFitTests(unittest.TestCase):
    """The wordmark always shows; tall terminals add the mascot above
    it, swapping live on resize."""

    def displayed(self, app, widget_id, widget_cls=Static):
        return app.query_one(widget_id, widget_cls).display

    def test_short_terminal_gets_wordmark_only(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                self.assertFalse(self.displayed(app, "#splash-top", Horizontal))
                self.assertTrue(self.displayed(app, "#logo"))

        asyncio.run(run())

    def test_mid_height_gets_wordmark_only(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 40)) as pilot:
                await pilot.pause()
                self.assertFalse(self.displayed(app, "#splash-top", Horizontal))
                self.assertTrue(self.displayed(app, "#logo"))

        asyncio.run(run())

    def test_tall_terminal_gets_both(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 50)) as pilot:
                await pilot.pause()
                self.assertTrue(self.displayed(app, "#splash-top", Horizontal))
                self.assertTrue(self.displayed(app, "#logo"))

        asyncio.run(run())

    def test_just_below_threshold_stays_wordmark(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 44)) as pilot:
                await pilot.pause()
                self.assertFalse(self.displayed(app, "#splash-top", Horizontal))
                self.assertTrue(self.displayed(app, "#logo"))

        asyncio.run(run())

    def test_tiny_terminal_pins_splash_top(self):
        from textual.containers import Vertical

        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 16)) as pilot:
                await pilot.pause()
                align = app.query_one("#splash", Vertical).styles.align
                self.assertEqual(tuple(align), ("center", "top"))

        asyncio.run(run())

    def test_resize_adds_mascot(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                self.assertFalse(self.displayed(app, "#splash-top", Horizontal))

                await pilot.resize_terminal(80, 50)
                await pilot.pause()
                self.assertTrue(self.displayed(app, "#splash-top", Horizontal))
                self.assertTrue(self.displayed(app, "#logo"))

        asyncio.run(run())

    def test_rows_centered_on_wide_terminal(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(230, 40)) as pilot:
                await pilot.pause()

                for widget_id, widget_cls in (
                    ("#logo", Static),
                    ("#splash-input", Input),
                    ("#splash-status", Static),
                    ("#splash-hints", Static),
                ):
                    region = app.query_one(
                        widget_id, widget_cls
                    ).region
                    center = region.x + region.width // 2
                    self.assertLessEqual(
                        abs(center - 115), 1,
                        f"{widget_id} not centered: {region}",
                    )

        asyncio.run(run())


class EmptyChatStateTests(unittest.TestCase):

    def test_feed_offers_examples(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                app.chatting = True
                await pilot.pause()
                texts = [str(line.text) for line in app.feed.lines]
                self.assertTrue(any("Try:" in t for t in texts))

        asyncio.run(run())


class SidebarTests(unittest.TestCase):

    def test_sidebar_only_on_wide_chat(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(160, 40)) as pilot:
                await pilot.pause()
                sidebar = app.query_one("#sidebar", Static)
                self.assertFalse(sidebar.display)
                app.chatting = True
                await pilot.pause()
                self.assertTrue(sidebar.display)
                self.assertIn("Project", str(sidebar.render()))
                await pilot.resize_terminal(80, 40)
                await pilot.pause()
                self.assertFalse(sidebar.display)

        asyncio.run(run())


class ContextMeterTests(unittest.TestCase):

    def test_empty_and_full(self):
        self.assertEqual(context_meter([], 0), "")
        self.assertEqual(context_meter([], 100), "ctx ░░░░░ 0%")
        meter = context_meter(
            [{"role": "user", "content": "x" * 400}], 100
        )
        self.assertIn("100%", meter)
        self.assertIn("▓▓▓▓▓", meter)


class PermissionCardTests(unittest.TestCase):

    def test_prompt_label_includes_diff_stats(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                app.chatting = True
                await pilot.pause()
                app.show_permission_prompt(
                    "modify file: a.py",
                    lambda approved: None,
                    diff="--- a/a.py\n+++ b/a.py\n-old\n+new\n+more\n",
                )
                await pilot.pause()
                label = str(
                    app.query_one("#prompt-label", Static).render()
                )
                self.assertIn("modify file: a.py", label)
                self.assertIn("(+2/-1)", label)

        asyncio.run(run())


class ToolBlockTests(unittest.TestCase):
    """Tool calls open a ruled block in the feed."""

    def test_tool_line_gets_rule_and_separator(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                app.chatting = True
                await pilot.pause()
                feed = app.query_one("#feed", RichLog)
                before = len(feed.lines)
                app.handle_agent_line("● read_file agent.py")
                await pilot.pause()
                self.assertEqual(len(feed.lines) - before, 2)
                last = str(feed.lines[-1].text)
                self.assertIn("▎", last)
                self.assertIn("read_file", last)

        asyncio.run(run())


class ThemeTests(unittest.TestCase):

    def test_invalid_theme_falls_back(self):
        self.assertEqual(SeaCodeApp(theme="bogus")._theme_name, "abyss")
        self.assertEqual(SeaCodeApp(theme="coral")._theme_name, "coral")

    def test_slash_theme_switches_live(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                app.chatting = True
                await pilot.pause()
                app.handle_slash_command("theme", "kelp")
                await pilot.pause()
                self.assertEqual(app._theme_name, "kelp")
                app.handle_slash_command("theme", "nope")
                self.assertEqual(app._theme_name, "kelp")

        asyncio.run(run())


class WorkingIndicatorTests(unittest.TestCase):

    def test_status_bar_shows_working_fish_during_turn(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                app.chatting = True
                app.turn_running = True
                await pilot.pause()
                left = str(
                    app.query_one("#status-left", Static).render()
                )
                self.assertIn("working", left)
                app.turn_running = False
                await pilot.pause()
                left = str(
                    app.query_one("#status-left", Static).render()
                )
                self.assertNotIn("working", left)

        asyncio.run(run())


class StatusBarDedupTests(unittest.TestCase):
    """Splash status bar shows only the version; chat shows all."""

    def test_splash_then_chat(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                right = str(
                    app.query_one("#status-right", Static).render()
                )
                self.assertNotIn(app._model, right)
                app.chatting = True
                await pilot.pause()
                right = str(
                    app.query_one("#status-right", Static).render()
                )
                self.assertIn(app._model, right)

        asyncio.run(run())


class SplashModeTests(unittest.TestCase):

    def test_breakpoints(self):
        self.assertEqual(splash_mode(10), "compact")
        self.assertEqual(splash_mode(19), "compact")
        self.assertEqual(splash_mode(24), "standard")
        self.assertEqual(splash_mode(44), "standard")
        self.assertEqual(splash_mode(50), "full")


class MaxRecentsTests(unittest.TestCase):
    """Recent sessions fill leftover splash rows, at most five."""

    def test_tiny_terminal_shows_few(self):
        self.assertEqual(max_recents_for_height(24, False), 3)

    def test_short_terminal_shows_five(self):
        self.assertEqual(max_recents_for_height(28, False), 5)

    def test_roomy_terminal_shows_five(self):
        self.assertEqual(max_recents_for_height(40, False), 5)

    def test_tall_mascot_mode_leaves_some_room(self):
        self.assertEqual(max_recents_for_height(50, True), 3)

    def test_huge_terminal_shows_five(self):
        self.assertEqual(max_recents_for_height(70, True), 5)


class MascotPulseTests(unittest.TestCase):

    def test_pulse_is_derived_from_theme_color(self):
        operate = mascot_pulse("#4f9de8", "operate")
        scout = mascot_pulse("#4f9de8", "scout")

        self.assertEqual(len(operate), 4)
        self.assertEqual(operate[0], "#4f9de8")
        self.assertNotEqual(operate, scout)
        # Default theme's seahorse is blue: blue channel dominates.
        for color in operate + scout:
            r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
            self.assertGreater(b, r)


class SplashAnimationTests(unittest.TestCase):
    """The splash shimmers until a chat starts (or --no-anim)."""

    def test_bottom_strips_flank_the_input_stack(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(140, 45)) as pilot:
                await pilot.pause()
                left = app.query_one("#sea-bottom-left").region
                right = app.query_one("#sea-bottom-right").region
                inp = app.query_one("#splash-input").region

                self.assertGreater(left.width, 0)
                self.assertGreater(right.width, 0)
                self.assertLessEqual(left.right, inp.x)
                self.assertGreaterEqual(right.x, inp.right)

        asyncio.run(run())

    def rule_text(self, app):
        return str(app.query_one("#splash-rule", Static).render())

    def test_tick_advances_frame(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                # Freeze the live timer so only the manual tick lands.
                app._stop_art()
                phase = app._art_phase
                before = self.rule_text(app)
                seas_before = [
                    str(
                        app.query_one(widget_id, Static).render()
                    )
                    for widget_id in SEA_STRIPS
                ]
                app._tick_art()
                await pilot.pause()
                self.assertEqual(app._art_phase, phase + 1)
                self.assertNotEqual(self.rule_text(app), before)

                for widget_id, previous in zip(SEA_STRIPS, seas_before):
                    self.assertNotEqual(
                        str(
                            app.query_one(widget_id, Static).render()
                        ),
                        previous,
                    )

        asyncio.run(run())

    def test_no_anim_starts_no_timer(self):
        async def run():
            app = SeaCodeApp(animate=False)
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                self.assertIsNone(app._art_timer)

        asyncio.run(run())

    def test_chat_stops_animation(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                self.assertIsNotNone(app._art_timer)
                app.chatting = True
                await pilot.pause()
                self.assertIsNone(app._art_timer)

        asyncio.run(run())


class ShowDiffTests(unittest.TestCase):
    """Diffs render inline in the transcript, truncated when long."""

    def test_short_diff_writes_header_and_hint(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                app.chatting = True
                await pilot.pause()
                feed = app.query_one("#feed", RichLog)
                before = len(feed.lines)
                app.show_diff(
                    "demo.txt",
                    "--- a/demo.txt\n+++ b/demo.txt\n@@ -1 +1 @@\n"
                    "-v1\n+v2\n",
                )
                await pilot.pause()
                # Header + diff + undo hint land in the transcript.
                self.assertGreaterEqual(len(feed.lines) - before, 3)

        asyncio.run(run())

    def test_long_diff_truncates(self):
        async def run():
            app = SeaCodeApp()
            async with app.run_test(size=(80, 30)) as pilot:
                await pilot.pause()
                app.chatting = True
                await pilot.pause()
                feed = app.query_one("#feed", RichLog)
                before = len(feed.lines)
                big = "\n".join(f"+line {n}" for n in range(100))
                app.show_diff("big.txt", big)
                await pilot.pause()
                # Far fewer lines than the raw diff: truncated + noted.
                self.assertLess(len(feed.lines) - before, 50)

        asyncio.run(run())


class SplashResumeTests(unittest.TestCase):
    """The splash input resumes sessions by number or id prefix."""

    def test_number_resumes_recent_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            session_id = save_session(
                root,
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "earlier work"},
                ],
                "m",
                turns=1,
            )

            async def run():
                app = SeaCodeApp()
                async with app.run_test(size=(80, 30)) as pilot:
                    await pilot.pause()
                    await pilot.press("1")
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertTrue(app.chatting)
                    self.assertEqual(app._session_id, session_id)
                    self.assertEqual(app._turns, 1)
                    self.assertFalse(app.turn_running)

            with patch.object(
                tools_module, "get_project_root", return_value=root
            ):
                asyncio.run(run())

    def test_unknown_number_stays_on_splash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            async def run():
                app = SeaCodeApp()
                async with app.run_test(size=(80, 30)) as pilot:
                    await pilot.pause()
                    await pilot.press("9")
                    await pilot.press("enter")
                    await pilot.pause()
                    self.assertFalse(app.chatting)
                    self.assertFalse(app.turn_running)

            with patch.object(
                tools_module, "get_project_root", return_value=root
            ):
                asyncio.run(run())

    def test_plain_text_is_not_a_resume(self):
        app = SeaCodeApp()
        self.assertFalse(app._try_resume_from_splash("hello there"))


if __name__ == "__main__":
    unittest.main()
