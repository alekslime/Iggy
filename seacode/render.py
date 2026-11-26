"""Terminal rendering for the agent transcript.

ANSI color helpers, tool-call / tool-result formatting, durations and
the "thinking" spinner. Pure output helpers: nothing here touches the
model, the tools or any agent state.

The line formats produced here are parsed by the TUI
(see seacode.tui.classify_line); change them in lockstep.
"""

import itertools
import json
import os
import sys
import threading


# -----------------------------------------------------------------------
# Transcript output -- everything run_agent prints is plain text (no
# ANSI codes): the TUI parses these exact formats to style its feed,
# so any change here must update tui.py's classifier in lockstep.
# Only run_repl's own header/prompt/answer prints use color, and only
# on a tty without NO_COLOR set.
# -----------------------------------------------------------------------

MAX_RESULT_LINES = 8

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_TEXT = " thinking…"
SPINNER_INTERVAL = 0.08


def _color_enabled() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _paint(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _paint(text, "1")


def _dim(text: str) -> str:
    return _paint(text, "2")


def _cyan(text: str) -> str:
    return _paint(text, "36")


def _green(text: str) -> str:
    return _paint(text, "32")


def format_tool_call(tool_name: str, arguments) -> str:
    """One-line `● tool target` summary for the transcript, where
    target is the path/command/query being acted on, if any."""

    target = ""

    if isinstance(arguments, dict):
        for key in ("path", "command", "query", "root", "prompt"):
            value = arguments.get(key)
            if isinstance(value, str) and value:
                target = value
                break

    line = f"● {tool_name}"

    if target:
        if len(target) > 70:
            target = target[:67] + "..."
        line += f" {target}"

    return line


def print_tool_result(result) -> None:
    """Print a tool's result, dimmed by the caller contract: first
    line marked ⎿, continuations indented, long output truncated."""

    if isinstance(result, (dict, list)):
        text = json.dumps(result, indent=2)
    else:
        text = str(result)

    lines = text.splitlines() or [""]

    for index, line in enumerate(lines[:MAX_RESULT_LINES]):
        if index == 0:
            print(f"  ⎿ {line}".rstrip())
        else:
            print(f"    {line}".rstrip())

    hidden = len(lines) - MAX_RESULT_LINES

    if hidden > 0:
        print(f"  … (+{hidden} more lines)")


def format_duration(seconds: float) -> str:
    """Compact turn duration: "3s", "1m02s"."""

    total = max(int(seconds), 0)
    minutes, secs = divmod(total, 60)

    if minutes:
        return f"{minutes}m{secs:02d}s"

    return f"{secs}s"


class _ThinkingSpinner:
    """Transient braille spinner while the model thinks. Active only
    on a tty; everywhere else (pipes, tests) it's a silent no-op."""

    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdout.isatty()
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if not self.enabled:
            return self

        self._thread = threading.Thread(
            target=self._spin, daemon=True
        )
        self._thread.start()

        return self

    def _spin(self) -> None:
        for frame in itertools.cycle(SPINNER_FRAMES):
            if self._stop.wait(SPINNER_INTERVAL):
                break
            print(f"\r{frame}{SPINNER_TEXT}", end="", flush=True)

    def __exit__(self, *exc_info) -> bool:
        if not self.enabled:
            return False

        self._stop.set()
        self._thread.join(timeout=1.0)
        width = len(SPINNER_FRAMES[0] + SPINNER_TEXT)
        print("\r" + " " * width + "\r", end="", flush=True)

        return False
