"""
SeaCode TUI -- a terminal interface for the agent, styled after OpenCode's
splash-screen look: art and a centered rounded input up front, which
hands off to a single-column transcript once a conversation actually
starts, with a slim status bar pinned to the bottom throughout.

Design choice (per project discussion): seacode.py's run_agent() keeps
its existing print()-based logging exactly as-is -- no refactor to
structured events. This app runs run_agent() in a background thread,
redirects stdout into itself so every line lands in the transcript as
it's produced, and layers inline diffs + a simple y/N permission
prompt on top, entirely from the outside.

Two things needed a small bridge to make that work inside a real
terminal app rather than a plain REPL:

  - stdout: Python's sys.stdout is one global, so _ThreadedStdoutBridge
    temporarily takes it over for the duration of a single agent turn
    (run from a worker thread) and forwards completed lines back to the
    UI thread via `call_from_thread`. This assumes only one agent turn
    is ever in flight at a time, which the UI already enforces by
    disabling input while a turn is running.

  - permissions: seacode/tools.py calls seacode.permissions.request_permission,
    which normally blocks on a raw input(). Textual owns the terminal,
    so that call is monkeypatched (on mount) to _PermissionBridge.request,
    which hands control to the UI thread, blocks the worker thread on a
    threading.Event, and resumes once the person answers in the prompt
    line at the bottom of the screen.

Run with:  seacode tui

Slash commands (typed like a chat message): /help /undo /sessions
/model /scout /operate /clear. Conversations autosave per completed
turn and resume with `seacode tui --resume`.
"""

import contextlib
import difflib
import random
import re
import threading
import time

from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Input, RichLog, Static

import seacode.agent as seacode_module
import seacode.context as context_module
import seacode.sessions as sessions_module
import seacode.tools as tools_module
from seacode import __version__ as VERSION
from seacode import permissions as permissions_module
from seacode.art import MASCOT, wave_rule, side_column

LOGO = (
    "█▀▀█ ▀█▀▀█   ▀▀▄  █▀▀█ █▀▀█ ▀█▀▀█ ▀█▀▀█\n"
    "█▄▄▄  █▄▄   █▄▄█  █    █  █  █  █  █▄▄\n"
    "   █  █  ▄  █  █  █  ▄ █  █  █  █  █  ▄\n"
    "▀▀▀▀ ▀▀▀▀▀ ▀▀  ▀ ▀▀▀▀▀ ▀▀▀▀ ▀▀▀▀▀ ▀▀▀▀▀"
)

# Splash idle animation: the dither rule shimmers through scatter
# phases while the mascot breathes through green shades. The timer
# stops the moment a chat starts, so nothing moves behind the user.
ART_TICK_SECONDS = 0.15
MASCOT_PULSE = ("#4caf78", "#5cc68a", "#4caf78", "#3d8f61")

# The mascot breathes in the mode's key: calm bright mint in Scout
# (read-only), the classic greens in Operate.
MASCOT_PULSE_BY_MODE = {
    "operate": MASCOT_PULSE,
    "scout": ("#5cc68a", "#8bf0b3", "#5cc68a", "#4caf78"),
}

WORKING_INDICATOR = "><((°> working…"

# Visual themes: one palette per name, applied live to the screen,
# feed, mascot, and wordmark. Configured via seacode.toml's theme
# key or switched mid-chat with /theme.
THEMES = {
    "abyss": {
        "bg": "#0b0e14", "panel": "#10141d", "accent": "#7dd3fc",
        "mascot": "#4caf78", "logo": "#e8eaed",
    },
    "kelp": {
        "bg": "#07130d", "panel": "#0c1f15", "accent": "#7be3a2",
        "mascot": "#3fa66a", "logo": "#e6f5ea",
    },
    "coral": {
        "bg": "#170e0a", "panel": "#241611", "accent": "#ffb08a",
        "mascot": "#f0a35e", "logo": "#fdf0e6",
    },
    "mono": {
        "bg": "#0a0a0a", "panel": "#141414", "accent": "#d0d5db",
        "mascot": "#9aa3ad", "logo": "#f2f2f2",
    },
}

DEFAULT_THEME = "abyss"


# Splash vertical budget, in rows, excluding the recent-sessions block
# (which sizes itself to leftover space -- see max_recents_for_height):
# art + dither rule + input + pills + hints + stats + tip + margins +
# the status bar row.
SPLASH_BASE_ROWS = 18
MASCOT_EXTRA_ROWS = len(MASCOT.splitlines()) + 1
MASCOT_MIN_HEIGHT = SPLASH_BASE_ROWS + MASCOT_EXTRA_ROWS + 1


def splash_mode(height: int) -> str:
    """Splash layout for a terminal height: "compact" (wordmark and
    input only), "standard" (plus sessions/stats), or "full" (the
    mascot joins above the wordmark). One place decides, so resize
    handling and tests agree."""

    if height < 20:
        return "compact"

    if height < MASCOT_MIN_HEIGHT:
        return "standard"

    return "full"


def context_meter(messages, limit_tokens) -> str:
    """One compact gauge: "ctx ▓▓░░░ 41%". Empty when no budget."""

    if not limit_tokens:
        return ""

    used = sum(
        context_module.estimate_message_tokens(m) for m in messages
    )
    percent = min(used * 100 // max(limit_tokens, 1), 100)
    filled = percent * 5 // 100
    bar = "▓" * filled + "░" * (5 - filled)

    return f"ctx {bar} {percent}%"


def max_recents_for_height(height: int, mascot_shown: bool) -> int:
    """How many recent sessions fit the splash at this height: leftover
    rows after the fixed stack, minus the block's own header row."""

    base = SPLASH_BASE_ROWS

    if mascot_shown:
        base += MASCOT_EXTRA_ROWS

    room = (height - 1) - base - 2

    return max(0, min(5, room))


# -----------------------------------------------------------------------
# stdout bridge
# -----------------------------------------------------------------------


class _ThreadedStdoutBridge:
    """Stand-in for sys.stdout while an agent turn runs in a worker
    thread. Buffers partial writes into whole lines and hands each
    finished line to the App on the UI thread.
    """

    def __init__(self, app: "SeaCodeApp"):
        self._app = app
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._app.call_from_thread(self._app.handle_agent_line, line)
        return len(text)

    def flush(self) -> None:
        pass

    def drain(self) -> None:
        """Flush any trailing partial line with no terminating \\n."""
        if self._buffer:
            self._app.call_from_thread(
                self._app.handle_agent_line, self._buffer
            )
            self._buffer = ""


# -----------------------------------------------------------------------
# permission bridge
# -----------------------------------------------------------------------


class _PermissionBridge:
    """Replaces seacode.tools.request_permission for the duration of the
    TUI session. Called from the worker thread; blocks it on a
    threading.Event until the person answers in the UI thread.
    """

    def __init__(self, app: "SeaCodeApp"):
        self._app = app

    def request(self, action: str, diff: str | None = None) -> bool:
        # Scout mode is read-only: decline every mutating action
        # automatically, without ever blocking the worker thread on an
        # interactive prompt. This is the only place that needs to know
        # about modes at all -- write_file/replace_in_file/run_command
        # all already route through request_permission, so nothing in
        # seacode.py or tools.py has to change.
        if self._app.mode == "scout":
            self._app.call_from_thread(
                self._app.note_scouted_action, action
            )
            return False

        # Otherwise honor the trust policy first: --trust edits/all
        # (or SEACODE_TRUST / seacode.toml) auto-approves without prompting.
        if permissions_module.should_auto_approve(action):
            self._app.call_from_thread(
                self._app.note_auto_approved_action, action
            )
            if diff:
                self._app.call_from_thread(
                    self._app.show_permission_diff, diff
                )
            return True

        event = threading.Event()
        result = {"approved": False}

        def on_answer(approved: bool) -> None:
            result["approved"] = approved
            event.set()

        self._app.call_from_thread(
            self._app.show_permission_prompt, action, on_answer, diff
        )
        event.wait()

        return result["approved"]


# -----------------------------------------------------------------------
# token bridge -- streaming deltas arrive from the worker thread via
# run_agent's on_token callback. Like _ThreadedStdoutBridge, this
# coalesces them into whole lines for the feed; a trailing partial
# line is flushed when the turn ends.
# -----------------------------------------------------------------------


class _TokenLineBuffer:
    def __init__(self, app: "SeaCodeApp"):
        self._app = app
        self._buffer = ""

    def push(self, delta: str) -> None:
        self._buffer += delta
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._app.call_from_thread(
                self._app.handle_stream_line, line
            )

    def drain(self) -> None:
        if self._buffer:
            self._app.call_from_thread(
                self._app.handle_stream_line, self._buffer
            )
            self._buffer = ""


# -----------------------------------------------------------------------
# slash commands -- parsed without a running app so they stay unit
# testable. See SeaCodeApp.handle_slash_command for the handlers.
# -----------------------------------------------------------------------

SLASH_HELP = (
    "/help            show this list\n"
    "/undo [path]     restore a file from its most recent backup\n"
    "/sessions        list saved conversation sessions\n"
    "/model [name]    show or switch the model\n"
    "/tools           list built-in, MCP and subagent tools\n"
    "/theme [name]    show or switch theme (abyss/kelp/coral/mono)\n"
    "/scout           read-only mode (auto-deny modifications)\n"
    "/operate         normal mode (prompt for modifications)\n"
    "/clear           clear the feed"
)


TIPS = (
    "Tip: /undo reverts the last change",
    "Tip: Shift+Tab toggles read-only Scout mode",
    "Tip: resume any session with --resume",
    "Tip: /model switches models mid-chat",
    "Tip: every edit is backed up before it's made",
    "Tip: /sessions lists your past conversations",
)


def parse_slash_command(text: str):
    """Split `/name arg...` into (name, arg); None if not a command."""

    if not text.startswith("/") or text.startswith("//"):
        return None

    parts = text[1:].split(None, 1)

    if not parts or not parts[0]:
        return None

    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    return name, arg


# -----------------------------------------------------------------------
# line classification -- lightweight, read-only parsing of seacode.py's
# existing print() output. Nothing here changes seacode.py's behavior.
# -----------------------------------------------------------------------

_MODIFICATION_SUCCESS_RE = re.compile(
    r"^(?:⎿ )?(Replacement applied successfully|File written successfully): "
    r"(.+)$"
)
_REVERT_RE = re.compile(
    r"^Verification failed for (?:newly created )?.+?; "
    r"(?:reverted to its pre-edit content|removed it)\.$"
)
_BLOCKED_RE = re.compile(r"^Blocking (modification|write_file):")
_REJECTED_RE = re.compile(r"^Rejecting (premature final answer|final answer):")
_STOPPED_RE = re.compile(r"^Agent stopped after")

STYLE_MAP = {
    "error": "bold red",
    "warning": "bold yellow",
    "success": "bold green",
    "tool": "bold #7dd3fc",
    "dim": "grey50",
    "default": "#d0d5db",
}


def classify_line(line: str) -> str:
    """Return a style key (see STYLE_MAP) for a line of stdout output.

    Mirrors run_agent's transcript formats (see seacode.py's output
    section): `● tool target` calls, `⎿` result lines, `…` truncation
    notes, and `✖` errors.
    """

    stripped = line.strip()

    if not stripped:
        return "dim"
    if _REVERT_RE.match(stripped) or _STOPPED_RE.match(stripped):
        return "error"
    if stripped.startswith("✖"):
        return "error"
    if _BLOCKED_RE.match(stripped) or _REJECTED_RE.match(stripped):
        return "warning"
    if _MODIFICATION_SUCCESS_RE.match(stripped):
        return "success"
    if stripped.startswith("● "):
        return "tool"
    if stripped.startswith("⎿") or stripped.startswith("…"):
        return "dim"
    if stripped.startswith("Model raw response:"):
        return "dim"
    return "default"


class SeaCodeApp(App):
    """The SeaCode TUI."""

    CSS = """
    Screen {
        background: #0b0e14;
    }

    /* ---------------- splash ---------------- */

    #splash {
        height: 1fr;
        align: center middle;
    }
    #splash-top {
        width: 1fr;
        height: auto;
        margin-bottom: 1;
    }
    #sea-left, #sea-right {
        width: 1fr;
        color: #2e6b85;
    }
    #splash-mascot {
        width: auto;
        height: auto;
    }
    #mascot {
        width: auto;
        height: auto;
        color: #4caf78;
        text-align: left;
    }
    #logo {
        width: auto;
        color: #e8eaed;
        text-style: bold;
        text-align: left;
        margin-bottom: 1;
    }
    #splash-input {
        width: 64;
        border: round #3a4152;
        background: #10141d;
        padding: 0 1;
    }
    #splash-input:focus {
        border: round #7dd3fc;
    }
    #splash-status {
        width: 64;
        content-align: center middle;
        margin-top: 1;
    }
    #splash-hints {
        width: 64;
        content-align: center middle;
        color: #7a8494;
        margin-top: 1;
    }
    #splash-recents {
        width: 64;
        margin-top: 1;
    }
    #splash-stats {
        width: 64;
        content-align: center middle;
        color: #7a8494;
        margin-top: 1;
    }
    #splash-tip {
        width: 64;
        content-align: center middle;
        color: #7a8494;
    }
    .dither-rule {
        width: 64;
        height: 1;
        color: #2e6b85;
        margin-top: 1;
        margin-bottom: 1;
    }
    .hidden {
        display: none;
    }

    /* ---------------- transcript (full width) ---------------- */

    #feed {
        height: 1fr;
        background: #0b0e14;
        scrollbar-color: #3a4152;
        scrollbar-color-hover: #7dd3fc;
        scrollbar-background: #0b0e14;
        padding: 0 1;
    }
    #chat-body {
        height: 1fr;
    }
    #sidebar {
        width: 32;
        border-left: solid #3a4152;
        padding: 0 1;
        color: #8b93a1;
    }
    #prompt-label {
        height: auto;
        min-height: 0;
        padding: 0 1;
        color: $warning;
        text-style: bold;
    }
    #prompt-label.active {
        background: #2a2118;
        border-left: thick $warning;
    }

    /* ---------------- bottom chrome ---------------- */

    #bottom-bar {
        dock: bottom;
        height: auto;
    }
    #chat-input {
        border: tall #3a4152;
    }
    #chat-input.mode-operate {
        border: tall #e8a33d;
    }
    #chat-input.mode-scout {
        border: tall #4caf78;
    }
    #status-bar {
        height: 1;
        background: #10141d;
        color: #7a8494;
    }
    #status-left {
        width: 1fr;
        content-align: left middle;
        padding-left: 1;
    }
    #status-right {
        width: auto;
        content-align: right middle;
        padding-right: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        # priority=True so this wins over the Screen's own shift+tab
        # binding (focus_previous), which would otherwise shadow it
        # any time an Input has focus -- i.e. almost always here.
        Binding("shift+tab", "toggle_mode", "Scout/Operate", priority=True),
    ]

    MODE_LABELS = {
        "scout": "Scout (read-only)",
        "operate": "Operate (can write)",
    }
    MODE_COLORS = {
        "scout": "#4caf78",
        "operate": "#e8a33d",
    }

    turn_running: reactive[bool] = reactive(False)
    awaiting_permission: reactive[bool] = reactive(False)
    mode: reactive[str] = reactive("operate", init=False)
    chatting: reactive[bool] = reactive(False, init=False)

    def __init__(
        self,
        model=None,
        max_tool_calls=None,
        stream=True,
        initial_messages=None,
        session_id=None,
        initial_turns=0,
        animate=True,
        theme=None,
    ) -> None:
        super().__init__()
        self.title = "SeaCode"
        self._animate = animate
        self._theme_name = theme if theme in THEMES else DEFAULT_THEME
        self._theme = THEMES[self._theme_name]
        self._art_phase = 0
        self._art_timer = None
        self._model = model or seacode_module.MODEL
        self._max_tool_calls = max_tool_calls
        self._stream = stream
        self._session_id = session_id
        self._turns = initial_turns
        self._recent_sessions = []
        self._tip = random.choice(TIPS)

        if initial_messages is not None:
            self.messages = context_module.fresh_messages(
                initial_messages,
                seacode_module.system_prompt_for_model(self._model),
                tools_module.get_project_root(),
            )
        else:
            self.messages = [
                {
                    "role": "system",
                    "content": context_module.build_system_prompt(
                        seacode_module.system_prompt_for_model(
                            self._model
                        ),
                        tools_module.get_project_root(),
                    ),
                }
            ]
        self._permission_bridge = _PermissionBridge(self)
        self._permission_callback = None
        self._suppressing_raw_response = False
        self._project_name = tools_module.get_project_root().name

    def compose(self) -> ComposeResult:
        # Every splash row gets its own Center: direct children of one
        # Vertical share a single x (the widest child's), and mixing
        # Center-wrapped rows with direct ones strands the direct rows
        # at x=0 on wide terminals. Uniform Centers center every row.
        with Vertical(id="splash"):
            with Horizontal(id="splash-top"):
                yield Static("", id="sea-left")
                with Vertical(id="splash-mascot"):
                    yield Static(MASCOT, id="mascot")
                yield Static("", id="sea-right")
            with Center():
                yield Static(LOGO, id="logo")
            with Center():
                yield Input(
                    placeholder='Ask anything... e.g. "delete example.txt"',
                    id="splash-input",
                )
            with Center():
                yield Static("", id="splash-status")
            with Center():
                yield Static(
                    "shift+tab scout/operate    ctrl+c quit",
                    id="splash-hints",
                )
            with Center():
                yield Static("", id="splash-recents")
            with Center():
                yield Static("", id="splash-stats")
            with Center():
                yield Static("", id="splash-tip")
            with Center():
                yield Static(
                    wave_rule(64),
                    classes="dither-rule",
                    id="splash-rule",
                )
        with Horizontal(id="chat-body", classes="hidden"):
            yield RichLog(
                id="feed",
                wrap=True,
                highlight=False,
                markup=False,
                classes="hidden",
            )
            yield Static("", id="sidebar", classes="hidden")
        with Vertical(id="bottom-bar"):
            yield Static("", id="prompt-label", classes="hidden")
            yield Input(
                placeholder="Ask SeaCode to do something...",
                id="chat-input",
                classes="hidden",
            )
            with Horizontal(id="status-bar"):
                yield Static("", id="status-left")
                yield Static("", id="status-right")

    def _apply_theme(self) -> None:
        """Paint the current theme onto the live widgets."""

        theme = self._theme
        self.screen.styles.background = theme["bg"]
        self.query_one("#feed", RichLog).styles.background = theme["bg"]
        self.query_one("#status-bar", Horizontal).styles.background = (
            theme["panel"]
        )
        self.query_one("#mascot", Static).styles.color = theme["mascot"]
        self.query_one("#logo", Static).styles.color = theme["logo"]

    def on_mount(self) -> None:
        # Route seacode/tools.py's permission checks through the TUI
        # instead of the raw input() prompt seacode.permissions uses by
        # default. Only rebinds the name in the tools module, exactly
        # like the test suite's own patch.object(tools_module, ...).
        tools_module.request_permission = self._permission_bridge.request

        # mode/chatting are reactive(init=False) so their watchers don't
        # fire on their own during construction -- apply the initial
        # styling by hand here, once widgets actually exist to style.
        self._apply_mode_styling(self.mode)
        self._apply_status_bar()
        self._apply_theme()
        self._fit_splash()

        self.feed.write(
            Text(
                "SeaCode is ready. Type a message below and press Enter.",
                style="bold",
            )
        )
        self.feed.write(
            Text(
                "Shift+Tab toggles Scout (read-only) / Operate (can write).",
                style="grey50",
            )
        )
        self.feed.write(
            Text(
                "Try:  explore this project  ·  explain how the agent "
                "loop works  ·  run the tests and fix any failures",
                style="#7a8494",
            )
        )

        self._populate_splash_extras()
        self._fit_splash_vertical()
        # Sizes aren't final until after first layout, so re-check
        # once everything has been measured.
        self.call_after_refresh(self._fit_splash_vertical)
        self.call_after_refresh(self._tick_art)

        if self._animate and not self.chatting:
            self._art_timer = self.set_interval(
                ART_TICK_SECONDS, self._tick_art
            )

        if len(self.messages) > 1:
            self.chatting = True
            self._announce_resume()

        if not self.chatting:
            self.query_one("#splash-input", Input).focus()

    def _announce_resume(self) -> None:
        self.feed.write(
            Text(
                f"\nResumed session {self._session_id} "
                f"({len(self.messages)} messages).",
                style="bold magenta",
            )
        )

        for message in self.messages[1:]:
            if message.get("role") == "user":
                self.feed.write(
                    Text(
                        f"❯ {message.get('content', '')}",
                        style="bold cyan",
                    )
                )

    def _fit_splash_vertical(self) -> None:
        """Center the splash column when it fits, else pin it to the
        top so overflow clips the least essential bottom rows (tip,
        recents) instead of the art and input."""

        splash = self.query_one("#splash", Vertical)
        total = 0

        for child in splash.children:
            if not child.display:
                continue
            total += child.outer_size.height

        if total > splash.size.height:
            splash.styles.align = ("center", "top")
        else:
            splash.styles.align = ("center", "middle")

    def _populate_splash_extras(self, height=None) -> None:
        if height is None:
            height = self.size.height

        mascot_shown = splash_mode(height) == "full"
        cap = max_recents_for_height(height, mascot_shown)
        recents = self._recent_lines(cap)

        if recents is None:
            self.query_one("#splash-recents", Static).set_class(
                True, "hidden"
            )
        else:
            self.query_one("#splash-recents", Static).set_class(
                False, "hidden"
            )
            self.query_one("#splash-recents", Static).update(recents)

        stats = self._stats_line()
        stats_widget = self.query_one("#splash-stats", Static)

        if stats:
            stats_widget.update(stats)
        else:
            stats_widget.set_class(True, "hidden")

        self.query_one("#splash-tip", Static).update(self._tip)

    def _recent_lines(self, cap: int):
        """Numbered recent sessions (at most cap), or None when there
        are none to show."""

        try:
            root = tools_module.get_project_root()
            entries = sessions_module.list_sessions(root)[:cap]
        except Exception:
            entries = []

        self._recent_sessions = entries

        if not entries:
            return None

        text = Text(
            "Recent sessions — type a number to resume:\n",
            style="grey50",
        )

        for number, entry in enumerate(entries, start=1):
            text.append(f"  {number}  ", style="bold #7dd3fc")
            preview = entry["preview"][:40]
            label = sessions_module.turns_label(entry["turns"])
            when = sessions_module.relative_time(entry["updated"])
            meta = f"{when} · {label}" if when else label
            text.append(
                f"{meta}  {preview}\n",
                style="#8b93a1",
            )

        return text

    def _stats_line(self) -> str:
        """One dim line: file count plus git working-tree state."""

        try:
            count = len(tools_module.list_files("."))
        except Exception:
            return ""

        parts = [f"{count} file" if count == 1 else f"{count} files"]

        try:
            status = tools_module.git_status()
        except Exception:
            status = None

        if status is not None:
            changed = len(status.strip().splitlines()) if status.strip() else 0

            if changed:
                parts.append(f"git: {changed} changed")
            else:
                parts.append("git: clean")

        return "  ·  ".join(parts)

    def on_resize(self, event) -> None:
        # event.size is the new size; self.size still holds the old
        # one while this handler runs.
        self._fit_splash(height=event.size.height)
        self._populate_splash_extras(height=event.size.height)
        self._fit_splash_vertical()
        self._update_sidebar(width=event.size.width)

    def _fit_splash(self, height=None) -> None:
        """Show the mascot row (seahorse between animated sea strips)
        above the wordmark when it fits; the wordmark itself always
        shows."""

        if height is None:
            height = self.size.height

        tall = splash_mode(height) == "full"
        self.query_one("#splash-top", Horizontal).set_class(
            not tall, "hidden"
        )
        self.query_one("#logo", Static).remove_class("hidden")

    @property
    def feed(self) -> RichLog:
        return self.query_one("#feed", RichLog)

    # -------------------------------------------------------------
    # input handling
    # -------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""

        if self.awaiting_permission:
            self._answer_permission(value)
            return

        if not value or self.turn_running:
            return

        slash = parse_slash_command(value)

        if slash is not None:
            # Slash commands never start a chat turn, so they work
            # from the splash screen too.
            self.handle_slash_command(*slash)
            return

        if not self.chatting and self._try_resume_from_splash(value):
            return

        if not self.chatting:
            self.chatting = True

        self.submit_user_message(value)

    def _try_resume_from_splash(self, value: str) -> bool:
        """Resume a session from the splash input: a 1-5 number picks
        a recent session, a bare id/prefix loads it directly. Anything
        else returns False so the text starts a chat instead."""

        session_ref = None

        if value.isdigit() and 1 <= int(value) <= len(
            self._recent_sessions
        ):
            session_ref = self._recent_sessions[int(value) - 1]["id"]
        elif re.fullmatch(r"[0-9][0-9\-]*", value):
            session_ref = value

        if session_ref is None:
            return False

        try:
            root = tools_module.get_project_root()
            session = sessions_module.load_session(root, session_ref)
        except (FileNotFoundError, ValueError) as e:
            self.feed.write(
                Text(
                    f"Cannot resume session: {e}",
                    style="bold yellow",
                )
            )
            return True

        self.messages = context_module.fresh_messages(
            session["messages"],
            seacode_module.system_prompt_for_model(self._model),
            tools_module.get_project_root(),
        )
        self._session_id = session["id"]
        self._turns = session.get("turns", 0)
        self.chatting = True
        self._announce_resume()
        self._apply_status_bar()

        return True

    def _answer_permission(self, value: str) -> None:
        approved = value.strip().lower() == "y"

        self.awaiting_permission = False
        label = self.query_one("#prompt-label", Static)
        label.update("")
        label.remove_class("active")

        callback = self._permission_callback
        self._permission_callback = None

        self.feed.write(
            Text(
                f"  -> {'approved' if approved else 'denied'}",
                style="bold green" if approved else "bold red",
            )
        )

        if callback is not None:
            callback(approved)

    # -------------------------------------------------------------
    # slash commands
    # -------------------------------------------------------------

    def handle_slash_command(self, name: str, arg: str) -> None:
        if name == "help":
            self.feed.write(Text(SLASH_HELP, style="#d0d5db"))
        elif name == "clear":
            self.feed.clear()
        elif name == "scout":
            self.mode = "scout"
        elif name == "operate":
            self.mode = "operate"
        elif name == "model":
            self._slash_model(arg)
        elif name == "tools":
            names = sorted(seacode_module.known_tool_names())
            self.feed.write(
                Text("Tools: " + ", ".join(names), style="#d0d5db")
            )
        elif name == "theme":
            self._slash_theme(arg)
        elif name == "sessions":
            self._slash_sessions()
        elif name == "undo":
            self._slash_undo(arg)
        else:
            self.feed.write(
                Text(
                    f"Unknown command /{name} (try /help).",
                    style="bold yellow",
                )
            )

    def _slash_theme(self, arg: str) -> None:
        if not arg:
            self.feed.write(
                Text(
                    f"Theme: {self._theme_name} "
                    f"(options: {', '.join(THEMES)})",
                    style="bold",
                )
            )
            return

        name = arg.strip().lower()

        if name not in THEMES:
            self.feed.write(
                Text(
                    f"Unknown theme {arg!r} "
                    f"(options: {', '.join(THEMES)}).",
                    style="bold yellow",
                )
            )
            return

        self._theme_name = name
        self._theme = THEMES[name]
        self._apply_theme()
        self.feed.write(
            Text(f"Switched theme to {name}", style="bold green")
        )

    def _slash_model(self, arg: str) -> None:
        if not arg:
            self.feed.write(
                Text(f"Model: {self._model}", style="bold")
            )
            return

        self._model = arg
        self._apply_mode_styling(self.mode)
        self._apply_status_bar()
        self.feed.write(
            Text(f"Switched model to {arg}", style="bold green")
        )

    def _slash_sessions(self) -> None:
        try:
            root = tools_module.get_project_root()
            entries = sessions_module.list_sessions(root)
        except Exception as e:
            self.feed.write(
                Text(f"Could not list sessions: {e}", style="bold red")
            )
            return

        if not entries:
            self.feed.write(Text("No saved sessions.", style="grey50"))
            return

        lines = ["Sessions (newest first):"]

        for entry in entries[:10]:
            label = sessions_module.turns_label(entry["turns"])
            lines.append(
                f"  {entry['id']}  {label}  "
                f"{entry['model']}"
            )
            if entry["preview"]:
                lines.append(f"    {entry['preview']}")

        self.feed.write(Text("\n".join(lines), style="#d0d5db"))

    def _slash_undo(self, arg: str) -> None:
        # Undo needs the permission bridge, which blocks its calling
        # thread -- so it runs in a worker, never on the UI thread.
        self.feed.write(Text("Restoring from backup...", style="grey50"))
        self._run_slash_undo(arg or None)

    @work(thread=True)
    def _run_slash_undo(self, path) -> None:
        try:
            result = tools_module.undo_last_change(path)
            style = "bold green"
        except Exception as e:
            result = f"Undo failed: {e}"
            style = "bold red"

        self.call_from_thread(self._note_slash_undo_result, result, style)

    def _note_slash_undo_result(self, result: str, style: str) -> None:
        self.feed.write(Text(result, style=style))

    # -------------------------------------------------------------
    # splash <-> chat transition
    # -------------------------------------------------------------

    def _tick_art(self) -> None:
        """Advance one splash animation frame: shimmer the dither rule,
        breathe the mascot color, and drift the sea strips. Content
        only -- never moves layout."""

        if not self.is_running:
            return

        self._art_phase += 1
        self.query_one("#splash-rule", Static).update(
            wave_rule(64, self._art_phase)
        )
        pulse = MASCOT_PULSE_BY_MODE.get(self.mode, MASCOT_PULSE)
        self.query_one("#mascot", Static).styles.color = pulse[
            self._art_phase % len(pulse)
        ]

        height = len(MASCOT.splitlines())

        for widget_id, seed, direction in (
            ("#sea-left", 0, 1),
            ("#sea-right", 3, -1),
        ):
            strip = self.query_one(widget_id, Static)
            strip.update(
                side_column(
                    max(strip.size.width, 1),
                    height,
                    self._art_phase,
                    seed,
                    direction,
                )
            )

    def _stop_art(self) -> None:
        if self._art_timer is not None:
            self._art_timer.stop()
            self._art_timer = None

    def on_unmount(self) -> None:
        self._stop_art()

    def watch_chatting(self, chatting: bool) -> None:
        self.query_one("#splash", Vertical).set_class(chatting, "hidden")
        self.query_one("#chat-body", Horizontal).set_class(
            not chatting, "hidden"
        )
        self.query_one("#feed", RichLog).set_class(not chatting, "hidden")

        if chatting:
            self._stop_art()
        self.query_one("#prompt-label", Static).set_class(
            not chatting, "hidden"
        )
        chat_input = self.query_one("#chat-input", Input)
        chat_input.set_class(not chatting, "hidden")

        if chatting:
            chat_input.focus()

        self._apply_status_bar()
        self._update_sidebar()

    def _update_sidebar(self, width=None) -> None:
        """Wide terminals (>=140 cols) get a context rail beside the
        transcript: project, git, session, and tool inventory. It
        hides again below that width or back on the splash."""

        try:
            sidebar = self.query_one("#sidebar", Static)
        except Exception:
            return

        if width is None:
            width = self.size.width

        show = self.chatting and width >= 140
        sidebar.set_class(not show, "hidden")

        if not show:
            return

        lines = [f"Project  {self._project_name}"]

        try:
            lines.append(
                f"Files    {len(tools_module.list_files('.'))}"
            )
        except Exception:
            pass

        try:
            status = tools_module.git_status()
            changed = (
                len(status.strip().splitlines()) if status.strip() else 0
            )
            lines.append(
                f"Git      {changed} changed" if changed
                else "Git      clean"
            )
        except Exception:
            pass

        if self._session_id:
            lines.append(f"Session  {self._session_id}")

        lines.append(f"Turns    {self._turns}")
        lines.append(
            f"Tools    {len(seacode_module.known_tool_names())}"
        )
        lines.append(f"Theme    {self._theme_name}")

        meter = context_meter(
            self.messages, context_module.get_context_limit()
        )

        if meter:
            lines.append(f"Context  {meter[4:]}")

        sidebar.update("\n".join(lines))

    # -------------------------------------------------------------
    # mode (Scout / Operate)
    # -------------------------------------------------------------

    def action_toggle_mode(self) -> None:
        self.mode = "operate" if self.mode == "scout" else "scout"

    def watch_mode(self, mode: str) -> None:
        self._apply_mode_styling(mode)
        self._apply_status_bar()

        if self.chatting:
            self.feed.write(
                Text(
                    f"\n-- switched to {self.MODE_LABELS[mode]} --",
                    style="bold magenta",
                )
            )

    def _apply_mode_styling(self, mode: str) -> None:
        chat_input = self.query_one("#chat-input", Input)
        chat_input.remove_class("mode-scout", "mode-operate")
        chat_input.add_class(f"mode-{mode}")

        status = self.query_one("#splash-status", Static)
        status.update(self._status_pills())

    PILL_BG = "#1a2332"

    def _status_pills(self) -> Text:
        pills = Text()

        for text, color in (
            (self._model, "#7dd3fc"),
            (self.MODE_LABELS[self.mode], self.MODE_COLORS[self.mode]),
            (self._project_name, "#d0d5db"),
        ):
            pills.append("  ")
            pills.append(
                f" {text} ", style=f"bold {color} on {self.PILL_BG}"
            )

        pills.append("  ")

        return pills

    def _apply_status_bar(self) -> None:
        # On the splash, the pills already show model/mode/project --
        # the status bar stays out of the way (version only) until a
        # chat starts and the pills are gone.
        if not self.chatting:
            self.query_one("#status-left", Static).update("")
            self.query_one("#status-right", Static).update(
                Text(f"v{VERSION}", style="#7a8494")
            )
            return

        left = self._project_name

        if self._session_id is not None:
            left += f"  {self._session_id}"

        if self.turn_running:
            left += f"   {WORKING_INDICATOR}"

        self.query_one("#status-left", Static).update(left)

        right = Text()
        right.append(self._model, style="#7dd3fc")
        right.append("   ")
        right.append(
            self.MODE_LABELS[self.mode],
            style=self.MODE_COLORS[self.mode],
        )

        try:
            trust = permissions_module.get_trust_level()
        except ValueError:
            trust = "ask"

        if trust != "ask":
            right.append(f"   trust:{trust}", style="#7a8494")

        meter = context_meter(
            self.messages, context_module.get_context_limit()
        )

        if meter:
            right.append(f"   {meter}", style="#7a8494")

        right.append(f"   v{VERSION}", style="#7a8494")
        self.query_one("#status-right", Static).update(right)

    def note_scouted_action(self, action: str) -> None:
        self.feed.write(
            Text(
                f"  (skipped -- Scout mode is read-only: {action})",
                style="bold yellow",
            )
        )

    def note_auto_approved_action(self, action: str) -> None:
        self.feed.write(
            Text(
                f"  (auto-approved -- "
                f"trust={permissions_module.get_trust_level()}: "
                f"{action})",
                style="grey50",
            )
        )

    def handle_stream_line(self, line: str) -> None:
        self.feed.write(Text(line, style="grey50"))

    def submit_user_message(self, text: str) -> None:
        self.turn_running = True
        self.messages.append({"role": "user", "content": text})
        self._turns += 1
        self.feed.write(Text(f"\n❯ {text}", style="bold cyan"))
        self.persist_session()
        self.run_agent_turn(text)

    # -------------------------------------------------------------
    # agent turn (background thread)
    # -------------------------------------------------------------

    @work(thread=True, exclusive=True)
    def run_agent_turn(self, user_text: str) -> None:
        bridge = _ThreadedStdoutBridge(self)
        tokens = _TokenLineBuffer(self)
        started = time.monotonic()

        def on_token(delta: str) -> None:
            tokens.push(delta)

        try:
            with contextlib.redirect_stdout(bridge):
                answer = seacode_module.run_agent(
                    self.messages,
                    model=self._model,
                    max_tool_calls=self._max_tool_calls,
                    stream=self._stream,
                    on_token=on_token if self._stream else None,
                )
        except Exception as e:
            bridge.drain()
            tokens.drain()
            self.call_from_thread(self.on_turn_complete, None, str(e))
            return

        bridge.drain()
        tokens.drain()
        elapsed = seacode_module.format_duration(
            time.monotonic() - started
        )
        self.call_from_thread(self.on_turn_complete, answer, None, elapsed)

    def persist_session(self) -> None:
        try:
            self._session_id = sessions_module.save_session(
                tools_module.get_project_root(),
                self.messages,
                self._model,
                self._session_id,
                self._turns,
            )
        except Exception as e:
            self.feed.write(
                Text(
                    f"\nWarning: could not save session: {e}",
                    style="bold yellow",
                )
            )
            return

        self._apply_status_bar()

    def on_turn_complete(
        self, answer, error: str | None, elapsed: str = ""
    ) -> None:
        self.turn_running = False

        if error is not None:
            self.feed.write(Text(f"\n[error] {error}", style="bold red"))
        elif answer is None:
            self.feed.write(
                Text(
                    "\nSeaCode stopped without a final answer "
                    "(see feed above for why).",
                    style="bold red",
                )
            )
        else:
            label = Text("\n● SeaCode", style="bold green")

            if elapsed:
                label.append(f" ({elapsed})", style="grey50")

            self.feed.write(label)
            self.write_answer(answer)

        self.persist_session()
        self._update_sidebar()
        self.query_one("#chat-input", Input).focus()

    def write_answer(self, answer: str) -> None:
        """Render a final answer as Markdown, falling back to plain
        text if it can't be parsed."""

        try:
            self.feed.write(Markdown(answer))
        except Exception:
            self.feed.write(Text(answer, style="#d0d5db"))

    # -------------------------------------------------------------
    # stdout -> UI
    # -------------------------------------------------------------

    def handle_agent_line(self, line: str) -> None:
        if self._should_suppress(line):
            return

        style_key = classify_line(line)

        # Tool calls open a visual block: a blank separator plus a
        # left rule so a tool call, its result, and its diff read as
        # one unit when scrolling back.
        if style_key == "tool":
            self.feed.write(Text(""))
            self.feed.write(
                Text(f"▎{line}", style=STYLE_MAP[style_key])
            )
        else:
            self.feed.write(Text(line, style=STYLE_MAP[style_key]))

        self.maybe_show_diff(line)

    def _should_suppress(self, line: str) -> bool:
        """Swallow seacode.py's `Model raw response:` debug dump.

        That print() emits a header line followed by the model's raw
        (often JSON) reply, which can span several lines with no
        internal blank line. We hide the whole block by suppressing
        everything from the header up to -- but not including -- the
        next blank line, which is seacode.py's own separator before its
        next print() call.
        """

        stripped = line.strip()

        if stripped == "Model raw response:":
            self._suppressing_raw_response = True
            return True

        if self._suppressing_raw_response:
            if stripped == "":
                self._suppressing_raw_response = False
                return False
            return True

        return False

    # -------------------------------------------------------------
    # diff panel
    # -------------------------------------------------------------

    def maybe_show_diff(self, line: str) -> None:
        match = _MODIFICATION_SUCCESS_RE.match(line.strip())

        if not match:
            return

        path = match.group(2)
        diff_text = self._compute_diff(path)

        if diff_text is not None:
            self.show_diff(path, diff_text)

    def _compute_diff(self, path: str):
        try:
            project_root = tools_module.get_project_root()
            backup_dir = (
                project_root / tools_module.BACKUP_DIR_NAME / path
            )

            if not backup_dir.exists():
                return None

            backups = sorted(backup_dir.glob("*.bak"))

            if not backups:
                return None

            pre_content = backups[-1].read_text(encoding="utf-8")

            current_path = project_root / path
            post_content = (
                current_path.read_text(encoding="utf-8")
                if current_path.exists()
                else ""
            )

            diff_lines = difflib.unified_diff(
                pre_content.splitlines(keepends=True),
                post_content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )

            diff_text = "".join(diff_lines)

            return diff_text or None

        except Exception:
            return None

    # Most edits produce a handful of diff lines; beyond this the
    # inline render truncates with a note instead of flooding the feed.
    MAX_INLINE_DIFF_LINES = 40

    def show_diff(self, path: str, diff_text: str) -> None:
        lines = diff_text.splitlines()
        added = sum(
            1 for line in lines
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1 for line in lines
            if line.startswith("-") and not line.startswith("---")
        )

        self.feed.write(
            Text(
                f"  ─ {path} (+{added}/-{removed})",
                style="grey50",
            )
        )

        self._write_diff_lines(diff_text)

        self.feed.write(
            Text(f"  ↩ /undo {path} to revert", style="grey50")
        )

    def _write_diff_lines(self, diff_text: str) -> None:
        lines = diff_text.splitlines()
        shown = "\n".join(lines[: self.MAX_INLINE_DIFF_LINES])

        try:
            from rich.syntax import Syntax

            self.feed.write(Syntax(shown, "diff", word_wrap=False))
        except Exception:
            for line in shown.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    style = "#4caf78"
                elif line.startswith("-") and not line.startswith("---"):
                    style = "#e06c75"
                elif line.startswith("@@"):
                    style = "#7dd3fc"
                else:
                    style = "grey50"

                self.feed.write(Text(f"  {line}", style=style))

        hidden = len(lines) - self.MAX_INLINE_DIFF_LINES

        if hidden > 0:
            self.feed.write(
                Text(
                    f"  … and {hidden} more lines",
                    style="grey50",
                )
            )

    def show_permission_diff(self, diff: str) -> None:
        """Render a proposed change in the feed before approval, so the
        y/N answer is about the actual edit, not just a filename."""

        self.feed.write(Text("  Proposed change:", style="grey50"))
        self._write_diff_lines(diff)

    # -------------------------------------------------------------
    # permission prompt
    # -------------------------------------------------------------

    def show_permission_prompt(self, action: str, callback, diff=None) -> None:
        self._permission_callback = callback
        self.awaiting_permission = True
        self.query_one("#prompt-label", Static).add_class("active")

        if diff:
            self.show_permission_diff(diff)

        stats = ""

        if diff:
            added = sum(
                1 for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
            removed = sum(
                1 for line in diff.splitlines()
                if line.startswith("-") and not line.startswith("---")
            )
            stats = f" (+{added}/-{removed})"

        self.query_one("#prompt-label", Static).update(
            f"SeaCode needs permission to: {action}{stats}   [y/N]:"
        )

        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""
        input_widget.focus()

    def watch_turn_running(self, running: bool) -> None:
        if not self.awaiting_permission:
            self.query_one("#chat-input", Input).disabled = running
        self._apply_status_bar()

    def watch_awaiting_permission(self, awaiting: bool) -> None:
        input_widget = self.query_one("#chat-input", Input)
        input_widget.disabled = self.turn_running and not awaiting

        if awaiting:
            input_widget.placeholder = "y/N"
        else:
            input_widget.placeholder = "Ask SeaCode to do something..."


def main(
    model=None,
    max_tool_calls=None,
    stream=True,
    initial_messages=None,
    session_id=None,
    initial_turns=0,
    animate=True,
    theme=None,
) -> None:
    SeaCodeApp(
        model=model,
        max_tool_calls=max_tool_calls,
        stream=stream,
        initial_messages=initial_messages,
        session_id=session_id,
        initial_turns=initial_turns,
        animate=animate,
        theme=theme,
    ).run()


if __name__ == "__main__":
    main()
