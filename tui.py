"""
Iggy TUI -- a Claude-Code-style terminal interface for the agent.

Design choice (per project discussion): agent.py's run_agent() keeps
its existing print()-based logging exactly as-is -- no refactor to
structured events. This app runs run_agent() in a background thread,
redirects stdout into itself so every line lands in the live feed as
it's produced, and layers a diff view + a simple y/N permission prompt
on top, entirely from the outside.

Two things needed a small bridge to make that work inside a real
terminal app rather than a plain REPL:

  - stdout: Python's sys.stdout is one global, so _ThreadedStdoutBridge
    temporarily takes it over for the duration of a single agent turn
    (run from a worker thread) and forwards completed lines back to the
    UI thread via `call_from_thread`. This assumes only one agent turn
    is ever in flight at a time, which the UI already enforces by
    disabling input while a turn is running.

  - permissions: agent/tools.py calls agent.permissions.request_permission,
    which normally blocks on a raw input(). Textual owns the terminal,
    so that call is monkeypatched (on mount) to _PermissionBridge.request,
    which hands control to the UI thread, blocks the worker thread on a
    threading.Event, and resumes once the person answers in the prompt
    line at the bottom of the screen.

Run with:  python3 tui.py
Requires:  pip install textual
"""

import contextlib
import difflib
import re
import threading

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

import agent.agent as agent_module
import agent.tools as tools_module


# -----------------------------------------------------------------------
# stdout bridge
# -----------------------------------------------------------------------


class _ThreadedStdoutBridge:
    """Stand-in for sys.stdout while an agent turn runs in a worker
    thread. Buffers partial writes into whole lines and hands each
    finished line to the App on the UI thread.
    """

    def __init__(self, app: "IggyApp"):
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
    """Replaces agent.tools.request_permission for the duration of the
    TUI session. Called from the worker thread; blocks it on a
    threading.Event until the person answers in the UI thread.
    """

    def __init__(self, app: "IggyApp"):
        self._app = app

    def request(self, action: str) -> bool:
        event = threading.Event()
        result = {"approved": False}

        def on_answer(approved: bool) -> None:
            result["approved"] = approved
            event.set()

        self._app.call_from_thread(
            self._app.show_permission_prompt, action, on_answer
        )
        event.wait()

        return result["approved"]


# -----------------------------------------------------------------------
# line classification -- lightweight, read-only parsing of agent.py's
# existing print() output. Nothing here changes agent.py's behavior.
# -----------------------------------------------------------------------

_MODIFICATION_SUCCESS_RE = re.compile(
    r"^(Replacement applied successfully|File written successfully): (.+)$"
)
_REVERT_RE = re.compile(
    r"^Verification failed for (?:newly created )?.+?; "
    r"(?:reverted to its pre-edit content|removed it)\.$"
)
_BLOCKED_RE = re.compile(r"^Blocking (modification|write_file):")
_REJECTED_RE = re.compile(r"^Rejecting (premature final answer|final answer):")
_TOOL_REQUESTED_RE = re.compile(r"^Tool requested: ")
_STOPPED_RE = re.compile(r"^Agent stopped after")

STYLE_MAP = {
    "error": "bold red",
    "warning": "bold yellow",
    "success": "bold green",
    "tool": "bold cyan",
    "dim": "grey50",
    "default": "white",
}


def classify_line(line: str) -> str:
    """Return a style key (see STYLE_MAP) for a line of stdout output."""

    stripped = line.strip()

    if not stripped:
        return "dim"
    if _REVERT_RE.match(stripped) or _STOPPED_RE.match(stripped):
        return "error"
    if _BLOCKED_RE.match(stripped) or _REJECTED_RE.match(stripped):
        return "warning"
    if _MODIFICATION_SUCCESS_RE.match(stripped):
        return "success"
    if _TOOL_REQUESTED_RE.match(stripped) or stripped.startswith("Arguments:"):
        return "tool"
    if stripped.startswith("Model raw response:"):
        return "dim"
    return "default"


class IggyApp(App):
    """The Iggy TUI."""

    CSS = """
    #main {
        height: 1fr;
    }
    #feed {
        width: 3fr;
        border: solid $accent;
        border-title-color: $accent;
    }
    #diff-log {
        width: 2fr;
        border: solid $secondary;
        border-title-color: $secondary;
    }
    #prompt-label {
        height: auto;
        min-height: 0;
        padding: 0 1;
        color: $warning;
        text-style: bold;
    }
    #chat-input {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    turn_running: reactive[bool] = reactive(False)
    awaiting_permission: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.title = "Iggy"
        self.sub_title = f"model: {agent_module.MODEL}"
        self.messages = [
            {"role": "system", "content": agent_module.SYSTEM_PROMPT}
        ]
        self._permission_bridge = _PermissionBridge(self)
        self._permission_callback = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield RichLog(id="feed", wrap=True, highlight=False, markup=False)
            yield RichLog(
                id="diff-log", wrap=False, highlight=False, markup=False
            )
        yield Static("", id="prompt-label")
        yield Input(
            placeholder="Ask Iggy to do something...", id="chat-input"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#feed", RichLog).border_title = "Live feed"
        self.query_one("#diff-log", RichLog).border_title = "Last diff"

        # Route agent/tools.py's permission checks through the TUI
        # instead of the raw input() prompt agent.permissions uses by
        # default. Only rebinds the name in the tools module, exactly
        # like the test suite's own patch.object(tools_module, ...).
        tools_module.request_permission = self._permission_bridge.request

        self.feed.write(
            Text(
                "Iggy is ready. Type a message below and press Enter.",
                style="bold",
            )
        )
        self.query_one("#chat-input", Input).focus()

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

        self.submit_user_message(value)

    def _answer_permission(self, value: str) -> None:
        approved = value.strip().lower() == "y"

        self.awaiting_permission = False
        self.query_one("#prompt-label", Static).update("")

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

    def submit_user_message(self, text: str) -> None:
        self.turn_running = True
        self.feed.write(Text(f"\nYou: {text}", style="bold cyan"))
        self.run_agent_turn(text)

    # -------------------------------------------------------------
    # agent turn (background thread)
    # -------------------------------------------------------------

    @work(thread=True, exclusive=True)
    def run_agent_turn(self, user_text: str) -> None:
        self.messages.append({"role": "user", "content": user_text})

        bridge = _ThreadedStdoutBridge(self)

        try:
            with contextlib.redirect_stdout(bridge):
                answer = agent_module.run_agent(self.messages)
        except Exception as e:
            bridge.drain()
            self.call_from_thread(self.on_turn_complete, None, str(e))
            return

        bridge.drain()
        self.call_from_thread(self.on_turn_complete, answer, None)

    def on_turn_complete(self, answer, error: str | None) -> None:
        self.turn_running = False

        if error is not None:
            self.feed.write(Text(f"\n[error] {error}", style="bold red"))
        elif answer is None:
            self.feed.write(
                Text(
                    "\nIggy stopped without a final answer "
                    "(see feed above for why).",
                    style="bold red",
                )
            )
        else:
            self.feed.write(Text(f"\nIggy: {answer}", style="bold green"))

        self.query_one("#chat-input", Input).focus()

    # -------------------------------------------------------------
    # stdout -> UI
    # -------------------------------------------------------------

    def handle_agent_line(self, line: str) -> None:
        style_key = classify_line(line)
        self.feed.write(Text(line, style=STYLE_MAP[style_key]))
        self.maybe_show_diff(line)

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

    def show_diff(self, path: str, diff_text: str) -> None:
        diff_log = self.query_one("#diff-log", RichLog)
        diff_log.clear()
        diff_log.write(Text(path, style="bold"))
        diff_log.write(Text(""))

        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                style = "green"
            elif line.startswith("-") and not line.startswith("---"):
                style = "red"
            elif line.startswith("@@"):
                style = "cyan"
            else:
                style = "grey50"

            diff_log.write(Text(line, style=style))

    # -------------------------------------------------------------
    # permission prompt
    # -------------------------------------------------------------

    def show_permission_prompt(self, action: str, callback) -> None:
        self._permission_callback = callback
        self.awaiting_permission = True

        self.query_one("#prompt-label", Static).update(
            f"Iggy needs permission to: {action}   [y/N]:"
        )

        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""
        input_widget.focus()

    def watch_turn_running(self, running: bool) -> None:
        if not self.awaiting_permission:
            self.query_one("#chat-input", Input).disabled = running

    def watch_awaiting_permission(self, awaiting: bool) -> None:
        input_widget = self.query_one("#chat-input", Input)
        input_widget.disabled = self.turn_running and not awaiting

        if awaiting:
            input_widget.placeholder = "y/N"
        else:
            input_widget.placeholder = "Ask Iggy to do something..."


if __name__ == "__main__":
    IggyApp().run()
