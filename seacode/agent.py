import itertools
import json
import os
import sys
import threading
import time

import ollama
from rich.console import Console
from rich.markdown import Markdown

from seacode import __version__
from seacode.context import (
    build_system_prompt,
    fresh_messages,
    get_context_limit,
    trim_messages,
)
from seacode.permissions import get_trust_level
from seacode.providers import TOOL_SCHEMAS
from seacode.dynamic_tools import (
    dynamic_tool_names,
    dynamic_tool_schemas,
    get_dynamic_handler,
)
from seacode.hooks import run_post_tool_hooks, run_pre_tool_hooks
from seacode.sessions import save_session
from seacode.tools import (
    TOOLS,
    get_project_root,
    remove_created_file,
    restore_file_content,
)
# The M7 verification checks live in seacode.verify; they are imported
# here (and re-exported) so existing callers/tests using
# `seacode.agent.<check>` keep working.
from seacode.verify import (
    CHANGE_PAIR_PATTERNS,
    FILENAME_PATTERN,
    GENERIC_PRESERVATION_THRESHOLD,
    MIN_LENGTH_FOR_GENERIC_DESTRUCTIVE_CHECK,
    MODIFICATION_TOOLS,
    REWRITE_INTENT_KEYWORDS,
    SCOPE_LIMIT_KEYWORDS,
    TARGETED_EDIT_PRESERVATION_THRESHOLD,
    VERIFICATION_CLAIM_KEYWORDS,
    VERIFICATION_TOOLS,
    check_incomplete_multi_target_edit,
    check_scope_creep,
    check_unnecessary_full_rewrite,
    check_unread_file_edit,
    claims_verification,
    classify_modification_result,
    content_preserved_ratio,
    extract_change_pair,
    extract_mentioned_filenames,
    mentions_full_rewrite_intent,
    safe_read_file,
    verify_python_syntax,
    verify_semantic_edit,
    verify_target_file,
)


DEFAULT_MODEL = "qwen2.5-coder:3b"
DEFAULT_HOST = "http://127.0.0.1:11434"
MAX_TOOL_CALLS = 10

# Tool budget for a delegated subagent (the `task` tool): deliberately
# smaller than a top-level run so a subtask stays a subtask.
MAX_SUBAGENT_TOOL_CALLS = 5


def config_value(config, key):
    """Read one key from a loaded seacode.toml dict (None-safe)."""

    if not isinstance(config, dict):
        return None

    return config.get(key)


def resolve_model(cli_value=None, config=None) -> str:
    """Return the model to use: CLI flag wins, then SEACODE_MODEL, then
    seacode.toml, then the built-in default."""

    if cli_value:
        return cli_value

    env_value = os.environ.get("SEACODE_MODEL")

    if env_value:
        return env_value

    config_value_model = config_value(config, "model")

    if config_value_model:
        return config_value_model

    return DEFAULT_MODEL


def resolve_host(cli_value=None, config=None) -> str:
    """Return the Ollama host to use: CLI flag wins, then SEACODE_HOST /
    OLLAMA_HOST, then seacode.toml, then the local default."""

    if cli_value:
        return cli_value

    env_value = os.environ.get("SEACODE_HOST") or os.environ.get("OLLAMA_HOST")

    if env_value:
        return env_value

    config_host = config_value(config, "host")

    if config_host:
        return config_host

    return DEFAULT_HOST


def coerce_max_tool_calls(value):
    """Best-effort int for a limit from env/config; None if unusable."""

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None

    return None


def resolve_max_tool_calls(cli_value=None, config=None) -> int:
    """Return the per-run tool-call limit: CLI flag wins, then
    SEACODE_MAX_TOOLS, then seacode.toml, then the built-in default.
    Unparseable values fall back to the next source rather than
    crashing the run."""

    if cli_value is not None:
        return cli_value

    from_env = coerce_max_tool_calls(os.environ.get("SEACODE_MAX_TOOLS"))

    if from_env is not None:
        return from_env

    from_config = coerce_max_tool_calls(config_value(config, "max_tools"))

    if from_config is not None:
        return from_config

    return MAX_TOOL_CALLS

# How many times the agent is forced to keep trying after a genuine
# modification failure before it is allowed to give up. Bounds the
# recovery loop so it can't run forever.
MAX_RECOVERY_ATTEMPTS = 2

# How many consecutive malformed responses (invalid JSON, or a
# hallucinated/unknown tool name) the agent gets to correct itself
# before the run is aborted (M7.5). This is a different failure class
# from MAX_RECOVERY_ATTEMPTS: no real tool was ever executed here.
MAX_INVALID_ACTION_ATTEMPTS = 3

# How many times the agent is forced to retry after a modification that
# *succeeded* but failed semantic verification (M7.6) -- e.g. it wrote
# the requested text but also destroyed unrelated parts of the file.
# A different failure class again: the tool call itself worked fine.
MAX_SEMANTIC_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after a modification that
# *succeeded*, and passed semantic verification, but appears to have
# landed on the wrong file entirely (M7.7) -- e.g. the request named
# `utils.py` but the edit was made to `helpers/utils.py`, or the
# project has two files with that name and it's not clear which one
# was meant. Yet another distinct failure class: the edit itself may be
# perfectly correct, just applied to the wrong target.
MAX_WRONG_FILE_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after claiming, in its
# final answer, that a change was "verified" / "tested" / "confirmed"
# without actually calling a tool capable of checking that since the
# modification (M7.8). This is orthogonal to whether the modification
# itself was correct -- M7.6/M7.7 already gate that. This catches the
# agent asserting a verification step happened when it didn't, which
# the system prompt explicitly forbids ("Never claim that a change was
# verified unless you actually performed verification").
MAX_VERIFICATION_CLAIM_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after a modification or
# file creation that succeeded, passed semantic and target-file
# verification, but left a .py file that doesn't parse as valid Python
# (M7.10). Checked regardless of whether the file already had a syntax
# error before the edit -- nothing SeaCode touches or creates should exit
# a run non-parseable, existing problems included.
MAX_SYNTAX_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after trying to give a
# final answer while the request named multiple files and at least one
# of them hasn't received a successful, verified modification yet
# (M7.11). Recomputed live at each final-answer attempt rather than
# tracked as a "pending" flag like M7.6/M7.7/M7.10 -- the underlying
# signal (which files have actually been modified so far) is cheap to
# recheck and naturally resolves itself as the agent does more work, so
# there's no separate failure state to remember between turns.
MAX_INCOMPLETE_MULTI_TARGET_RETRY_ATTEMPTS = 2

# How many times the agent is forced to retry after modifying a file
# beyond what an explicitly scope-limited request named (M7.12). Like
# M7.11, recomputed live rather than tracked as a pending flag.
MAX_SCOPE_CREEP_RETRY_ATTEMPTS = 2

# The M7 keyword lists / thresholds / filename patterns that used to
# live here now live in seacode.verify (imported above).

# Import-time defaults, honoring the environment. The CLI overrides
# these per-run (see resolve_* and run_agent's parameters); tests
# patch `client` wholesale.
MODEL = resolve_model()
client = ollama.Client(host=resolve_host())


SYSTEM_PROMPT = """
You are a local computer engineering agent.

You have access to tools and are expected to ACT on tasks when the user asks you to modify the project.

When you need to act, call one of the available tools, with complete
arguments. Independent calls may go out together in one block; they
run in order.

When you are completely finished and no more tools are needed, answer
in plain prose. Never emit a tool call and a final answer together:
either keep working, or answer.

IMPORTANT TASK RULES:

1. If the user asks you to INSPECT, CREATE, MODIFY, CHANGE, FIX, REMOVE, REFACTOR, IMPLEMENT, UPDATE, or otherwise alter something in the project, this is an ACTION REQUEST.

2. For an ACTION REQUEST:
   - Do not simply tell the user how to do it.
   - Do not give your final answer before performing the requested work.
   - Inspect the relevant project files first.
   - Determine what needs to change.
   - Use the appropriate modification tool.
   - Continue working until the requested task is completed or you cannot safely complete it.

3. If the user asks for information about the project:
   - Use tools when necessary to inspect the project.
   - Answer only after you have enough information.

4. After modifying something, verify the result when possible.

5. Never claim that a change was completed unless the modification tool actually succeeded.

6. Never claim that a change was verified unless you actually performed verification.

Before answering, ask yourself:

1. Is this an action request or an information request?
2. If it is an action request, what files do I need to inspect?
3. Have I inspected the relevant code?
4. What is the smallest safe change that completes the task?
5. Which modification tool should I use?
6. Has the modification actually succeeded?
7. Can I verify the result?
8. Only after completing the task should I give my final answer.

TOOL SELECTION:

- Use list_files when you need to discover the project structure.
- Use search_files when you need to find specific code, variables, functions, classes, imports, or text.
- Use read_file when you need to understand the contents of a file.
- Use replace_in_file when making a targeted change to existing code.
- Prefer replace_in_file over write_file when modifying an existing file.
- Use write_file when creating a new file or when replacing an entire file is genuinely necessary.
- Use delete_file when the user asks to remove a file. Read it first; the content is backed up automatically.
- Use undo_last_change to revert your own most recent modification when verification shows it was wrong.
- Use run_command when a command is needed for inspection or verification.
- Use git_status when the user asks about Git status, modified files, untracked files, staged files, or changes in the working tree.
- Use task when a self-contained subtask (exploring one area, answering one research question) can be delegated to a subagent; give it complete instructions, since it cannot see this conversation. Additional tools whose names start with mcp_ come from user-configured MCP servers; use them as their descriptions indicate.

TOOL DETAILS:

- list_files(root): lists relevant files inside a project directory.
- read_file(path): reads a text file inside the project directory.
- search_files(query): searches project files for matching text and returns file names, line numbers, and matching lines.
- git_status(): returns the current Git working tree status.
- run_command(command): runs an approved command inside the project.
- write_file(path, content): writes or replaces a file inside the project after requesting user permission.
- replace_in_file(path, old_text, new_text): replaces one exact piece of existing text inside a project file after requesting user permission.
- delete_file(path): deletes a file inside the project after requesting user permission. The content is backed up first and can be recovered.
- undo_last_change(path): restores a file from its most recent backup after requesting user permission. Omit path to undo the most recent backup of any file.

SAFETY AND MODIFICATION RULES:

- Never access files outside the project directory.
- Never invent directory names or paths.
- Tool arguments must refer to real paths or values supported by the tools.
- Before modifying an existing file, read it first unless its relevant contents are already available.
- Do not modify files blindly.
- Make the smallest change necessary to complete the user's request.
- Preserve existing code and behavior unless the user explicitly asks for a behavior change.
- When using replace_in_file, old_text must match exactly one occurrence.
- If the exact text does not exist, do not invent a replacement.
- If the exact text occurs more than once, do not make the change.
- If a modification is denied, do not pretend that it succeeded.
- If a tool returns an error, treat the operation as unsuccessful.
- Do not invent tools.
- Do not paste large tool results back into your answer; summarize instead.
- After receiving a tool result, decide what to do next.

SEARCH RULES:

- For search_files, use short, exact keywords rather than descriptive phrases.
- For example, search for "ollama", not "Ollama usage".
- For list_files, use "." for the project root unless you have a specific directory name from a previous tool result.
- Never invent paths.

COMPLETION RULE:

For an action request, do not stop simply because you have figured out what should be changed.

The task is only complete after:
- the requested modification was successfully applied, OR
- the agent has determined that it cannot safely complete the task.

When possible, perform a verification step after the modification.

Distinguish clearly between:
- planned
- modified
- verified

Only report what actually happened.
"""


TASK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "task",
        "description": (
            "Delegate a self-contained subtask to a subagent with its "
            "own conversation and tool budget, e.g. exploring a part "
            "of the project or researching one question. Returns the "
            "subagent's final summary. Subagents cannot spawn "
            "subagents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Complete, self-contained instructions for "
                        "the subagent (it cannot see this conversation)."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
}

# Per-thread agent state: subagent nesting depth (max 1) and the
# model the current run is using, so a `task` call can spawn its
# subagent on the same model.
_agent_state = threading.local()


def subagent_depth() -> int:
    return getattr(_agent_state, "depth", 0)


def known_tool_names() -> set:
    """All tool names the model may call this run: built-ins, dynamic
    tools (MCP etc.), and `task` (except inside a subagent)."""

    names = set(TOOLS) | set(dynamic_tool_names())

    if subagent_depth() == 0:
        names.add("task")

    return names


def effective_tool_schemas() -> list:
    """Schemas sent to the provider: built-ins + dynamic + task."""

    schemas = list(TOOL_SCHEMAS) + dynamic_tool_schemas()

    if subagent_depth() == 0:
        schemas.append(TASK_TOOL_SCHEMA)

    return schemas


def execute_tool(tool_name: str, arguments: dict):
    """Execute a registered tool: built-in, `task`, or dynamic."""

    if tool_name in TOOLS:
        return TOOLS[tool_name](**arguments)

    if tool_name == "task":
        return _run_subagent_task(arguments)

    handler = get_dynamic_handler(tool_name)

    if handler is not None:
        return handler(**arguments)

    raise ValueError(f"Unknown tool: {tool_name}")


def _run_subagent_task(arguments: dict):
    """Run a nested agent for a delegated subtask (depth-limited)."""

    if subagent_depth() >= 1:
        return "Error: subagents cannot spawn further subagents."

    if not isinstance(arguments, dict):
        return "Error: task requires a prompt."

    prompt = arguments.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        return "Error: task requires a non-empty prompt."

    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + "\n\nYou are a subagent handling one delegated "
                "subtask. Work independently, then give a concise "
                "final summary of what you found or did, including "
                "file paths. Do not ask follow-up questions."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    _agent_state.depth = subagent_depth() + 1

    try:
        answer = run_agent(
            messages,
            model=getattr(_agent_state, "model", None),
            max_tool_calls=MAX_SUBAGENT_TOOL_CALLS,
        )
    finally:
        _agent_state.depth -= 1

    if answer is None:
        return (
            "Subagent stopped without a final answer "
            "(tool limit or unrecoverable error)."
        )

    return answer


def clean_model_response(raw_response: str) -> str:
    """Remove Markdown code fences from the model response."""

    clean_response = raw_response.strip()

    if clean_response.startswith("```"):
        lines = clean_response.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        clean_response = "\n".join(lines).strip()

    return clean_response


def parse_action(raw_response: str) -> dict:
    """Convert the model's response into a tool/action dictionary."""

    clean_response = clean_model_response(raw_response)

    action = json.loads(clean_response)

    if not isinstance(action, dict):
        raise ValueError("Model response must be a JSON object.")

    return action


def extract_chunk_content(chunk) -> str:
    """Pull the text delta out of one streamed chat chunk.

    Accepts both the ollama client's object form (chunk.message.content)
    and the plain-dict form, returning "" when a chunk carries no text
    (e.g. the final done chunk).
    """

    message = getattr(chunk, "message", None)

    if message is None and isinstance(chunk, dict):
        message = chunk.get("message")

    if message is None:
        return ""

    content = getattr(message, "content", None)

    if content is None and isinstance(message, dict):
        content = message.get("content")

    return content or ""


def _message_field(message, name, default=None):
    """Read object-or-dict message fields (ollama models, plain dicts,
    and test doubles all flow through here)."""

    value = getattr(message, name, None)

    if value is None and isinstance(message, dict):
        value = message.get(name, default)

    return default if value is None else value


def normalize_tool_call(call):
    """(name, arguments, raw) from one native tool call, object- or
    dict-form. arguments is a dict, or None when present but not
    parseable (raw keeps the original for the error message)."""

    function = _message_field(call, "function", {}) or {}
    name = _message_field(function, "name", "") or ""
    args = _message_field(function, "arguments", None)
    raw = ""

    if isinstance(args, str):
        raw = args

        try:
            args = json.loads(args) if args else {}
        except ValueError:
            args = None
    elif args is not None and not isinstance(args, dict):
        raw = json.dumps(args)
        args = None

    if args is None and not raw:
        args = {}

    return name, args, raw


def extract_chunk_tool_calls(chunk):
    """Tool calls from one streamed chunk (usually just the final one
    carries them)."""

    message = getattr(chunk, "message", None)

    if message is None and isinstance(chunk, dict):
        message = chunk.get("message")

    if message is None:
        return []

    raw_calls = _message_field(message, "tool_calls", None) or []

    return [normalize_tool_call(call) for call in raw_calls]


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


def normalize_tool_name(tool_name):
    """Normalize tool names returned by the model."""

    if not isinstance(tool_name, str):
        return tool_name

    if "(" in tool_name:
        tool_name = tool_name.split("(", 1)[0].strip()

    return tool_name


def run_agent(
    messages: list[dict],
    model=None,
    max_tool_calls=None,
    stream=False,
    on_token=None,
    show_thinking=False,
    verbose=False,
    max_context_tokens=None,
):
    """Run the agent until it produces a final answer.

    model and max_tool_calls override the import-time defaults
    (MODEL / MAX_TOOL_CALLS) for this run only; the CLI uses this
    to apply --model / --max-tools without touching globals.

    With stream=True, model replies are read token by token, each
    text delta passed to on_token as it arrives. With
    show_thinking=True (and no on_token to show progress), a transient
    spinner line covers the wait instead. The raw-response dump only
    appears with verbose=True (--verbose); both frontends leave it
    off since they already show the response live.
    """

    model = model or MODEL
    tool_call_limit = max_tool_calls or MAX_TOOL_CALLS

    # Remember the model for any `task` subagent spawned in this run.
    _agent_state.model = model

    tool_calls = 0

    # M7.3 recovery state: tracks whether the most recent modification
    # attempt failed for a genuine (recoverable) reason, and how many
    # times we've forced the agent to keep trying instead of quitting.
    pending_recovery = False
    recovery_attempts = 0

    # M7.5 recovery state: tracks consecutive malformed responses
    # (invalid JSON or an unknown/hallucinated tool name), so the
    # agent can be corrected instead of the run silently dying.
    invalid_action_attempts = 0

    # M7.6 recovery state: tracks whether the most recent modification
    # *succeeded* as a tool call but failed semantic verification (wrong
    # and/or destructive result), and how many times we've forced a
    # retry instead of letting the agent claim success.
    pending_semantic_recovery = False
    semantic_recovery_attempts = 0
    last_semantic_failure_message = None

    # M7.7 recovery state: tracks whether the most recent modification
    # succeeded and passed semantic verification, but appears to have
    # landed on the wrong file (name mismatch or unresolved ambiguity
    # against the user's request), and how many times we've forced a
    # retry on the correct file instead of letting the agent claim
    # success.
    pending_wrong_file_recovery = False
    wrong_file_recovery_attempts = 0
    last_wrong_file_failure_message = None

    # M7.10 recovery state: tracks whether the most recent modification
    # or file creation succeeded, passed semantic and target-file
    # verification, but left a .py file that doesn't parse, and how
    # many times we've forced a retry instead of letting the agent
    # claim success with broken syntax.
    pending_syntax_recovery = False
    syntax_recovery_attempts = 0
    last_syntax_failure_message = None

    # M7.11/M7.12 shared state: tracks which project-relative paths
    # have received a modification that passed *every* check this run
    # (M7.6 semantic, M7.7 target-file, M7.10 syntax) -- i.e. edits
    # that actually stuck, not ones that got reverted. Both checks are
    # recomputed live from this set at each final-answer attempt rather
    # than tracked as separate pending flags.
    successfully_modified_paths = set()
    incomplete_multi_target_retry_attempts = 0
    scope_creep_retry_attempts = 0

    # M7.8 recovery state: tracks whether the most recent *successful*
    # modification has gone unverified by any tool call since (no
    # run_command / read_file / search_files), so a final answer that
    # claims verification happened can be caught and forced to either
    # actually check, or stop claiming it did.
    unverified_modification_pending = False
    verification_claim_recovery_attempts = 0

    # M7.9 guard state: tracks which project-relative paths have
    # actually had their content exposed to the model this run, via
    # read_file, a matching search_files hit, or a successful
    # modification. Used to block write_file/replace_in_file calls
    # against existing files the model hasn't actually looked at.
    files_seen_this_run = set()

    # M7.6: the genuine user request driving this run, captured once up
    # front. Used to infer edit intent (small targeted change vs. full
    # rewrite). Deliberately captured before the loop starts so that the
    # corrective messages this loop injects as "user" turns don't get
    # mistaken for the actual task.
    last_user_request = ""

    for message in reversed(messages):
        if message.get("role") == "user":
            last_user_request = message.get("content", "")
            break

    # Native multi-call turns queue here: every call lands in the
    # transcript's single assistant message up front, then each is
    # processed through the normal pipeline one per loop pass (each
    # counting against the tool budget like any other call).
    pending_calls = []

    while tool_calls < tool_call_limit:

        from_native = False
        native_call_id = None
        native_raw_arguments = ""

        if pending_calls:
            call = pending_calls.pop(0)
            native_call_id = call["id"]
            native_raw_arguments = call["raw"]
            action = {
                "tool": call["name"],
                "arguments": call["arguments"],
            }
            from_native = True
            raw_response = ""
        else:
            # Only the wire copy is trimmed to the context budget; the
            # real transcript (and saved sessions) keep everything.
            wire_messages = trim_messages(
                messages,
                max_context_tokens
                if max_context_tokens is not None
                else get_context_limit(),
            )

            with _ThinkingSpinner(show_thinking):
                if stream:
                    chunks = client.chat(
                        model=model,
                        messages=wire_messages,
                        tools=effective_tool_schemas(),
                        stream=True,
                    )

                    parts = []
                    turn_calls = []

                    for chunk in chunks:
                        delta = extract_chunk_content(chunk)

                        if delta:
                            parts.append(delta)

                            if on_token is not None:
                                on_token(delta)

                        turn_calls.extend(
                            extract_chunk_tool_calls(chunk)
                        )

                    raw_response = "".join(parts)
                else:
                    response = client.chat(
                        model=model,
                        messages=wire_messages,
                        tools=effective_tool_schemas(),
                    )

                    raw_response = extract_chunk_content(response)
                    turn_calls = extract_chunk_tool_calls(response)

            if verbose:
                print(f"\nModel raw response:\n{raw_response}")

            if turn_calls:
                queued = []

                for index, (name, arguments, raw) in enumerate(
                    turn_calls
                ):
                    queued.append({
                        "id": f"call_{len(messages)}_{index}",
                        "name": name,
                        "arguments": arguments,
                        "raw": raw,
                    })

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": (
                                    call["arguments"]
                                    if call["arguments"] is not None
                                    else {}
                                ),
                            },
                        }
                        for call in queued
                    ],
                })

                pending_calls.extend(queued)
                continue

            cleaned = clean_model_response(raw_response)

            if cleaned and not cleaned.startswith("{"):
                # Plain prose with no tool calls: a genuine final
                # answer, handled by the exact same block as the
                # legacy {"tool": "none"} form below.
                action = {"tool": "none", "answer": raw_response}
            else:
                try:
                    if not cleaned:
                        raise ValueError("empty response")

                    action = parse_action(raw_response)

                except (json.JSONDecodeError, ValueError) as e:

                    invalid_action_attempts += 1

                    print(f"\nAgent produced invalid JSON: {e}")

                    if invalid_action_attempts > MAX_INVALID_ACTION_ATTEMPTS:
                        print(
                            "\nAgent stopped after "
                            f"{MAX_INVALID_ACTION_ATTEMPTS} consecutive "
                            "invalid responses."
                        )
                        return None

                    messages.append({
                        "role": "assistant",
                        "content": raw_response,
                    })

                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your last response could not be parsed as JSON: "
                            f"{e}\n\n"
                            "Respond with ONLY a single valid JSON object in "
                            "the required format, and nothing else."
                        ),
                    })

                    continue

        tool_name = normalize_tool_name(
            action.get("tool")
        )

        if from_native and action.get("arguments") is None:
            invalid_action_attempts += 1

            print(
                f"\nTool call arguments are not valid JSON: "
                f"{native_raw_arguments}"
            )

            if invalid_action_attempts > MAX_INVALID_ACTION_ATTEMPTS:
                print(
                    "\nAgent stopped after "
                    f"{MAX_INVALID_ACTION_ATTEMPTS} consecutive "
                    "invalid responses."
                )
                return None

            messages.append({
                "role": "user",
                "content": (
                    f'Tool "{tool_name}" was called with arguments '
                    f"that are not valid JSON:\n"
                    f"{native_raw_arguments}\n\n"
                    "Call it again with corrected arguments."
                ),
            })

            continue

        if tool_name == "none" or tool_name in known_tool_names():
            invalid_action_attempts = 0

        # -----------------------------------------
        # FINAL ANSWER
        # -----------------------------------------

        if tool_name == "none":

            if pending_recovery and recovery_attempts < MAX_RECOVERY_ATTEMPTS:

                recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: a modification "
                    f"attempt failed and has not been recovered from "
                    f"(forced retry {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification attempt failed. This is a "
                        "recoverable tool failure, not a user permission "
                        "denial, so you may not give up yet.\n\n"
                        "Re-read or search the file to find the correct "
                        "exact text, then retry the modification with "
                        "corrected arguments.\n\n"
                        "Only give your final answer again if, after "
                        "actually retrying, you determine you genuinely "
                        "cannot safely complete the modification. In that "
                        "case, explain specifically what you tried and why "
                        "it cannot proceed."
                    ),
                })

                continue

            if (
                pending_semantic_recovery
                and semantic_recovery_attempts < MAX_SEMANTIC_RETRY_ATTEMPTS
            ):

                semantic_recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: the last "
                    "modification failed semantic verification (forced "
                    f"retry {semantic_recovery_attempts}/"
                    f"{MAX_SEMANTIC_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification technically succeeded as "
                        "a tool call, but did not pass verification, so "
                        "the task is not complete and the file has been "
                        "reverted:\n\n"
                        f"{last_semantic_failure_message}\n\n"
                        "Do not tell the user the task is complete. Fix "
                        "the issue above and retry the modification."
                    ),
                })

                continue

            if (
                pending_wrong_file_recovery
                and wrong_file_recovery_attempts < MAX_WRONG_FILE_RETRY_ATTEMPTS
            ):

                wrong_file_recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: the last "
                    "modification appears to have landed on the wrong "
                    f"file (forced retry {wrong_file_recovery_attempts}/"
                    f"{MAX_WRONG_FILE_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification succeeded and passed "
                        "content verification, but appears to have been "
                        "made to the wrong file, so the task is not "
                        "complete and the file has been reverted:\n\n"
                        f"{last_wrong_file_failure_message}\n\n"
                        "Do not tell the user the task is complete. Fix "
                        "the issue above and retry the modification on "
                        "the correct file."
                    ),
                })

                continue

            if (
                pending_syntax_recovery
                and syntax_recovery_attempts < MAX_SYNTAX_RETRY_ATTEMPTS
            ):

                syntax_recovery_attempts += 1

                print(
                    "\nRejecting premature final answer: the last "
                    "modification left a file that doesn't parse "
                    f"(forced retry {syntax_recovery_attempts}/"
                    f"{MAX_SYNTAX_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your last modification succeeded and passed "
                        "content and target-file verification, but left "
                        "the file with invalid syntax, so the task is "
                        "not complete:\n\n"
                        f"{last_syntax_failure_message}\n\n"
                        "Do not tell the user the task is complete. Fix "
                        "the syntax issue above and retry."
                    ),
                })

                continue

            if (
                incomplete_multi_target_retry_attempts
                < MAX_INCOMPLETE_MULTI_TARGET_RETRY_ATTEMPTS
            ):

                incomplete_message = check_incomplete_multi_target_edit(
                    last_user_request,
                    successfully_modified_paths,
                )

                if incomplete_message is not None:

                    incomplete_multi_target_retry_attempts += 1

                    print(
                        "\nRejecting premature final answer: the "
                        "request named multiple files and at least one "
                        "hasn't been modified yet (forced retry "
                        f"{incomplete_multi_target_retry_attempts}/"
                        f"{MAX_INCOMPLETE_MULTI_TARGET_RETRY_ATTEMPTS})."
                    )

                    messages.append({
                        "role": "assistant",
                        "content": raw_response,
                    })

                    messages.append({
                        "role": "user",
                        "content": incomplete_message,
                    })

                    continue

            if (
                scope_creep_retry_attempts
                < MAX_SCOPE_CREEP_RETRY_ATTEMPTS
            ):

                scope_creep_message = check_scope_creep(
                    last_user_request,
                    successfully_modified_paths,
                )

                if scope_creep_message is not None:

                    scope_creep_retry_attempts += 1

                    print(
                        "\nRejecting premature final answer: the "
                        "request was scope-limited but extra files "
                        "were modified (forced retry "
                        f"{scope_creep_retry_attempts}/"
                        f"{MAX_SCOPE_CREEP_RETRY_ATTEMPTS})."
                    )

                    messages.append({
                        "role": "assistant",
                        "content": raw_response,
                    })

                    messages.append({
                        "role": "user",
                        "content": scope_creep_message,
                    })

                    continue

            answer = action.get("answer", "")

            if (
                unverified_modification_pending
                and claims_verification(answer)
                and verification_claim_recovery_attempts
                < MAX_VERIFICATION_CLAIM_RETRY_ATTEMPTS
            ):

                verification_claim_recovery_attempts += 1

                print(
                    "\nRejecting final answer: it claims verification "
                    "that was never actually performed (forced retry "
                    f"{verification_claim_recovery_attempts}/"
                    f"{MAX_VERIFICATION_CLAIM_RETRY_ATTEMPTS})."
                )

                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Your answer claims the change was verified, "
                        "tested, or confirmed, but no verification step "
                        "(run_command, read_file, or search_files) has "
                        "actually been performed since the modification "
                        "was made.\n\n"
                        "Either actually verify the change now (e.g. "
                        "re-read the file, search for the change, or run "
                        "a relevant command), or give your answer again "
                        "without claiming a verification step that "
                        "didn't happen."
                    ),
                })

                continue

            messages.append({
                "role": "assistant",
                "content": raw_response,
            })

            return answer

        # -----------------------------------------
        # TOOL CALL
        # -----------------------------------------

        if tool_name not in known_tool_names():

            invalid_action_attempts += 1

            print(f"\nUnknown tool requested: {tool_name}")

            if invalid_action_attempts > MAX_INVALID_ACTION_ATTEMPTS:
                print(
                    "\nAgent stopped after "
                    f"{MAX_INVALID_ACTION_ATTEMPTS} consecutive "
                    "invalid responses."
                )
                return None

            # Native turns already recorded their assistant message
            # (with all calls) when the turn was queued.
            if not from_native:
                messages.append({
                    "role": "assistant",
                    "content": raw_response,
                })

            valid_tools = ", ".join(sorted(known_tool_names()))

            messages.append({
                "role": "user",
                "content": (
                    f'"{tool_name}" is not a real tool and cannot be '
                    "used.\n\n"
                    f"Available tools: {valid_tools}.\n\n"
                    "Call one of these, or give your final answer if "
                    "you are done."
                ),
            })

            continue

        arguments = action.get("arguments", {})

        print(f"\n{format_tool_call(tool_name, arguments)}")

        # M7.9: before letting any modification tool touch an existing
        # file, check whether the model has actually read or searched
        # that file at all this run. Applies to write_file,
        # replace_in_file, and delete_file alike -- a blind change is
        # exactly as risky whichever tool makes it. Checked first,
        # ahead of M7.6's rewrite guard, since there's no point
        # steering a blind edit towards a "better" tool before the
        # model has even looked at the file.
        if tool_name in MODIFICATION_TOOLS:

            unread_file_guard_message = check_unread_file_edit(
                tool_name,
                arguments,
                files_seen_this_run,
            )

            if unread_file_guard_message is not None:

                print(
                    "\nBlocking modification: target file hasn't been "
                    "read or searched yet this run."
                )

                if not from_native:
                    messages.append({
                        "role": "assistant",
                        "content": raw_response,
                    })

                messages.append({
                    "role": "user",
                    "content": unread_file_guard_message,
                })

                tool_calls += 1

                continue

        # M7.6: before letting an existing file be fully replaced, check
        # whether the request actually described a small, targeted edit
        # that replace_in_file should handle instead. If so, block the
        # write_file call before it ever touches disk.
        if tool_name == "write_file":

            rewrite_guard_message = check_unnecessary_full_rewrite(
                arguments,
                last_user_request,
            )

            if rewrite_guard_message is not None:

                print(
                    "\nBlocking write_file: looks like an unnecessary "
                    "full-file rewrite of a small edit."
                )

                if not from_native:
                    messages.append({
                        "role": "assistant",
                        "content": raw_response,
                    })

                messages.append({
                    "role": "user",
                    "content": rewrite_guard_message,
                })

                tool_calls += 1

                continue

        # M7.6: capture the file's content immediately before a
        # modification tool runs, so it can be compared against the
        # result afterwards (and restored if verification fails).
        modification_path = (
            arguments.get("path")
            if tool_name in MODIFICATION_TOOLS and isinstance(arguments, dict)
            else None
        )
        pre_content = (
            safe_read_file(modification_path)
            if modification_path is not None
            else None
        )

        # Hooks: a non-zero pre_tool hook blocks the call entirely
        # (reported to the model like any other tool failure); post
        # hooks run after execution and can only warn.
        hook_block = run_pre_tool_hooks(
            tool_name, arguments, get_project_root()
        )

        if hook_block is not None:
            print(f"\nBlocking tool call: {hook_block}")
            result = {"error": hook_block}
        else:
            try:
                result = execute_tool(
                    tool_name,
                    arguments,
                )

            except Exception as e:
                result = {
                    "error": str(e)
                }

            run_post_tool_hooks(
                tool_name, arguments, result, get_project_root()
            )

        if isinstance(result, dict) and "error" in result:
            print(f"  ✖ {result['error']}")
        else:
            print_tool_result(result)

        # M7.9: record that this path's real content has now been
        # exposed to the model this run, so a later modification
        # against the same path isn't blocked as blind.
        if tool_name == "read_file" and not (
            isinstance(result, dict) and "error" in result
        ):

            read_path = (
                arguments.get("path") if isinstance(arguments, dict) else None
            )

            if isinstance(read_path, str) and read_path.strip():
                files_seen_this_run.add(read_path)

        elif tool_name == "search_files" and isinstance(result, list):

            for match in result:
                if isinstance(match, dict) and isinstance(
                    match.get("file"), str
                ):
                    files_seen_this_run.add(match["file"])

        outcome = classify_modification_result(tool_name, result)

        if outcome == "failed":
            pending_recovery = True
        elif outcome in ("success", "denied"):
            pending_recovery = False
            recovery_attempts = 0

        # M7.9: a successful modification also counts as having "seen"
        # the file -- the model either wrote that content itself
        # (write_file) or matched an exact substring of it
        # (replace_in_file) -- so a follow-up edit to the same path
        # later in this run isn't blocked as blind.
        if outcome == "success" and modification_path is not None:
            files_seen_this_run.add(modification_path)

        # M7.8: track whether a successful modification still hasn't
        # been checked by any tool call since. Independent of whether
        # M7.6/M7.7 end up flagging this same modification -- those
        # gate the "none" response on their own, so it doesn't matter
        # if this stays set through a revert; the model can't reach a
        # verification-claim check until it gets past those first.
        if outcome == "success":
            unverified_modification_pending = True
            verification_claim_recovery_attempts = 0
        elif tool_name in VERIFICATION_TOOLS:
            unverified_modification_pending = False

        # M7.6: a tool call can report success while still not having
        # done what was asked (wrote the right text but destroyed the
        # rest of the file, etc). Only meaningful when we know what the
        # file looked like both before and after.
        #
        # M7.7: separately, a tool call can report success, write
        # perfectly consistent content, and still have landed on the
        # wrong file. Checked second (only if M7.6 didn't already flag
        # a content problem) since "right file, wrong content" and
        # "wrong file entirely" are different failure classes with
        # their own recovery messages; a single edit only triggers one.
        semantic_failure_message = None
        wrong_file_failure_message = None
        syntax_failure_message = None

        post_content = (
            safe_read_file(modification_path)
            if outcome == "success" and modification_path is not None
            else None
        )

        if pre_content is not None and post_content is not None:

            semantic_failure_message = verify_semantic_edit(
                path=modification_path,
                pre_content=pre_content,
                post_content=post_content,
                last_user_request=last_user_request,
            )

            if semantic_failure_message is None:
                wrong_file_failure_message = verify_target_file(
                    path=modification_path,
                    last_user_request=last_user_request,
                )

        # M7.10: checked whenever a modification/creation succeeded and
        # nothing above already flagged it -- covers both edits to
        # existing files (post_content already fetched above) and
        # brand-new files (pre_content is None, but post_content is
        # still the file's freshly-written content).
        if (
            outcome == "success"
            and post_content is not None
            and semantic_failure_message is None
            and wrong_file_failure_message is None
        ):
            syntax_failure_message = verify_python_syntax(
                modification_path,
                post_content,
            )

        if (
            semantic_failure_message is not None
            or wrong_file_failure_message is not None
            or syntax_failure_message is not None
        ):

            try:
                if pre_content is not None:
                    restore_file_content(modification_path, pre_content)
                    print(
                        f"\nVerification failed for {modification_path}; "
                        "reverted to its pre-edit content."
                    )
                else:
                    remove_created_file(modification_path)
                    print(
                        f"\nVerification failed for newly created "
                        f"{modification_path}; removed it."
                    )
            except Exception as e:
                print(
                    f"\nWarning: could not revert {modification_path} "
                    f"after failed verification: {e}"
                )

        if semantic_failure_message is not None:
            pending_semantic_recovery = True
            last_semantic_failure_message = semantic_failure_message
        elif outcome == "success":
            pending_semantic_recovery = False
            semantic_recovery_attempts = 0
            last_semantic_failure_message = None

        if wrong_file_failure_message is not None:
            pending_wrong_file_recovery = True
            last_wrong_file_failure_message = wrong_file_failure_message
        elif outcome == "success" and semantic_failure_message is None:
            pending_wrong_file_recovery = False
            wrong_file_recovery_attempts = 0
            last_wrong_file_failure_message = None

        if syntax_failure_message is not None:
            pending_syntax_recovery = True
            last_syntax_failure_message = syntax_failure_message
        elif (
            outcome == "success"
            and semantic_failure_message is None
            and wrong_file_failure_message is None
        ):
            pending_syntax_recovery = False
            syntax_recovery_attempts = 0
            last_syntax_failure_message = None

        # M7.11/M7.12: a modification only counts as "done" for the
        # multi-target-completeness and scope-creep checks once it has
        # cleared every other check above -- a reverted or removed
        # edit was never actually applied, so it shouldn't count either
        # towards "this file was handled" or towards "this file was
        # touched beyond what was asked."
        if (
            outcome == "success"
            and modification_path is not None
            and semantic_failure_message is None
            and wrong_file_failure_message is None
            and syntax_failure_message is None
        ):
            successfully_modified_paths.add(modification_path)

        if not from_native:
            messages.append({
                "role": "assistant",
                "content": raw_response,
            })

        if semantic_failure_message is not None:
            followup_content = (
                f"Tool `{tool_name}` reported success, but automatic "
                f"verification found a problem:\n\n"
                f"{semantic_failure_message}\n\n"
                "The file has been reverted to its state before this "
                "change. Do not tell the user the task is complete. Fix "
                "the issue above and try again."
            )
        elif wrong_file_failure_message is not None:
            followup_content = (
                f"Tool `{tool_name}` reported success, and the content "
                "looked correct, but automatic verification found a "
                "problem with which file was modified:\n\n"
                f"{wrong_file_failure_message}\n\n"
                "The file has been reverted to its state before this "
                "change. Do not tell the user the task is complete. Fix "
                "the issue above and try again."
            )
        elif syntax_failure_message is not None:
            revert_note = (
                "reverted to its state before this change"
                if pre_content is not None
                else "removed, since it was newly created"
            )
            followup_content = (
                f"Tool `{tool_name}` reported success, and content/target "
                "verification passed, but automatic syntax checking found "
                f"a problem:\n\n"
                f"{syntax_failure_message}\n\n"
                f"The file has been {revert_note}. Do not tell the user "
                "the task is complete. Fix the syntax issue above and "
                "try again."
            )
        else:
            followup_content = (
                f"Tool `{tool_name}` returned:\n"
                f"{json.dumps(result, indent=2)}\n\n"
                "Continue working. "
                "Use another tool if necessary, "
                "or return your final answer."
            )

        if from_native:
            messages.append({
                "role": "tool",
                "tool_call_id": native_call_id,
                "content": followup_content,
            })
        else:
            messages.append({
                "role": "user",
                "content": followup_content,
            })

        tool_calls += 1

    print(
        f"\nAgent stopped after reaching the "
        f"{tool_call_limit} tool-call limit."
    )

    return None


def run_repl(
    model=None,
    max_tool_calls=None,
    stream=False,
    on_token=None,
    initial_messages=None,
    session_id=None,
    initial_turns=0,
    verbose=False,
):

    model = model or MODEL
    turns = initial_turns

    if initial_messages is not None:
        # Refresh the system message so current project instructions
        # (SEACODE.md/AGENTS.md/CLAUDE.md) apply to resumed sessions.
        messages = fresh_messages(
            initial_messages, SYSTEM_PROMPT, get_project_root()
        )
    else:
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    SYSTEM_PROMPT, get_project_root()
                ),
            }
        ]

    # Without a token callback the model replies accumulate silently;
    # the thinking indicator covers the wait instead of a raw JSON dump.
    show_thinking = on_token is None

    def persist():
        nonlocal session_id

        try:
            session_id = save_session(
                get_project_root(),
                messages,
                model,
                session_id,
                turns,
            )
        except Exception as e:
            print(f"\nWarning: could not save session: {e}")

        return session_id

    header = (
        f"{_bold(f'SeaCode {__version__}')} {_dim('·')} {_cyan(model)} "
        f"{_dim('·')} {_dim(str(get_project_root()))}"
    )

    if session_id is not None:
        header += f" {_dim('·')} {_dim(f'session {session_id}')}"

    try:
        trust = get_trust_level()
    except ValueError:
        trust = "ask"

    if trust != "ask":
        header += f" {_dim('·')} {_dim(f'trust:{trust}')}"

    print(header)
    print(_dim("Type 'quit' to exit."))

    console = Console()

    while True:

        user_input = input(f"\n{_cyan('❯')} ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        turns += 1
        persist()

        started = time.monotonic()

        answer = run_agent(
            messages,
            model=model,
            max_tool_calls=max_tool_calls,
            stream=stream,
            on_token=on_token,
            show_thinking=show_thinking,
            verbose=verbose,
        )

        elapsed = format_duration(time.monotonic() - started)

        persist()

        if answer is not None:
            print(f"\n{_green('●')} {_dim(f'({elapsed})')}")

            try:
                console.print(Markdown(answer))
            except Exception:
                print(answer)


def main():
    """Legacy entry point (equivalent to `seacode` with defaults)."""

    run_repl()


if __name__ == "__main__":
    main()
