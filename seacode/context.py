"""Project instructions and context-window management for SeaCode.

Two jobs:

  Instructions -- if the project root contains SEACODE.md, AGENTS.md,
  or CLAUDE.md, their contents are appended to the system prompt so
  the model works by the project's own rules. All three are loaded
  (in that order) and the total is capped, so a huge CLAUDE.md can't
  evict the actual conversation.

  Context budget -- transcripts grow unbounded across a long session,
  and small local models have small windows. Before each model call,
  run_agent sends trim_messages()' output instead of the raw
  transcript: system message(s) first, then as many of the most
  recent messages as fit the token budget, with a note where older
  history was dropped. The stored transcript (and saved sessions)
  keep everything; only the wire copy is trimmed.

Token counts are estimates (chars / 4, the usual rough ratio), which
is fine for a budget guard -- this is a safety valve, not billing.
"""

import os
from pathlib import Path


INSTRUCTION_FILENAMES = ("SEACODE.md", "AGENTS.md", "CLAUDE.md")

MAX_INSTRUCTION_CHARS = 12_000

DEFAULT_CONTEXT_TOKENS = 16_000

CHARS_PER_TOKEN = 4

_context_limit = None


def load_project_instructions(project_root) -> str:
    """Concatenated project instruction files, or "" when none exist.

    Files are read in INSTRUCTION_FILENAMES order, each under its own
    heading, and the combined text is capped at MAX_INSTRUCTION_CHARS.
    Unreadable files are skipped rather than fatal.
    """

    root = Path(project_root)
    sections = []

    for name in INSTRUCTION_FILENAMES:
        path = root / name

        if not path.is_file():
            continue

        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if text:
            sections.append(f"## Project instructions ({name})\n\n{text}")

    combined = "\n\n".join(sections)

    if len(combined) > MAX_INSTRUCTION_CHARS:
        combined = (
            combined[:MAX_INSTRUCTION_CHARS]
            + "\n\n[... project instructions truncated ...]"
        )

    return combined


def build_system_prompt(base_prompt: str, project_root) -> str:
    """The base system prompt plus any project instructions."""

    instructions = load_project_instructions(project_root)

    if not instructions:
        return base_prompt

    return (
        base_prompt.rstrip()
        + "\n\nPROJECT INSTRUCTIONS:\n\n"
        + instructions
        + "\n"
    )


def fresh_messages(messages, base_prompt: str, project_root):
    """Refresh a transcript's system message in place from the current
    base prompt + project instructions, returning the same list.

    Used both for new conversations and on resume, so editing
    SEACODE.md (or switching projects) takes effect without starting
    a new session. Non-system history is preserved as-is; a missing
    system message is inserted at the front. In-place refresh keeps
    the identity callers (and their tests) already rely on.
    """

    content = build_system_prompt(base_prompt, project_root)

    if messages and isinstance(messages[0], dict) \
            and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": content}
    else:
        messages.insert(0, {"role": "system", "content": content})

    return messages


def estimate_tokens(text) -> int:
    """Rough token estimate for a string (chars / CHARS_PER_TOKEN)."""

    if not isinstance(text, str):
        return 0

    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def estimate_message_tokens(message) -> int:
    if not isinstance(message, dict):
        return 0

    total = estimate_tokens(message.get("content") or "")

    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        total += estimate_tokens(str(function.get("arguments", "")))
        total += estimate_tokens(str(function.get("name", "")))

    return total + 4  # per-message framing overhead


def coerce_context_tokens(value):
    """Best-effort positive int for a limit from env/config; None if
    unusable."""

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    return None


def set_context_limit(value) -> None:
    """Pin the context budget explicitly (the CLI flag/config path)."""

    parsed = coerce_context_tokens(value)

    if parsed is None:
        raise ValueError(
            f"Unknown context token limit: {value!r} (expected a "
            "positive integer)"
        )

    global _context_limit
    _context_limit = parsed


def reset_context_limit() -> None:
    """Clear the explicit override (used by tests)."""

    global _context_limit
    _context_limit = None


def get_context_limit(config=None) -> int:
    """Effective budget: pinned value, then SEACODE_CONTEXT_TOKENS,
    then seacode.toml's context_tokens, then the default."""

    if _context_limit is not None:
        return _context_limit

    from_env = coerce_context_tokens(os.environ.get("SEACODE_CONTEXT_TOKENS"))

    if from_env is not None:
        return from_env

    if isinstance(config, dict):
        from_config = coerce_context_tokens(config.get("context_tokens"))

        if from_config is not None:
            return from_config

    return DEFAULT_CONTEXT_TOKENS


def resolve_context_tokens(cli_value=None, config=None) -> int:
    """CLI flag wins over get_context_limit()'s env/config/default."""

    if cli_value is not None:
        parsed = coerce_context_tokens(cli_value)

        if parsed is None:
            raise ValueError(
                f"Unknown context token limit: {cli_value!r} (expected "
                "a positive integer)"
            )

        return parsed

    return get_context_limit(config)


def trim_messages(messages, limit_tokens=None):
    """Return a wire copy of `messages` that fits the token budget.

    Leading system messages are always kept. The most recent messages
    that fit the remaining budget follow, preceded by a short system
    note when anything was dropped. Leading `tool` results whose
    assistant call was trimmed away are dropped too (providers reject
    orphaned tool messages). The final message is always kept, even
    if it alone exceeds the budget (its content is truncated instead).
    The input list is never mutated.
    """

    if limit_tokens is None:
        limit_tokens = get_context_limit()

    if not messages:
        return []

    system = []
    rest = []

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system" and not rest:
            system.append(dict(message))
        else:
            rest.append(message)

    budget = limit_tokens - sum(
        estimate_message_tokens(m) for m in system
    )

    if budget <= 0:
        return system + [dict(m) for m in rest[-1:]]

    kept = []
    used = 0

    for message in reversed(rest):
        cost = estimate_message_tokens(message)

        if kept and used + cost > budget:
            break

        kept.append(message)
        used += cost

    kept.reverse()
    dropped = len(rest) - len(kept)

    # Never lead with an orphaned tool result.
    while kept and isinstance(kept[0], dict) and kept[0].get("role") == "tool":
        kept.pop(0)
        dropped += 1

    result = list(system)

    if dropped:
        result.append({
            "role": "system",
            "content": (
                f"[Context trimmed: {dropped} earlier message(s) were "
                "omitted to fit the context window. The full transcript "
                "is preserved in the saved session.]"
            ),
        })

    result.extend(dict(m) for m in kept)

    # A single oversized final message: truncate its content rather
    # than dropping the user's actual request.
    if result and estimate_message_tokens(result[-1]) > limit_tokens:
        last = result[-1]
        allowance = max(limit_tokens * CHARS_PER_TOKEN - 64, 0)
        content = last.get("content") or ""

        if len(content) > allowance:
            last["content"] = (
                content[:allowance] + "\n[... message truncated ...]"
            )

    return result
