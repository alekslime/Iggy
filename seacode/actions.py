"""Turning model output into tool actions.

Cleans raw replies, extracts a tool call from plain text or legacy
JSON, normalizes native tool-call objects from either provider, and
pulls text / tool calls out of streamed chunks. Pure functions: no
client, no tools, no agent state.
"""

import json


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
    """Convert the model's response into a tool/action dictionary.

    Tolerates a leading JSON object followed by extra material
    (a second object, trailing prose): small models often emit
    both, and the first complete object is the actionable one.
    """

    clean_response = clean_model_response(raw_response)

    try:
        action = json.loads(clean_response)
    except json.JSONDecodeError:
        start = clean_response.find("{")

        if start == -1:
            raise

        action, _ = json.JSONDecoder().raw_decode(
            clean_response[start:]
        )

    if not isinstance(action, dict):
        raise ValueError("Model response must be a JSON object.")

    return action


def normalize_action(action: dict) -> dict:
    """Normalize the dialects models actually emit into the canonical
    {"tool": name, "arguments": {...}} / {"tool": "none", "answer": ...}
    shapes the loop consumes.

    Observed in the wild (qwen2.5-coder via Ollama ignores native
    tool_calls and writes JSON by hand, in whatever shape its
    training favored):

      {"name": "replace_in_file", "arguments": {...}}   (OpenAI-ish)
      {"function": "read_file", "parameters": {...}}
      {"tool_calls": [{"function": {"name": ..., "arguments": ...}}]}
      {"answer": "..."}                                  (final reply)

    Arguments given as a JSON string are parsed; an unparseable
    string is passed through so execution fails visibly and the
    recovery loop can correct it, rather than silently dropping it.
    """

    if not isinstance(action, dict):
        return action

    if action.get("tool"):
        return action

    name = action.get("name") or action.get("function")

    if not isinstance(name, str) or not name:
        calls = action.get("tool_calls")

        if isinstance(calls, list) and calls and isinstance(calls[0], dict):
            function = calls[0].get("function") or {}
            name = function.get("name")
            if isinstance(name, str) and name:
                action = {
                    "tool": name,
                    "arguments": function.get("arguments", {}),
                }
                return _coerce_arguments(action)
        elif "answer" in action:
            return {"tool": "none", "answer": action.get("answer") or ""}
        else:
            return action
    else:
        action = {
            "tool": name,
            "arguments": (
                action.get("arguments")
                if action.get("arguments") is not None
                else action.get("parameters", {})
            ),
        }

    return _coerce_arguments(action)


def _coerce_arguments(action: dict) -> dict:
    arguments = action.get("arguments")

    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return action

        if isinstance(parsed, dict):
            action["arguments"] = parsed

    return action


def find_embedded_action(text: str):
    """First recognizable action object embedded in prose, or None.

    Small models narrate a plan ("Step 2: {...} Step 3: {...}")
    instead of emitting one bare call. The enclosing prose must not
    be mistaken for a final answer -- it usually ends by claiming
    work that never ran.
    """

    if not isinstance(text, str):
        return None

    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue

        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue

        if isinstance(candidate, dict) and (
            candidate.get("tool")
            or candidate.get("name")
            or candidate.get("function")
            or candidate.get("tool_calls")
        ):
            return candidate

    return None


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


def normalize_tool_name(tool_name):
    """Normalize tool names returned by the model."""

    if not isinstance(tool_name, str):
        return tool_name

    if "(" in tool_name:
        tool_name = tool_name.split("(", 1)[0].strip()

    return tool_name
