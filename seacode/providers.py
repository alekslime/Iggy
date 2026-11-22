"""Model providers for SeaCode.

Two providers, one interface:

  ollama  local Ollama server via the ollama package (the default).
  openai  any OpenAI-compatible chat-completions endpoint
          (OpenAI, OpenRouter, vLLM, llama.cpp, LM Studio -- even
          Ollama itself, which serves /v1/* too).

Both speak native function calling with the same schema shape, so
TOOL_SCHEMAS is defined once here and passed straight through. The
agent loop (agent.run_agent) also accepts plain-text replies and
legacy {"tool": ...} JSON, so weaker models that ignore the function
contract still work through the same code path.

OpenAIClient mirrors the ollama.Client.chat() calling convention
(chat(model, messages, tools, stream)) and returns plain-dict
responses in the same shape ({"message": {"content", "tool_calls"}}),
so run_agent never branches on provider type.
"""

import json
import os

import httpx
import ollama


PROVIDERS = ("ollama", "openai")

OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"
OPENAI_DEFAULT_HOST = "https://api.openai.com/v1"

# Window requested from Ollama by default. The runtime's own default
# (4096) leaves almost no working room after the system prompt.
DEFAULT_NUM_CTX = 16384


def _string_param(description):
    return {"type": "string", "description": description}


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "list_files",
        "List project files. Use '.' for the project root.",
        {"root": _string_param(
            "Directory to list, relative to the project root."
        )},
        [],
    ),
    _tool(
        "read_file",
        "Read a text file inside the project.",
        {"path": _string_param(
            "File path, relative to the project root."
        )},
        ["path"],
    ),
    _tool(
        "search_files",
        "Search project files for text. Returns file names, line "
        "numbers, and matching lines. Use short, exact keywords.",
        {"query": _string_param("Text to search for.")},
        ["query"],
    ),
    _tool(
        "git_status",
        "Return the current Git working tree status.",
        {},
        [],
    ),
    _tool(
        "run_command",
        "Run an allowlisted shell command inside the project "
        "(python, pytest, git, ls, cat, grep, ...). Asks the user "
        "first unless trusted.",
        {"command": _string_param("Full command line to run.")},
        ["command"],
    ),
    _tool(
        "write_file",
        "Create a new file, or replace an entire existing file. "
        "Prefer replace_in_file for edits to existing files.",
        {
            "path": _string_param(
                "File path, relative to the project root."
            ),
            "content": _string_param("Full file content to write."),
        },
        ["path", "content"],
    ),
    _tool(
        "replace_in_file",
        "Replace one exact piece of text inside a project file. "
        "old_text must occur exactly once.",
        {
            "path": _string_param(
                "File path, relative to the project root."
            ),
            "old_text": _string_param("Exact text to replace."),
            "new_text": _string_param("Replacement text."),
        },
        ["path", "old_text", "new_text"],
    ),
    _tool(
        "delete_file",
        "Delete a file inside the project. The content is backed up "
        "first and can be recovered.",
        {"path": _string_param(
            "File path, relative to the project root."
        )},
        ["path"],
    ),
    _tool(
        "undo_last_change",
        "Restore a file from its most recent backup. Omit path to "
        "undo the most recent backup of any file.",
        {"path": _string_param(
            "File to restore (optional)."
        )},
        [],
    ),
]


def resolve_provider(cli_value=None, config=None) -> str:
    """Return the provider to use: CLI flag wins, then SEACODE_PROVIDER,
    then seacode.toml, then ollama. Unknown values raise ValueError so a
    typo fails fast at startup instead of mid-run."""

    value = None

    if cli_value:
        value = cli_value
    elif os.environ.get("SEACODE_PROVIDER"):
        value = os.environ["SEACODE_PROVIDER"]
    elif isinstance(config, dict) and config.get("provider"):
        value = config["provider"]

    if value is None:
        return "ollama"

    provider = str(value).strip().lower()

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {value!r} "
            f"(expected one of {', '.join(PROVIDERS)})"
        )

    return provider


def resolve_base_url(cli_value=None, config=None, provider="ollama"):
    """Return the endpoint for a provider: CLI --host wins, then
    SEACODE_HOST (plus OLLAMA_HOST for ollama), then seacode.toml's
    host key, then the provider's built-in default."""

    if cli_value:
        return cli_value

    env_value = os.environ.get("SEACODE_HOST")

    if not env_value and provider == "ollama":
        env_value = os.environ.get("OLLAMA_HOST")

    if env_value:
        return env_value

    if isinstance(config, dict) and config.get("host"):
        return config["host"]

    if provider == "openai":
        return OPENAI_DEFAULT_HOST

    return OLLAMA_DEFAULT_HOST


def resolve_num_ctx(cli_value=None, config=None):
    """Context window to request from the local runtime: CLI flag,
    then SEACODE_NUM_CTX, then seacode.toml's num_ctx, then
    DEFAULT_NUM_CTX. Ollama otherwise loads models with a 4096-token
    window, which the system prompt alone nearly fills."""

    for value in (
        cli_value,
        os.environ.get("SEACODE_NUM_CTX"),
        config.get("num_ctx") if isinstance(config, dict) else None,
    ):
        if value is None:
            continue

        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue

        if parsed > 0:
            return parsed

    return DEFAULT_NUM_CTX


def build_client(provider=None, host=None, api_key=None, config=None):
    """Construct the chat client for a provider.

    Both clients expose chat(model, messages, tools, stream) and return
    {"message": {"content", "tool_calls"}} dicts, so the agent loop
    never branches on which one it holds. `host` is the already-resolved
    endpoint (see resolve_base_url); when omitted, the provider default
    is used. Ollama clients ignore api_key.
    """

    provider = resolve_provider(provider, config)
    base_url = host or resolve_base_url(None, config, provider)

    if provider == "ollama":
        return ollama.Client(host=base_url)

    return OpenAIClient(
        base_url=base_url,
        api_key=api_key or resolve_api_key(None, config),
    )


def resolve_api_key(cli_value=None, config=None):
    """Return the API key for keyed providers: CLI flag wins, then
    SEACODE_API_KEY / OPENAI_API_KEY, then seacode.toml. May be None:
    local OpenAI-compatible servers need no key."""

    if cli_value:
        return cli_value

    env_value = os.environ.get("SEACODE_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )

    if env_value:
        return env_value

    if isinstance(config, dict):
        return config.get("api_key")

    return None


def to_openai_message(message: dict) -> dict:
    """Convert one transcript message to OpenAI wire shape.

    Our transcript already uses OpenAI-compatible roles; the only
    fixup is serializing tool-call arguments, which we store parsed
    but OpenAI expects as a JSON string.
    """

    converted = dict(message)
    tool_calls = converted.get("tool_calls")

    if tool_calls:
        normalized = []

        for call in tool_calls:
            call = dict(call)
            function = dict(call.get("function", {}))
            arguments = function.get("arguments")

            if not isinstance(arguments, str):
                function["arguments"] = json.dumps(arguments or {})

            call["function"] = function
            normalized.append(call)

        converted["tool_calls"] = normalized

    if converted.get("content") is None:
        converted["content"] = ""

    return converted


def parse_sse_payloads(body: str):
    """Split a server-sent-events body into decoded JSON payloads,
    skipping comments, blanks, and the terminal [DONE] marker."""

    payloads = []

    for line in body.splitlines():
        line = line.strip()

        if not line.startswith("data:"):
            continue

        data = line[len("data:"):].strip()

        if not data or data == "[DONE]":
            continue

        payloads.append(json.loads(data))

    return payloads


class OpenAIClient:
    """Minimal OpenAI-compatible chat client with an ollama-style
    interface, so the agent loop treats both providers identically.

    Streaming yields text-delta dicts as they arrive and one final
    dict carrying any complete tool calls, mirroring how the ollama
    client surfaces streamed tool calls.
    """

    def __init__(self, base_url, api_key=None, timeout=180,
                 transport=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport

    def _headers(self):
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    def _client(self):
        if self._transport is not None:
            return httpx.Client(
                transport=self._transport, timeout=self.timeout
            )

        return httpx.Client(timeout=self.timeout)

    def _payload(self, model, messages, tools, stream):
        payload = {
            "model": model,
            "messages": [to_openai_message(m) for m in messages],
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        return payload

    def _raise_for_response(self, response):
        if response.status_code < 400:
            return

        try:
            detail = response.json()
            message = detail.get("error", {}).get("message", detail)
        except ValueError:
            message = response.text[:300]

        raise RuntimeError(
            f"OpenAI API error {response.status_code}: {message}"
        )

    def chat(self, model, messages, tools=None, stream=False):
        if stream:
            return self._chat_stream(model, messages, tools)

        with self._client() as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(model, messages, tools, False),
            )

        self._raise_for_response(response)

        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as e:
            raise RuntimeError(
                f"OpenAI API returned an unusable response: {e}"
            )

        return {"message": self._normalize_message(message)}

    def _normalize_message(self, message):
        """Convert one response message to {content, tool_calls} with
        parsed argument dicts (None when unparseable, raw kept)."""

        content = message.get("content") or ""
        tool_calls = []

        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            name = function.get("name", "")
            raw = function.get("arguments", "")

            try:
                arguments = (
                    json.loads(raw) if isinstance(raw, str) else raw
                )
            except ValueError:
                arguments = None

            if not isinstance(arguments, dict):
                arguments = None if raw else {}

            tool_calls.append({
                "id": call.get("id"),
                "function": {
                    "name": name,
                    "arguments": arguments,
                    "raw": raw if isinstance(raw, str) else "",
                },
            })

        return {"content": content, "tool_calls": tool_calls}

    def _chat_stream(self, model, messages, tools):
        with self._client() as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(model, messages, tools, True),
            ) as response:
                self._raise_for_response(response)

                content_parts = []
                calls_by_index = {}

                for line in response.iter_lines():
                    for payload in parse_sse_payloads(line + "\n"):
                        delta = payload["choices"][0].get(
                            "delta", {}
                        )

                        content = delta.get("content")

                        if content:
                            content_parts.append(content)
                            yield {
                                "message": {"content": content}
                            }

                        for call in delta.get("tool_calls") or []:
                            index = call.get("index", 0)
                            slot = calls_by_index.setdefault(index, {
                                "id": None,
                                "name": "",
                                "arguments": "",
                            })

                            if call.get("id"):
                                slot["id"] = call["id"]

                            function = call.get("function", {})

                            if function.get("name"):
                                slot["name"] += function["name"]

                            if function.get("arguments"):
                                slot["arguments"] += function[
                                    "arguments"
                                ]

                tool_calls = []

                for index in sorted(calls_by_index):
                    slot = calls_by_index[index]
                    raw = slot["arguments"]

                    try:
                        arguments = json.loads(raw) if raw else {}
                    except ValueError:
                        arguments = None

                    tool_calls.append({
                        "id": slot["id"],
                        "function": {
                            "name": slot["name"],
                            "arguments": arguments,
                            "raw": raw,
                        },
                    })

                if tool_calls or not content_parts:
                    yield {
                        "message": {
                            "content": "",
                            "tool_calls": tool_calls,
                        }
                    }
