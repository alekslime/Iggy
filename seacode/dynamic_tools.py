"""Registry for tools added at runtime (MCP servers, subagents).

Built-in tools live in seacode.tools.TOOLS with schemas in
seacode.providers.TOOL_SCHEMAS. Dynamic tools follow the same shape:
a function-calling schema plus a Python handler taking **arguments
and returning a string/dict/list result. The agent loop merges both
sets (see agent.execute_tool and the schemas it sends), so dynamic
tools get the same transcript, guardrail, and permission treatment
as built-ins -- with one deliberate exception: the M7 modification
checks only classify the known built-in modification tools, so a
dynamic tool that writes files is responsible for its own safety
(MCP servers are user-configured and trusted by definition).

Registration is process-global, like tools.set_extra_commands: the
CLI registers MCP tools at startup; tests register fakes and clear
them afterwards.
"""


_dynamic_tools = {}  # name -> (schema, handler)


def register_dynamic_tool(schema, handler) -> str:
    """Register one dynamic tool from its function-calling schema.

    Returns the tool name. Raises ValueError on a malformed schema or
    a name clash with an already-registered dynamic tool.
    """

    try:
        name = schema["function"]["name"]
    except (KeyError, TypeError):
        raise ValueError(f"Malformed tool schema: {schema!r}")

    if not name or not callable(handler):
        raise ValueError(f"Malformed tool schema: {schema!r}")

    if name in _dynamic_tools:
        raise ValueError(f"Dynamic tool already registered: {name}")

    _dynamic_tools[name] = (schema, handler)
    return name


def unregister_dynamic_tool(name: str) -> None:
    _dynamic_tools.pop(name, None)


def clear_dynamic_tools() -> None:
    _dynamic_tools.clear()


def dynamic_tool_schemas() -> list:
    return [schema for schema, _ in _dynamic_tools.values()]


def dynamic_tool_names() -> list:
    return list(_dynamic_tools)


def get_dynamic_handler(name: str):
    entry = _dynamic_tools.get(name)
    return entry[1] if entry else None
