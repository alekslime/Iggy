"""MCP (Model Context Protocol) client over stdio.

SeaCode reads the standard `.mcp.json` in the project root:

    {"mcpServers": {
        "files": {"command": "python", "args": ["server.py"], "env": {}}
    }}

Each configured server is spawned as a subprocess speaking JSON-RPC
2.0 over stdin/stdout (the MCP stdio transport): `initialize`, then
`tools/list`, and `tools/call` per invocation. Discovered tools are
registered in seacode.dynamic_tools with namespaced names
(`mcp_<server>_<tool>`) and MCP inputSchemas (already JSON Schema,
the same shape function-calling uses).

Servers are programs named by the project, so the CLI asks the user
before starting them (see seacode.project_trust); this module only
does the spawning. Once approved, their tools are *not*
permission-gated individually and MCP tool results are treated like
any other tool result by the agent loop. A server that fails to start
or answer is skipped with a warning -- one bad server must not take
the whole session down.
"""

import json
import os
import re
import subprocess
from pathlib import Path


MCP_CONFIG_FILENAME = ".mcp.json"
PROTOCOL_VERSION = "2024-11-05"
REQUEST_TIMEOUT_SECONDS = 30


def load_mcp_config(project_root) -> dict:
    """Return {server_name: {"command", "args", "env"}} from .mcp.json
    ({} when missing or malformed)."""

    path = Path(project_root) / MCP_CONFIG_FILENAME

    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    servers = data.get("mcpServers") or data.get("servers") or {}

    if not isinstance(servers, dict):
        return {}

    config = {}

    for name, spec in servers.items():
        if not isinstance(spec, dict) or not spec.get("command"):
            continue

        config[name] = {
            "command": spec["command"],
            "args": list(spec.get("args") or []),
            "env": dict(spec.get("env") or {}),
        }

    return config


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def tool_name_for(server_name: str, tool_name: str) -> str:
    return f"mcp_{_safe_name(server_name)}_{_safe_name(tool_name)}"


class MCPError(RuntimeError):
    pass


class MCPServerClient:
    """One MCP server subprocess with a minimal JSON-RPC client."""

    def __init__(self, name, command, args=(), env=None):
        self.name = name
        self._command = [command, *args]
        self._env = {**os.environ, **(env or {})}
        self._process = None
        self._next_id = 1
        self.tools = []  # raw MCP tool descriptors

    # -- lifecycle ---------------------------------------------------

    def start(self, cwd=None):
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
            env=self._env,
        )

        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "seacode", "version": "0.1.0"},
        })
        self._notify("notifications/initialized", {})
        self.tools = self._request("tools/list", {}).get("tools", [])
        return self.tools

    def stop(self):
        if self._process is None:
            return

        try:
            if self._process.stdin:
                self._process.stdin.close()
        except OSError:
            pass

        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass

        self._process = None

    # -- JSON-RPC plumbing --------------------------------------------

    def _send(self, payload):
        if self._process is None or self._process.stdin is None:
            raise MCPError(f"MCP server '{self.name}' is not running.")

        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

    def _notify(self, method, params):
        self._send({
            "jsonrpc": "2.0", "method": method, "params": params,
        })

    def _request(self, method, params):
        request_id = self._next_id
        self._next_id += 1

        self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        })

        while True:
            line = self._process.stdout.readline()

            if not line:
                raise MCPError(
                    f"MCP server '{self.name}' closed the connection "
                    f"during '{method}'."
                )

            try:
                message = json.loads(line)
            except ValueError:
                continue  # non-JSON noise on stdout; keep waiting

            if message.get("id") != request_id:
                continue  # notification or unrelated response

            if "error" in message:
                raise MCPError(
                    f"MCP server '{self.name}' error on '{method}': "
                    f"{message['error']}"
                )

            return message.get("result") or {}

    # -- tools ---------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })

        parts = []

        for item in result.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                parts.append(json.dumps(item))

        text = "\n".join(parts)

        if result.get("isError"):
            return f"MCP tool '{tool_name}' reported an error: {text}"

        return text or "(no output)"


class MCPManager:
    """Starts every configured server and registers its tools."""

    def __init__(self):
        self.clients = {}  # server name -> MCPServerClient
        self.registered = []  # dynamic tool names

    def start_all(self, project_root):
        """Spawn all servers from .mcp.json. Returns the dynamic tool
        names registered. Bad servers warn and are skipped."""

        from seacode.dynamic_tools import register_dynamic_tool

        config = load_mcp_config(project_root)

        for server_name, spec in config.items():
            client = MCPServerClient(
                server_name,
                spec["command"],
                spec["args"],
                spec["env"],
            )

            try:
                tools = client.start(cwd=project_root)
            except Exception as e:
                print(f"Warning: MCP server '{server_name}' "
                      f"did not start: {e}")
                client.stop()
                continue

            self.clients[server_name] = client

            for tool in tools:
                mcp_name = tool.get("name", "")
                dynamic_name = tool_name_for(server_name, mcp_name)
                schema = {
                    "type": "function",
                    "function": {
                        "name": dynamic_name,
                        "description": (
                            tool.get("description")
                            or f"MCP tool {mcp_name} from {server_name}"
                        ),
                        "parameters": tool.get(
                            "inputSchema",
                            {"type": "object", "properties": {}},
                        ),
                    },
                }

                def make_handler(c=client, n=mcp_name):
                    def handler(**arguments):
                        return c.call_tool(n, arguments)
                    return handler

                try:
                    register_dynamic_tool(schema, make_handler())
                    self.registered.append(dynamic_name)
                except ValueError as e:
                    print(f"Warning: {e}")

        return list(self.registered)

    def stop_all(self):
        from seacode.dynamic_tools import unregister_dynamic_tool

        for client in self.clients.values():
            client.stop()

        for name in self.registered:
            unregister_dynamic_tool(name)

        self.clients = {}
        self.registered = []
