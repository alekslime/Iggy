"""
Tests for #5 ecosystem features: dynamic tools, the `task` subagent,
pre/post tool hooks, and the MCP stdio client/manager.

The MCP tests spawn a real (fake) MCP server subprocess over stdio --
no network, no external server needed.

Run with:  python3 -m pytest tests/test_ecosystem.py -v
       or: python3 -m unittest tests.test_ecosystem -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.dynamic_tools as dynamic_module
import seacode.hooks as hooks_module
import seacode.mcp as mcp_module
import seacode.tools as tools_module


def fake_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def tool_call(name, **arguments):
    return json.dumps({"tool": name, "arguments": arguments})


def final(answer):
    return json.dumps({"tool": "none", "answer": answer})


class ScriptedChat:
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen = []

    def __call__(self, model, messages, **kwargs):
        self.seen.append((model, list(messages), kwargs))
        if not self._responses:
            raise AssertionError("ScriptedChat ran out of responses.")
        return fake_response(self._responses.pop(0))


class ProjectTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)
        root_patcher = patch.object(
            tools_module, "get_project_root",
            return_value=self.project_root,
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)
        # agent.run_agent resolves the project root through its own
        # imported binding.
        agent_root_patcher = patch.object(
            seacode_module, "get_project_root",
            return_value=self.project_root,
        )
        agent_root_patcher.start()
        self.addCleanup(agent_root_patcher.stop)
        perm_patcher = patch.object(
            tools_module, "request_permission", return_value=True
        )
        perm_patcher.start()
        self.addCleanup(perm_patcher.stop)
        self.addCleanup(dynamic_module.clear_dynamic_tools)
        self.addCleanup(hooks_module.clear_hooks)

    def tearDown(self):
        self._tmpdir.cleanup()

    def run_scripted(self, responses, request="do the thing"):
        chat = ScriptedChat(responses)
        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ]
        with patch.object(seacode_module.client, "chat", side_effect=chat):
            answer = seacode_module.run_agent(messages)
        return answer, chat, messages


class DynamicToolTests(ProjectTestCase):

    def test_dynamic_tool_callable_by_model(self):
        dynamic_module.register_dynamic_tool(
            {
                "type": "function",
                "function": {
                    "name": "echo_thing",
                    "description": "echo",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            },
            lambda text: f"echo:{text}",
        )
        answer, chat, _ = self.run_scripted([
            tool_call("echo_thing", text="hi"),
            final("done"),
        ])
        self.assertEqual(answer, "done")
        # The dynamic schema was offered to the provider.
        schemas = chat.seen[0][2]["tools"]
        names = [s["function"]["name"] for s in schemas]
        self.assertIn("echo_thing", names)
        self.assertIn("task", names)

    def test_register_validation(self):
        with self.assertRaises(ValueError):
            dynamic_module.register_dynamic_tool({}, lambda: None)
        dynamic_module.register_dynamic_tool(
            {"function": {"name": "x_tool"}}, lambda: "x"
        )
        with self.assertRaises(ValueError):
            dynamic_module.register_dynamic_tool(
                {"function": {"name": "x_tool"}}, lambda: "y"
            )


class SubagentTests(ProjectTestCase):

    def test_task_delegates_and_returns_summary(self):
        answer, chat, _ = self.run_scripted([
            tool_call("task", prompt="find the answer"),
            "Subagent reporting.",  # subagent's final (plain prose)
            final("parent done"),
        ])
        self.assertEqual(answer, "parent done")
        # Three chats: parent->task, subagent, parent final.
        self.assertEqual(len(chat.seen), 3)
        sub_messages = chat.seen[1][1]
        self.assertIn("subagent", sub_messages[0]["content"])
        self.assertEqual(sub_messages[1]["content"], "find the answer")
        # The subagent was not offered the task tool.
        sub_names = [
            s["function"]["name"] for s in chat.seen[1][2]["tools"]
        ]
        self.assertNotIn("task", sub_names)

    def test_subagent_cannot_spawn_subagent(self):
        seacode_module._agent_state.depth = 1
        try:
            result = seacode_module.execute_tool(
                "task", {"prompt": "nested"}
            )
        finally:
            seacode_module._agent_state.depth = 0
        self.assertIn("cannot spawn", result)

    def test_task_requires_prompt(self):
        result = seacode_module.execute_tool("task", {})
        self.assertIn("prompt", result)


HOOK_GUARD = """
import json, sys
payload = json.load(sys.stdin)
if payload["tool"] == "write_file":
    print("writes are forbidden by policy")
    sys.exit(3)
"""

HOOK_LOG = """
import json, sys
payload = json.load(sys.stdin)
with open("hook-log.txt", "a") as fh:
    fh.write(payload["event"] + ":" + payload["tool"] + "\\n")
"""


class HookTests(ProjectTestCase):

    def write_hook(self, name, source):
        path = self.project_root / name
        path.write_text(source)
        return f"{sys.executable} {path}"

    def test_load_hooks_from_toml(self):
        (self.project_root / "seacode.toml").write_text(
            '[hooks]\npre_tool = ["python a.py"]\n'
            'post_tool_use = ["python b.py"]\n'
        )
        hooks = hooks_module.load_hooks(self.project_root)
        self.assertEqual(hooks["pre_tool"], ["python a.py"])
        self.assertEqual(hooks["post_tool"], ["python b.py"])

    def test_pre_hook_blocks_write(self):
        guard = self.write_hook("guard.py", HOOK_GUARD)
        log = self.write_hook("log.py", HOOK_LOG)
        hooks_module.set_hooks(
            {"pre_tool": [guard], "post_tool": [log]}
        )
        (self.project_root / "a.txt").write_text("original\n")

        answer, _, _ = self.run_scripted([
            tool_call("read_file", path="a.txt"),
            tool_call("write_file", path="a.txt", content="hacked\n"),
            # The blocked write counts as a failed modification, so
            # M7.3 forces retries before accepting the final answer.
            final("could not write"),
            final("could not write"),
            final("could not write"),
        ], request="update a.txt")

        self.assertEqual(answer, "could not write")
        self.assertEqual(
            (self.project_root / "a.txt").read_text(), "original\n"
        )
        log_text = (self.project_root / "hook-log.txt").read_text()
        self.assertIn("post_tool:read_file", log_text)

    def test_no_hooks_no_block(self):
        (self.project_root / "a.txt").write_text("original\n")
        answer, _, _ = self.run_scripted([
            tool_call("read_file", path="a.txt"),
            tool_call("write_file", path="a.txt", content="new\n"),
            final("wrote it"),
        ], request="update a.txt")
        self.assertEqual(answer, "wrote it")
        self.assertEqual((self.project_root / "a.txt").read_text(), "new\n")


FAKE_MCP_SERVER = """
import json, sys

def send(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("method") == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "serverInfo": {"name": "fake", "version": "0"}}})
    elif msg.get("method") == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [{
            "name": "shout",
            "description": "Uppercase text",
            "inputSchema": {"type": "object", "properties": {
                "text": {"type": "string"}}, "required": ["text"]},
        }]}})
    elif msg.get("method") == "tools/call":
        text = msg["params"]["arguments"].get("text", "")
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "content": [{"type": "text", "text": text.upper()}]}})
"""


class MCPTests(ProjectTestCase):

    def write_server(self):
        server = self.project_root / "fake_mcp.py"
        server.write_text(FAKE_MCP_SERVER)
        (self.project_root / ".mcp.json").write_text(json.dumps({
            "mcpServers": {
                "fake": {
                    "command": sys.executable,
                    "args": [str(server)],
                }
            }
        }))

    def test_load_config(self):
        self.write_server()
        config = mcp_module.load_mcp_config(self.project_root)
        self.assertIn("fake", config)
        self.assertEqual(config["fake"]["command"], sys.executable)

    def test_manager_registers_and_calls_tools(self):
        self.write_server()
        manager = mcp_module.MCPManager()
        try:
            registered = manager.start_all(self.project_root)
            self.assertEqual(registered, ["mcp_fake_shout"])
            result = seacode_module.execute_tool(
                "mcp_fake_shout", {"text": "hello"}
            )
            self.assertEqual(result, "HELLO")
        finally:
            manager.stop_all()
        self.assertEqual(dynamic_module.dynamic_tool_names(), [])

    def test_bad_server_skipped(self):
        (self.project_root / ".mcp.json").write_text(json.dumps({
            "mcpServers": {
                "nope": {"command": "/nonexistent/binary-xyz", "args": []}
            }
        }))
        manager = mcp_module.MCPManager()
        try:
            self.assertEqual(manager.start_all(self.project_root), [])
        finally:
            manager.stop_all()

    def test_mcp_tool_end_to_end_through_agent(self):
        self.write_server()
        manager = mcp_module.MCPManager()
        try:
            manager.start_all(self.project_root)
            answer, _, _ = self.run_scripted([
                tool_call("mcp_fake_shout", text="hey"),
                final("shouted"),
            ])
            self.assertEqual(answer, "shouted")
        finally:
            manager.stop_all()


if __name__ == "__main__":
    unittest.main()
