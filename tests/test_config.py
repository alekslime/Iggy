"""
Tests for runtime configuration: model, host, tool-call limit, and
project directory resolution.

Precedence everywhere is CLI flag > environment variable > built-in
default. run_agent() accepts per-run overrides so the CLI can apply
--model / --max-tools without touching module globals.

Run with:  python3 -m pytest tests/test_config.py -v
       or: python3 -m unittest tests.test_config -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.tools as tools_module


def fake_response(content: str):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def final_answer(answer):
    return json.dumps({"tool": "none", "answer": answer})


class RecordingChat:
    """Scripted chat that also records which model each turn used."""

    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.models = []

    def __call__(self, model, messages):
        self.models.append(model)
        if not self._responses:
            raise AssertionError(
                "RecordingChat ran out of scripted responses."
            )
        return fake_response(self._responses.pop(0))


class ResolveConfigTests(unittest.TestCase):

    def test_model_cli_wins_over_env_and_default(self):
        with patch.dict(os.environ, {"SEACODE_MODEL": "env-model"}):
            self.assertEqual(
                seacode_module.resolve_model("cli-model"), "cli-model"
            )

    def test_model_env_wins_over_default(self):
        with patch.dict(os.environ, {"SEACODE_MODEL": "env-model"}):
            self.assertEqual(
                seacode_module.resolve_model(), "env-model"
            )

    def test_model_default(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_MODEL", None)
            self.assertEqual(
                seacode_module.resolve_model(),
                seacode_module.DEFAULT_MODEL,
            )

    def test_host_cli_wins(self):
        with patch.dict(
            os.environ,
            {"SEACODE_HOST": "http://seacode:11434",
             "OLLAMA_HOST": "http://ollama:11434"},
        ):
            self.assertEqual(
                seacode_module.resolve_host("http://cli:11434"),
                "http://cli:11434",
            )

    def test_host_prefers_seacode_host_over_ollama_host(self):
        with patch.dict(
            os.environ,
            {"SEACODE_HOST": "http://seacode:11434",
             "OLLAMA_HOST": "http://ollama:11434"},
        ):
            self.assertEqual(
                seacode_module.resolve_host(), "http://seacode:11434"
            )

    def test_host_falls_back_to_ollama_host(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://ollama:11434"}):
            os.environ.pop("SEACODE_HOST", None)
            self.assertEqual(
                seacode_module.resolve_host(), "http://ollama:11434"
            )

    def test_host_default(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_HOST", None)
            os.environ.pop("OLLAMA_HOST", None)
            self.assertEqual(
                seacode_module.resolve_host(), seacode_module.DEFAULT_HOST
            )

    def test_max_tools_cli_wins(self):
        with patch.dict(os.environ, {"SEACODE_MAX_TOOLS": "7"}):
            self.assertEqual(
                seacode_module.resolve_max_tool_calls(3), 3
            )

    def test_max_tools_env(self):
        with patch.dict(os.environ, {"SEACODE_MAX_TOOLS": "7"}):
            self.assertEqual(seacode_module.resolve_max_tool_calls(), 7)

    def test_max_tools_default(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_MAX_TOOLS", None)
            self.assertEqual(
                seacode_module.resolve_max_tool_calls(),
                seacode_module.MAX_TOOL_CALLS,
            )

    def test_max_tools_unparseable_env_falls_back_to_default(self):
        with patch.dict(os.environ, {"SEACODE_MAX_TOOLS": "lots"}):
            self.assertEqual(
                seacode_module.resolve_max_tool_calls(),
                seacode_module.MAX_TOOL_CALLS,
            )


class ProjectRootTests(unittest.TestCase):

    def test_seacode_dir_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SEACODE_DIR": tmpdir}):
                self.assertEqual(
                    tools_module.get_project_root(),
                    Path(tmpdir).resolve(),
                )

    def test_defaults_to_working_directory(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_DIR", None)
            self.assertEqual(
                tools_module.get_project_root(), Path.cwd().resolve()
            )


class RunAgentOverridesTests(unittest.TestCase):

    def test_run_agent_uses_given_model(self):
        chat = RecordingChat([final_answer("Done.")])
        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": "Hi."},
        ]

        with patch.object(
            seacode_module.client, "chat", side_effect=chat
        ):
            answer = seacode_module.run_agent(
                messages, model="custom-model:tag"
            )

        self.assertEqual(answer, "Done.")
        self.assertEqual(chat.models, ["custom-model:tag"])

    def test_run_agent_defaults_to_module_model(self):
        chat = RecordingChat([final_answer("Done.")])
        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": "Hi."},
        ]

        with patch.object(
            seacode_module.client, "chat", side_effect=chat
        ):
            seacode_module.run_agent(messages)

        self.assertEqual(chat.models, [seacode_module.MODEL])

    def test_run_agent_honors_max_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / "a.txt").write_text("a\n")
            (project_root / "b.txt").write_text("b\n")
            (project_root / "c.txt").write_text("c\n")

            scripted = [
                json.dumps({
                    "tool": "read_file",
                    "arguments": {"path": name},
                })
                for name in ("a.txt", "b.txt", "c.txt")
            ]
            chat = RecordingChat(scripted)
            messages = [
                {
                    "role": "system",
                    "content": seacode_module.SYSTEM_PROMPT,
                },
                {"role": "user", "content": "Read everything."},
            ]

            with (
                patch.object(
                    tools_module,
                    "get_project_root",
                    return_value=project_root,
                ),
                patch.object(
                    seacode_module.client, "chat", side_effect=chat
                ),
            ):
                answer = seacode_module.run_agent(
                    messages, max_tool_calls=2
                )

        # Two tools executed, then the run stops without an answer.
        self.assertIsNone(answer)
        self.assertEqual(len(chat.models), 2)


if __name__ == "__main__":
    unittest.main()
