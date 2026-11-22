"""
Tests for model providers (seacode.providers) and their CLI wiring.

Covers provider/api-key/base-url resolution precedence, the client
factory, the OpenAI-compatible client's request/response translation
(via httpx MockTransport -- no network), and that `seacode --provider
openai` actually swaps the agent's client.

Run with:  python3 -m pytest tests/test_providers.py -v
       or: python3 -m unittest tests.test_providers -v
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import seacode.agent as seacode_module
import seacode.cli as cli_module
import seacode.providers as providers_module
from seacode.config import load_project_config


class ResolveProviderTests(unittest.TestCase):

    def test_default_is_ollama(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_PROVIDER", None)
            self.assertEqual(providers_module.resolve_provider(), "ollama")

    def test_config_then_env_then_cli(self):
        config = {"provider": "openai"}
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_PROVIDER", None)
            self.assertEqual(
                providers_module.resolve_provider(None, config), "openai"
            )
        with patch.dict(os.environ, {"SEACODE_PROVIDER": "openai"}):
            self.assertEqual(
                providers_module.resolve_provider(None, {}), "openai"
            )
            self.assertEqual(
                providers_module.resolve_provider("ollama", config), "ollama"
            )

    def test_unknown_provider_fails_fast(self):
        with self.assertRaises(ValueError):
            providers_module.resolve_provider("bogus")
        with patch.dict(os.environ, {"SEACODE_PROVIDER": "bogus"}):
            with self.assertRaises(ValueError):
                providers_module.resolve_provider()


class ResolveApiKeyTests(unittest.TestCase):

    def test_precedence(self):
        config = {"api_key": "config-key"}
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertEqual(
                providers_module.resolve_api_key(None, config), "config-key"
            )
            self.assertIsNone(providers_module.resolve_api_key(None, {}))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}):
            os.environ.pop("SEACODE_API_KEY", None)
            self.assertEqual(
                providers_module.resolve_api_key(None, config), "openai-key"
            )
        with patch.dict(os.environ, {"SEACODE_API_KEY": "sea-key"}):
            self.assertEqual(
                providers_module.resolve_api_key("cli-key", config), "cli-key"
            )
            self.assertEqual(
                providers_module.resolve_api_key(None, config), "sea-key"
            )


class ResolveBaseUrlTests(unittest.TestCase):

    def test_defaults_per_provider(self):
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_HOST", None)
            os.environ.pop("OLLAMA_HOST", None)
            self.assertEqual(
                providers_module.resolve_base_url(None, {}, "ollama"),
                providers_module.OLLAMA_DEFAULT_HOST,
            )
            self.assertEqual(
                providers_module.resolve_base_url(None, {}, "openai"),
                providers_module.OPENAI_DEFAULT_HOST,
            )

    def test_ollama_host_env_only_applies_to_ollama(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://ollama:1"}):
            os.environ.pop("SEACODE_HOST", None)
            self.assertEqual(
                providers_module.resolve_base_url(None, {}, "ollama"),
                "http://ollama:1",
            )
            self.assertEqual(
                providers_module.resolve_base_url(None, {}, "openai"),
                providers_module.OPENAI_DEFAULT_HOST,
            )

    def test_cli_and_config(self):
        config = {"host": "http://config:1"}
        with patch.dict(os.environ):
            os.environ.pop("SEACODE_HOST", None)
            os.environ.pop("OLLAMA_HOST", None)
            self.assertEqual(
                providers_module.resolve_base_url(None, config, "openai"),
                "http://config:1",
            )
            self.assertEqual(
                providers_module.resolve_base_url(
                    "http://cli:1", config, "openai"
                ),
                "http://cli:1",
            )


class BuildClientTests(unittest.TestCase):

    def test_ollama_client(self):
        with patch("ollama.Client") as mock_client:
            client = providers_module.build_client(
                "ollama", host="http://example:11434"
            )
            mock_client.assert_called_once_with(host="http://example:11434")
            self.assertIs(client, mock_client.return_value)

    def test_openai_client(self):
        client = providers_module.build_client(
            "openai", host="http://local:1234/v1/", api_key="k"
        )
        self.assertIsInstance(client, providers_module.OpenAIClient)
        self.assertEqual(client.base_url, "http://local:1234/v1")
        self.assertEqual(client.api_key, "k")


def mock_transport(handler):
    return httpx.MockTransport(handler)


class OpenAIClientTests(unittest.TestCase):

    def test_chat_translates_tool_calls(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.py"}',
                        },
                    }],
                }}],
            })

        client = providers_module.OpenAIClient(
            "http://api.test/v1", api_key="secret",
            transport=mock_transport(handler),
        )
        result = client.chat(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=providers_module.TOOL_SCHEMAS,
        )

        self.assertEqual(seen["url"], "http://api.test/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer secret")
        self.assertEqual(seen["body"]["tools"], providers_module.TOOL_SCHEMAS)
        message = result["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"],
                         "read_file")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"],
                         {"path": "a.py"})

    def test_chat_unparseable_arguments_kept_raw(self):
        def handler(request):
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "content": "x",
                    "tool_calls": [{
                        "id": "c",
                        "function": {"name": "t", "arguments": "{nope"},
                    }],
                }}],
            })

        client = providers_module.OpenAIClient(
            "http://api.test/v1", transport=mock_transport(handler)
        )
        call = client.chat(model="m", messages=[])["message"]["tool_calls"][0]
        self.assertIsNone(call["function"]["arguments"])
        self.assertEqual(call["function"]["raw"], "{nope")

    def test_chat_http_error_raises(self):
        def handler(request):
            return httpx.Response(401, json={
                "error": {"message": "bad key"}
            })

        client = providers_module.OpenAIClient(
            "http://api.test/v1", transport=mock_transport(handler)
        )
        with self.assertRaises(RuntimeError) as ctx:
            client.chat(model="m", messages=[])
        self.assertIn("401", str(ctx.exception))
        self.assertIn("bad key", str(ctx.exception))

    def test_stream_assembles_text_and_tool_calls(self):
        def sse(payload):
            return f"data: {json.dumps(payload)}\n\n"

        body = (
            sse({"choices": [{"delta": {"content": "Hel"}}]})
            + sse({"choices": [{"delta": {"content": "lo"}}]})
            + sse({"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "c1",
                "function": {"name": "read_file", "arguments": '{"pa'},
            }]}}]})
            + sse({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": 'th": "a.py"}'},
            }]}}]})
            + "data: [DONE]\n\n"
        )

        def handler(request):
            return httpx.Response(
                200, content=body,
                headers={"Content-Type": "text/event-stream"},
            )

        client = providers_module.OpenAIClient(
            "http://api.test/v1", transport=mock_transport(handler)
        )
        chunks = list(client.chat(model="m", messages=[], stream=True))
        text = "".join(c["message"].get("content", "") for c in chunks)
        self.assertEqual(text, "Hello")
        final = chunks[-1]["message"]
        self.assertEqual(
            final["tool_calls"][0]["function"]["arguments"],
            {"path": "a.py"},
        )

    def test_to_openai_message_serializes_arguments(self):
        message = providers_module.to_openai_message({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c",
                "type": "function",
                "function": {"name": "t", "arguments": {"a": 1}},
            }],
        })
        self.assertEqual(message["content"], "")
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"], '{"a": 1}'
        )


class ProviderConfigAndCliTests(unittest.TestCase):

    def test_config_loads_provider_keys(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "seacode.toml").write_text(
                '[seacode]\nprovider = "openai"\napi_key = "k"\n',
                encoding="utf-8",
            )
            config = load_project_config(tmpdir)
        self.assertEqual(config["provider"], "openai")
        self.assertEqual(config["api_key"], "k")

    def test_parser_provider_flags(self):
        args = cli_module.build_parser().parse_args(
            ["repl", "--provider", "openai", "--api-key", "k"]
        )
        self.assertEqual(args.provider, "openai")
        self.assertEqual(args.api_key, "k")

        args = cli_module.build_parser().parse_args(["--provider", "openai"])
        self.assertEqual(args.provider, "openai")

    def test_apply_common_options_builds_openai_client(self):
        from seacode import permissions as permissions_module
        import seacode.context as context_module

        self.addCleanup(permissions_module.reset_trust_level)
        self.addCleanup(context_module.reset_context_limit)
        args = SimpleNamespace(
            model=None, host="http://local:9/v1", max_tools=None,
            trust=None, yes=False, dir=None, resume=None,
            no_stream=False, verbose=False,
            provider="openai", api_key="k",
        )
        original_client = seacode_module.client
        try:
            with patch.dict(os.environ):
                os.environ.pop("SEACODE_PROVIDER", None)
                cli_module.apply_common_options(args)
                self.assertIsInstance(
                    seacode_module.client, providers_module.OpenAIClient
                )
                self.assertEqual(
                    seacode_module.client.base_url, "http://local:9/v1"
                )
        finally:
            seacode_module.client = original_client


if __name__ == "__main__":
    unittest.main()
