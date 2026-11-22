"""Tests for model management: profiles, `seacode models`, /model."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.cli as cli_module
from seacode.config import load_model_profiles


class ProfileTests(unittest.TestCase):

    def test_profiles_loaded_and_validated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "seacode.toml").write_text(
                '[models."qwen2.5-coder:3b"]\n'
                "num_ctx = 8192\ntemperature = 0.2\n"
                '[models."bad"]\nnum_ctx = -5\ntemperature = 9\n',
                encoding="utf-8",
            )
            profiles = load_model_profiles(tmpdir)

        self.assertEqual(
            profiles["qwen2.5-coder:3b"],
            {"num_ctx": 8192, "temperature": 0.2},
        )
        self.assertNotIn("bad", profiles)

    def test_no_file_no_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(load_model_profiles(tmpdir), {})


class ModelsCommandTests(unittest.TestCase):

    def test_parser(self):
        args = cli_module.build_parser().parse_args(["models"])
        self.assertIs(args.func, cli_module.run_models)
        args = cli_module.build_parser().parse_args(
            ["models", "--pull", "qwen2.5-coder:7b"]
        )
        self.assertEqual(args.pull, "qwen2.5-coder:7b")

    def test_format_model_list(self):
        models = [
            SimpleNamespace(model="a:3b", size=2_000_000_000),
            {"model": "b:7b", "size": 4_500_000_000},
        ]
        lines = cli_module.format_model_list(models)
        self.assertEqual(lines[0], "  a:3b  (2.0 GB)")
        self.assertEqual(lines[1], "  b:7b  (4.5 GB)")

    def test_run_models_lists(self):
        import io
        from contextlib import redirect_stdout

        fake = SimpleNamespace(
            models=[SimpleNamespace(model="m:3b", size=1_900_000_000)]
        )

        with patch("ollama.Client") as mock_client:
            mock_client.return_value.list.return_value = fake
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli_module.run_models(
                    SimpleNamespace(pull=None, host=None)
                )

        self.assertEqual(code, 0)
        self.assertIn("m:3b", out.getvalue())

    def test_run_models_server_down(self):
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.list.side_effect = RuntimeError(
                "connection refused"
            )
            code = cli_module.run_models(
                SimpleNamespace(pull=None, host=None)
            )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
