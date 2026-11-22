"""Tests for the compact system-prompt profile (small local models)."""

import unittest

import seacode.agent as seacode_module


class PromptProfileTests(unittest.TestCase):

    def test_size_parsing(self):
        self.assertEqual(
            seacode_module.model_size_billions("qwen2.5-coder:3b"), 3.0
        )
        self.assertEqual(
            seacode_module.model_size_billions("llama3.1:70b"), 70.0
        )
        self.assertEqual(
            seacode_module.model_size_billions("phi3:3.8b"), 3.8
        )
        self.assertIsNone(
            seacode_module.model_size_billions("llama3.2:latest")
        )
        self.assertIsNone(seacode_module.model_size_billions(None))

    def test_small_models_get_small_prompt(self):
        prompt = seacode_module.system_prompt_for_model(
            "qwen2.5-coder:3b"
        )
        self.assertIs(prompt, seacode_module.SYSTEM_PROMPT_SMALL)
        prompt = seacode_module.system_prompt_for_model("codellama:7b")
        self.assertIs(prompt, seacode_module.SYSTEM_PROMPT_SMALL)

    def test_large_and_unknown_get_full_prompt(self):
        self.assertIs(
            seacode_module.system_prompt_for_model("qwen2.5-coder:32b"),
            seacode_module.SYSTEM_PROMPT,
        )
        self.assertIs(
            seacode_module.system_prompt_for_model("mystery-model"),
            seacode_module.SYSTEM_PROMPT,
        )

    def test_small_prompt_is_much_shorter_and_keeps_key_rules(self):
        small = seacode_module.SYSTEM_PROMPT_SMALL
        self.assertLess(len(small), len(seacode_module.SYSTEM_PROMPT) // 2)
        for phrase in ("Read a file before modifying", "Never claim",
                       '{"name": "tool_name"'):
            self.assertIn(phrase, small)


if __name__ == "__main__":
    unittest.main()
