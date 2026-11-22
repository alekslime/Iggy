"""Tests for the eval benchmark, with a scripted model (no live
server). The checkers and runner plumbing are what's tested here;
model quality is what the live command measures."""

import json
import unittest
from types import SimpleNamespace

import seacode.agent as seacode_module
import seacode.evals as evals_module


def fake_response(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def call(name, **arguments):
    return json.dumps({"tool": name, "arguments": arguments})


class ScriptedClient:
    """Chat client returning scripted replies in order."""

    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, model, messages, **kwargs):
        if not self._responses:
            raise AssertionError("ScriptedClient ran out of replies.")
        return fake_response(self._responses.pop(0))


class EvalRunnerTests(unittest.TestCase):

    def test_typo_fix_task_passes_with_scripted_model(self):
        client = ScriptedClient([
            call("read_file", path="hello.py"),
            call("replace_in_file", path="hello.py",
                 old_text="helllo", new_text="hello"),
            "Fixed it.",
        ])
        results = evals_module.run_eval(
            model="scripted:3b", client=client,
            task_ids=["typo-fix"],
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["passed"])

    def test_task_fails_when_model_does_nothing(self):
        client = ScriptedClient(["I could fix that for you."])
        # M7.15 rejects the no-work answer twice, then it is
        # accepted; either way no edit happened, so it must FAIL.
        client._responses.extend([
            "Let me look at that.", "Still thinking about it.",
        ])
        results = evals_module.run_eval(
            model="scripted:3b", client=client,
            task_ids=["typo-fix"],
        )
        self.assertFalse(results[0]["passed"])

    def test_explain_task_checks_answer(self):
        client = ScriptedClient([
            call("read_file", path="hello.py"),
            "It prints 42.",
        ])
        results = evals_module.run_eval(
            model="scripted:3b", client=client, task_ids=["explain"],
        )
        self.assertTrue(results[0]["passed"])

    def test_unknown_task_id_runs_nothing(self):
        results = evals_module.run_eval(
            model="scripted:3b", client=ScriptedClient([]),
            task_ids=["nope"],
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
