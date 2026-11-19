"""
Headless tests for the TUI (agent.tui).

Constructing the app without running it is enough to cover model
selection: the configured model must reach the status pills and the
agent turns. Interactive behavior is intentionally not covered here.

Run with:  python3 -m pytest tests/test_tui.py -v
       or: python3 -m unittest tests.test_tui -v
"""

import unittest

import agent.agent as agent_module
from agent.tui import IggyApp


class TuiModelTests(unittest.TestCase):

    def test_default_model_matches_agent(self):
        app = IggyApp()

        self.assertEqual(app._model, agent_module.MODEL)
        self.assertIsNone(app._max_tool_calls)

    def test_custom_model_reaches_status_pills(self):
        app = IggyApp(model="custom-model:tag", max_tool_calls=25)

        self.assertEqual(app._model, "custom-model:tag")
        self.assertEqual(app._max_tool_calls, 25)
        self.assertIn("custom-model:tag", app._status_pills().plain)


if __name__ == "__main__":
    unittest.main()
