"""
Tests for streamed model output (run_agent with stream=True).

Each scripted turn is a list of chunk contents, delivered as
object-form or dict-form chunks like the ollama client yields.
Non-streaming behavior is unchanged (see the M7.x suites).

Scenarios covered:
  1. extract_chunk_content reads object and dict chunks, and
     tolerates chunks without text.
  2. A streamed final answer assembles from deltas, notifies
     on_token per delta, and skips the raw-response dump.
  3. A streamed tool call executes normally.
  4. An empty stream goes through invalid-JSON recovery.

Run with:  python3 -m pytest tests/test_stream.py -v
       or: python3 -m unittest tests.test_stream -v
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import seacode.agent as seacode_module
import seacode.tools as tools_module


def object_chunk(content):
    return SimpleNamespace(message=SimpleNamespace(content=content))


def dict_chunk(content):
    return {"message": {"content": content}}


def tool_call(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def final_answer(answer):
    return json.dumps({"tool": "none", "answer": answer})


def split_into_chunks(text, size=7, form=object_chunk):
    return [
        form(text[i:i + size]) for i in range(0, len(text), size)
    ]


class StreamChat:
    """Scripts whole turns; each turn is a list of chunk objects."""

    def __init__(self, scripted_turns):
        self._turns = [list(turn) for turn in scripted_turns]
        self.calls = []

    def __call__(self, model, messages, stream=False):
        self.calls.append({
            "model": model,
            "stream": stream,
            "messages": [dict(m) for m in messages],
        })
        if not self._turns:
            raise AssertionError(
                "StreamChat ran out of scripted turns."
            )
        return self._turns.pop(0)


class ExtractChunkContentTests(unittest.TestCase):

    def test_object_form(self):
        self.assertEqual(
            seacode_module.extract_chunk_content(object_chunk("hi")),
            "hi",
        )

    def test_dict_form(self):
        self.assertEqual(
            seacode_module.extract_chunk_content(dict_chunk("hi")),
            "hi",
        )

    def test_chunks_without_text(self):
        self.assertEqual(
            seacode_module.extract_chunk_content(object_chunk(None)),
            "",
        )
        self.assertEqual(
            seacode_module.extract_chunk_content({}), ""
        )
        self.assertEqual(
            seacode_module.extract_chunk_content(
                {"message": {"content": None}}
            ),
            "",
        )


class StreamedRunTests(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self._tmpdir.name)

        root_patcher = patch.object(
            tools_module, "get_project_root", return_value=self.project_root
        )
        root_patcher.start()
        self.addCleanup(root_patcher.stop)

        permission_patcher = patch.object(
            tools_module, "request_permission", return_value=True
        )
        permission_patcher.start()
        self.addCleanup(permission_patcher.stop)

    def run_streamed(self, turns, user_request="Hi."):
        chat = StreamChat(turns)
        tokens = []
        output = io.StringIO()
        messages = [
            {"role": "system", "content": seacode_module.SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]

        with (
            patch.object(
                seacode_module.client, "chat", side_effect=chat
            ),
            redirect_stdout(output),
        ):
            answer = seacode_module.run_agent(
                messages, stream=True, on_token=tokens.append
            )

        return answer, chat, tokens, output.getvalue()

    def test_final_answer_assembles_from_deltas(self):
        answer, chat, tokens, output = self.run_streamed([
            split_into_chunks(final_answer("Done.")),
        ])

        self.assertEqual(answer, "Done.")
        self.assertTrue(chat.calls[0]["stream"])
        self.assertEqual("".join(tokens), final_answer("Done."))
        self.assertGreater(len(tokens), 1)

    def test_raw_dump_skipped_while_streaming(self):
        _, _, _, output = self.run_streamed([
            split_into_chunks(final_answer("Done.")),
        ])

        self.assertNotIn("Model raw response", output)

    def test_dict_form_chunks(self):
        answer, _, tokens, _ = self.run_streamed([
            split_into_chunks(
                final_answer("Done."), size=5, form=dict_chunk
            ),
        ])

        self.assertEqual(answer, "Done.")
        self.assertEqual("".join(tokens), final_answer("Done."))

    def test_streamed_tool_call_executes(self):
        (self.project_root / "a.txt").write_text("hello\n")

        answer, chat, _, _ = self.run_streamed(
            [
                split_into_chunks(
                    tool_call("read_file", path="a.txt")
                ),
                split_into_chunks(final_answer("It says hello.")),
            ],
            user_request="What is in a.txt?",
        )

        self.assertEqual(answer, "It says hello.")
        self.assertEqual(len(chat.calls), 2)
        self.assertTrue(all(call["stream"] for call in chat.calls))

    def test_empty_stream_recovers(self):
        answer, chat, tokens, _ = self.run_streamed([
            [],
            split_into_chunks(final_answer("Recovered.")),
        ])

        self.assertEqual(answer, "Recovered.")
        self.assertEqual("".join(tokens), final_answer("Recovered."))
        self.assertEqual(len(chat.calls), 2)


if __name__ == "__main__":
    unittest.main()
