"""A tiny live benchmark for local models.

Four canned tasks, each in a throwaway project, run through the
real agent loop with trust=all. It answers the question mocked
tests can't: which of the models you actually have pulled, with
which SeaCode version, completes real work end-to-end?

    seacode eval                      # default model
    seacode eval --model qwen2.5-coder:7b
    seacode eval --task typo-fix      # one task

Scoring is mechanical (file contents / answer text), never a
model judging itself. Expect minutes, not seconds, on small
hardware: every task is several full model turns.
"""

import os
import tempfile
import time
from pathlib import Path

import seacode.agent as agent_module
from seacode import permissions as permissions_module
from seacode.tools import get_project_root


def _check_typo_fix(root, answer):
    text = (Path(root) / "hello.py").read_text(encoding="utf-8")
    return "hello world" in text and "helllo" not in text


def _check_failing_test(root, answer):
    calc = (Path(root) / "calc.py").read_text(encoding="utf-8")
    test = (Path(root) / "test_calc.py").read_text(encoding="utf-8")
    return "a + b" in calc and "assert add(2, 3) == 5" in test


def _check_create_file(root, answer):
    path = Path(root) / "greet.py"

    if not path.is_file():
        return False

    text = path.read_text(encoding="utf-8")
    return "def greet" in text and "hi" in text


def _check_explain(root, answer):
    return bool(answer) and "42" in answer


TASKS = [
    {
        "id": "typo-fix",
        "title": "Fix a typo in hello.py",
        "files": {
            "hello.py": 'def greet():\n    return "helllo world"\n',
        },
        "prompt": "In hello.py, change `helllo` to `hello`.",
        "check": _check_typo_fix,
    },
    {
        "id": "failing-test",
        "title": "Make the failing test pass",
        "files": {
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": (
                "from calc import add\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
            ),
        },
        "prompt": (
            "The test in test_calc.py fails. Fix calc.py so the "
            "test passes, and run the test to check."
        ),
        "check": _check_failing_test,
    },
    {
        "id": "create-file",
        "title": "Create greet.py from scratch",
        "files": {},
        "prompt": (
            "Create greet.py with a function greet() that "
            "returns the string \"hi\"."
        ),
        "check": _check_create_file,
    },
    {
        "id": "explain",
        "title": "Read and explain hello.py",
        "files": {"hello.py": "x = 41 + 1\nprint(x)\n"},
        "prompt": "Read hello.py and tell me what value it prints.",
        "check": _check_explain,
    },
]


def run_eval(model=None, client=None, task_ids=None,
             max_tool_calls=None):
    """Run the benchmark, returning [{id, title, passed, seconds}].

    `client` lets tests inject a scripted chat client; by default
    the agent module's current client is used (the CLI builds the
    provider client first, including num_ctx options).
    """

    selected = [
        task for task in TASKS
        if task_ids is None or task["id"] in task_ids
    ]

    model = model or agent_module.MODEL
    results = []

    previous_client = agent_module.client
    previous_dir = os.environ.get("SEACODE_DIR")

    if client is not None:
        agent_module.client = client

    permissions_module.set_trust_level("all")

    try:
        for task in selected:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)

                for name, content in task["files"].items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")

                os.environ["SEACODE_DIR"] = str(root)

                messages = [
                    {
                        "role": "system",
                        "content": agent_module.build_system_prompt(
                            agent_module.system_prompt_for_model(model),
                            root,
                        ),
                    },
                    {"role": "user", "content": task["prompt"]},
                ]

                started = time.monotonic()

                try:
                    answer = agent_module.run_agent(
                        messages,
                        model=model,
                        max_tool_calls=max_tool_calls,
                    )
                except Exception:
                    answer = None

                passed = False

                try:
                    passed = bool(task["check"](root, answer))
                except Exception:
                    passed = False

                results.append({
                    "id": task["id"],
                    "title": task["title"],
                    "passed": passed,
                    "seconds": round(time.monotonic() - started, 1),
                })
    finally:
        agent_module.client = previous_client
        permissions_module.reset_trust_level()

        if previous_dir is None:
            os.environ.pop("SEACODE_DIR", None)
        else:
            os.environ["SEACODE_DIR"] = previous_dir

    return results
