"""User-configured hooks around tool execution.

Hooks are shell-free commands declared in the project's seacode.toml:

    [hooks]
    pre_tool = ["python scripts/guard.py"]
    post_tool = ["python scripts/log.py"]

(pre_tool_use / post_tool_use are accepted as aliases.)

Each hook receives a JSON payload on stdin:

    {"event": "pre_tool", "tool": "write_file",
     "arguments": {...}, "result": null}

For post_tool, "result" carries the tool's return value (stringified
when it isn't JSON-shaped). A pre_tool hook that exits non-zero
*blocks* the tool call: the model is told the hook blocked it and why
(the hook's stdout/stderr, trimmed). Post-tool hooks can't block --
their failures are only printed as warnings. Hooks are the user's
own configuration, so their commands are not subject to the
run_command allowlist; they run with shell=False and a timeout.
"""

import json
import shlex
import subprocess
import tomllib
from pathlib import Path


HOOK_TIMEOUT_SECONDS = 15

_hooks = {"pre_tool": [], "post_tool": []}


def load_hooks(project_root) -> dict:
    """Read the [hooks] table of seacode.toml ({} when absent/invalid
    in shape -- wrong-typed entries are ignored, not fatal)."""

    path = Path(project_root) / "seacode.toml"

    if not path.is_file():
        return {"pre_tool": [], "post_tool": []}

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {"pre_tool": [], "post_tool": []}

    table = data.get("hooks", {})

    if not isinstance(table, dict):
        return {"pre_tool": [], "post_tool": []}

    loaded = {"pre_tool": [], "post_tool": []}

    for event, aliases in (
        ("pre_tool", ("pre_tool", "pre_tool_use", "PreToolUse")),
        ("post_tool", ("post_tool", "post_tool_use", "PostToolUse")),
    ):
        for alias in aliases:
            commands = table.get(alias)

            if isinstance(commands, str):
                commands = [commands]

            if isinstance(commands, list):
                loaded[event].extend(
                    c for c in commands
                    if isinstance(c, str) and c.strip()
                )

    return loaded


def set_hooks(hooks) -> None:
    """Install hooks for this process (the CLI startup path)."""

    global _hooks
    _hooks = {
        "pre_tool": list((hooks or {}).get("pre_tool", [])),
        "post_tool": list((hooks or {}).get("post_tool", [])),
    }


def clear_hooks() -> None:
    global _hooks
    _hooks = {"pre_tool": [], "post_tool": []}


def get_hooks() -> dict:
    return {event: list(cmds) for event, cmds in _hooks.items()}


def _run_one(command, payload, project_root):
    args = shlex.split(command)

    if not args:
        return 0, ""

    result = subprocess.run(
        args,
        cwd=project_root,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=HOOK_TIMEOUT_SECONDS,
        check=False,
        shell=False,
    )

    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output.strip()


def run_pre_tool_hooks(tool_name, arguments, project_root):
    """Run pre-tool hooks. Returns None to allow the call, or a
    blocking message (the tool must not run) when a hook exits
    non-zero or can't run at all."""

    for command in _hooks["pre_tool"]:
        payload = {
            "event": "pre_tool",
            "tool": tool_name,
            "arguments": arguments if isinstance(arguments, dict) else {},
            "result": None,
        }

        try:
            code, output = _run_one(command, payload, project_root)
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"Pre-tool hook `{command}` could not run: {e}"

        if code != 0:
            detail = output[:500] if output else f"exit code {code}"
            return f"Pre-tool hook `{command}` blocked this call: {detail}"

    return None


def run_post_tool_hooks(tool_name, arguments, result, project_root) -> None:
    """Run post-tool hooks. Never blocks; failures print a warning."""

    for command in _hooks["post_tool"]:
        payload = {
            "event": "post_tool",
            "tool": tool_name,
            "arguments": arguments if isinstance(arguments, dict) else {},
            "result": result if isinstance(result, (str, dict, list)) else str(result),
        }

        try:
            code, output = _run_one(command, payload, project_root)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"  (post-tool hook `{command}` failed: {e})")
            continue

        if code != 0:
            detail = output[:200] if output else f"exit code {code}"
            print(f"  (post-tool hook `{command}` failed: {detail})")
