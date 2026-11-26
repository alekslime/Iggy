"""Startup trust for project-supplied launch config.

Two files in a project can make SeaCode *launch programs* before the
model has done anything:

  .mcp.json          spawns every configured MCP server
  seacode.toml       [hooks] run around every tool call

Opening a repository you just cloned must not run its author's code
silently, so SeaCode asks first, once per project, and remembers the
answer -- but only for the exact config that was approved. The answer
is bound to a hash of the launch-relevant config, so adding a server or
editing a hook asks again.

Approvals live in a per-user file (``~/.config/seacode/
trusted_projects.json``; override with $SEACODE_TRUST_FILE), never in
the project, so a repository cannot ship its own approval.

``--yes`` / ``--trust all`` deliberately do NOT skip this prompt: they
govern what the *model* may do, while this gate protects against the
*project*. Scripted runs opt in explicitly with ``--trust-project`` or
$SEACODE_TRUST_PROJECT=1. Without a terminal and without that opt-in,
untrusted config is skipped (fail closed) with a warning.
"""

import hashlib
import json
import os
import sys
from pathlib import Path


KIND_MCP = "mcp"
KIND_HOOKS = "hooks"

_TRUTHY = {"1", "true", "yes", "on"}


def trust_store_path() -> Path:
    override = os.environ.get("SEACODE_TRUST_FILE")

    if override:
        return Path(override).expanduser()

    base = os.environ.get("XDG_CONFIG_HOME")
    base_path = Path(base).expanduser() if base else Path.home() / ".config"

    return base_path / "seacode" / "trusted_projects.json"


def fingerprint(payload) -> str:
    """Stable hash of a config payload (key order does not matter)."""

    text = json.dumps(payload, sort_keys=True, default=str)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_store() -> dict:
    try:
        data = json.loads(trust_store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    projects = data.get("projects") if isinstance(data, dict) else None

    return projects if isinstance(projects, dict) else {}


def _save_store(projects: dict) -> None:
    path = trust_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"projects": projects}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _key(project_root) -> str:
    return str(Path(project_root).resolve())


def is_trusted(project_root, kind: str, payload) -> bool:
    entry = _load_store().get(_key(project_root))

    if not isinstance(entry, dict):
        return False

    return entry.get(kind) == fingerprint(payload)


def remember(project_root, kind: str, payload) -> None:
    projects = _load_store()
    entry = projects.get(_key(project_root))

    if not isinstance(entry, dict):
        entry = {}

    entry[kind] = fingerprint(payload)
    projects[_key(project_root)] = entry
    _save_store(projects)


def forget(project_root) -> bool:
    """Drop every stored approval for a project. True if any existed."""

    projects = _load_store()

    if projects.pop(_key(project_root), None) is None:
        return False

    _save_store(projects)

    return True


def env_opt_in() -> bool:
    return os.environ.get("SEACODE_TRUST_PROJECT", "").lower() in _TRUTHY


def describe_mcp(config: dict) -> list[str]:
    """One line per server. Env *values* are never shown (they are
    often secrets); only the variable names are."""

    lines = []

    for name, spec in config.items():
        command = " ".join([spec["command"], *spec.get("args", [])])
        line = f"{name}: {command}"
        env_names = sorted(spec.get("env") or {})

        if env_names:
            line += f"  (env: {', '.join(env_names)})"

        lines.append(line)

    return lines


def describe_hooks(hooks: dict) -> list[str]:
    lines = []

    for event in ("pre_tool", "post_tool"):
        for command in hooks.get(event, []):
            lines.append(f"{event}: {command}")

    return lines


def confirm(
    project_root,
    kind: str,
    payload,
    lines: list[str],
    *,
    opt_in: bool = False,
    interactive: bool | None = None,
    ask=input,
    out=print,
) -> bool:
    """Decide whether this project's launch config may run.

    Returns True when there is nothing to run, the exact config was
    approved before, the caller opted in (``opt_in`` or
    $SEACODE_TRUST_PROJECT), or the user answers y/a. Otherwise False.
    """

    if not payload:
        return True

    if is_trusted(project_root, kind, payload):
        return True

    if opt_in or env_opt_in():
        return True

    label = {
        KIND_MCP: "MCP servers (.mcp.json)",
        KIND_HOOKS: "tool hooks (seacode.toml)",
    }.get(kind, kind)

    if interactive is None:
        interactive = sys.stdin.isatty()

    if not interactive:
        print(
            f"Warning: skipping untrusted {label} in {project_root}. "
            "Run `seacode trust` in a terminal, or pass --trust-project.",
            file=sys.stderr,
        )
        return False

    out(f"This project wants to run {label}:")

    for line in lines:
        out(f"    {line}")

    out("These are programs from the project folder, run as you.")

    try:
        answer = ask(
            "  Run them? [y]es this time / [a]lways for this exact "
            "config / [N]o: "
        ).strip().lower()
    except EOFError:
        answer = ""

    if answer in ("a", "always"):
        remember(project_root, kind, payload)
        return True

    return answer in ("y", "yes")
