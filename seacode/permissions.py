"""Permission policy for SeaCode's gated actions.

Every file modification and command execution goes through
request_permission(). What it does depends on the trust level:

  ask    prompt for everything (the default)
  edits  auto-approve file changes, still prompt for commands
  all    auto-approve everything (the `--yes` flag)

Precedence: set_trust_level() (the CLI flag) wins, then SEACODE_TRUST,
then the project config's `trust` key, then "ask". Auto-approvals
are always printed so a run never silently stops asking.

Action kinds are recognized by the message prefix the tools use
("modify file:", "delete file:", "restore file from backup:",
"run:"). Anything unrecognized fails closed to a prompt.
"""

import os


TRUST_LEVELS = ("ask", "edits", "all")

EDIT_ACTION_PREFIXES = (
    "modify file:",
    "delete file:",
    "restore file from backup:",
)

COMMAND_ACTION_PREFIXES = ("run:",)

_trust_level = None


def set_trust_level(level: str) -> None:
    """Pin the trust level explicitly (the CLI flag path)."""

    if level not in TRUST_LEVELS:
        raise ValueError(
            f"Unknown trust level: {level!r} "
            f"(expected one of {', '.join(TRUST_LEVELS)})"
        )

    global _trust_level
    _trust_level = level


def reset_trust_level() -> None:
    """Clear the explicit override (used by tests)."""

    global _trust_level
    _trust_level = None


def get_trust_level(config=None) -> str:
    """Return the effective trust level."""

    if _trust_level is not None:
        return _trust_level

    env_level = os.environ.get("SEACODE_TRUST")

    if env_level:
        if env_level not in TRUST_LEVELS:
            raise ValueError(
                f"Unknown SEACODE_TRUST value: {env_level!r} "
                f"(expected one of {', '.join(TRUST_LEVELS)})"
            )
        return env_level

    if config is not None:
        config_level = config.get("trust")

        if config_level is not None:
            if config_level not in TRUST_LEVELS:
                raise ValueError(
                    f"Unknown seacode.toml trust value: "
                    f"{config_level!r} (expected one of "
                    f"{', '.join(TRUST_LEVELS)})"
                )
            return config_level

    return "ask"


def action_kind(action: str) -> str:
    """Classify a permission prompt: "edit", "command", or "other"."""

    if action.startswith(EDIT_ACTION_PREFIXES):
        return "edit"

    if action.startswith(COMMAND_ACTION_PREFIXES):
        return "command"

    return "other"


def should_auto_approve(action: str, config=None) -> bool:
    """Check whether the trust level approves without prompting."""

    level = get_trust_level(config)

    if level == "all":
        return True

    if level == "edits" and action_kind(action) == "edit":
        return True

    return False


def request_permission(action: str, diff: str | None = None) -> bool:
    """Ask the user to approve an action, honoring the trust level.

    Prints `? action` and prompts for a y/N confirmation, returning
    True only on an explicit "y" -- unless the trust level
    auto-approves it, in which case a `✓` line is printed and no
    prompt is shown.

    When `diff` is given (file modifications pass their proposed
    unified diff), it is printed *before* the prompt so the user
    approves the actual change, not just a filename. Auto-approved
    actions show it too, so the transcript still records what landed.
    """

    if should_auto_approve(action):
        print(f"  ✓ {action} (auto-approved, trust={get_trust_level()})")
        if diff:
            _print_diff(diff)
        return True

    print(f"  ? {action}")

    if diff:
        _print_diff(diff)

    response = input("  Proceed? [y/N]: ").strip().lower()

    return response == "y"


MAX_DIFF_LINES = 60


def _print_diff(diff: str) -> None:
    lines = diff.splitlines()

    for line in lines[:MAX_DIFF_LINES]:
        print(f"    {line}")

    hidden = len(lines) - MAX_DIFF_LINES

    if hidden > 0:
        print(f"    … (+{hidden} more diff lines)")
