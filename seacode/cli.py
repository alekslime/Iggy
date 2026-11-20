"""Command-line entry point for SeaCode (`seacode`).

Commands:
  seacode              start the chat REPL (default)
  seacode tui          start the full-screen terminal UI
  seacode undo         restore a file from its most recent backup

Global options (--model, --host, --max-tools, --dir) apply to the
REPL and the TUI; `undo` only needs --dir. Each also has an
environment-variable equivalent (SEACODE_MODEL, SEACODE_HOST,
SEACODE_MAX_TOOLS, SEACODE_DIR); the flag wins when both are given.
"""

import argparse
import os
import sys

import ollama

import seacode.agent as seacode_module
from seacode import __version__
from seacode.tools import get_project_root, list_backups, undo_last_change


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seacode",
        description="A local-first CLI AI programming tool.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model",
        default=None,
        help=(
            f"Ollama model to use (default: $SEACODE_MODEL or "
            f"'{seacode_module.DEFAULT_MODEL}')"
        ),
    )
    common.add_argument(
        "--host",
        default=None,
        help=(
            "Ollama host (default: $SEACODE_HOST, $OLLAMA_HOST, or "
            f"'{seacode_module.DEFAULT_HOST}')"
        ),
    )
    common.add_argument(
        "--max-tools",
        type=int,
        default=None,
        help=(
            "Max tool calls per agent turn (default: $SEACODE_MAX_TOOLS "
            f"or {seacode_module.MAX_TOOL_CALLS})"
        ),
    )
    common.add_argument(
        "--dir",
        default=None,
        help=(
            "Project directory to operate on "
            "(default: $SEACODE_DIR or the current directory)"
        ),
    )

    subparsers = parser.add_subparsers(dest="command")

    repl = subparsers.add_parser(
        "repl",
        parents=[common],
        help="Start the chat REPL (the default when no command is given).",
    )
    repl.set_defaults(func=run_repl)

    tui = subparsers.add_parser(
        "tui",
        parents=[common],
        help="Start the full-screen terminal UI.",
    )
    tui.set_defaults(func=run_tui)

    undo = subparsers.add_parser(
        "undo",
        help="Restore a file from its most recent backup.",
    )
    undo.add_argument(
        "path",
        nargs="?",
        default=None,
        help=(
            "File to restore (default: whichever file was most "
            "recently backed up)."
        ),
    )
    undo.add_argument(
        "--list",
        action="store_true",
        help="List available backups instead of restoring.",
    )
    undo.add_argument(
        "--dir",
        default=None,
        help=(
            "Project directory to operate on "
            "(default: $SEACODE_DIR or the current directory)"
        ),
    )
    undo.set_defaults(func=run_undo)

    # Bare `seacode` behaves like `seacode repl`, accepting the same flags.
    # argparse can't attach one shared-flag group to two parser levels
    # at once, so the top-level parser carries its own hidden copies
    # for that case.
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--host", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-tools", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--dir", default=None, help=argparse.SUPPRESS)
    parser.set_defaults(func=run_repl)

    return parser


def apply_common_options(args) -> tuple:
    """Apply --dir/--host and resolve --model/--max-tools.

    Returns (model, max_tool_calls). --dir is exported as SEACODE_DIR
    so the tools pick it up; --host rebuilds the Ollama client.
    """

    if getattr(args, "dir", None):
        os.environ["SEACODE_DIR"] = args.dir

    if getattr(args, "host", None):
        seacode_module.client = ollama.Client(
            host=seacode_module.resolve_host(args.host)
        )

    model = seacode_module.resolve_model(getattr(args, "model", None))
    max_tool_calls = seacode_module.resolve_max_tool_calls(
        getattr(args, "max_tools", None)
    )

    return model, max_tool_calls


def run_repl(args) -> int:
    model, max_tool_calls = apply_common_options(args)
    seacode_module.run_repl(model=model, max_tool_calls=max_tool_calls)
    return 0


def run_tui(args) -> int:
    from seacode.tui import SeaCodeApp

    model, max_tool_calls = apply_common_options(args)
    SeaCodeApp(model=model, max_tool_calls=max_tool_calls).run()
    return 0


def run_undo(args) -> int:
    if args.dir:
        os.environ["SEACODE_DIR"] = args.dir

    if args.list:
        backups = list_backups(args.path)

        if not backups:
            print("No backups found.")
            return 0

        root = get_project_root()
        print(f"Backups in {root} (oldest first):")
        for entry in backups:
            print(f"  {entry['timestamp']}  {entry['path']}")

        return 0

    print(undo_last_change(args.path))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
