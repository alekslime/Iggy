"""Command-line entry point for SeaCode (`seacode`).

Commands:
  seacode              start the chat REPL (default)
  seacode tui          start the full-screen terminal UI
  seacode undo         restore a file from its most recent backup
  seacode sessions     list saved conversation sessions
  seacode init         write a starter seacode.toml in the project

Global options (--model, --host, --max-tools, --trust, --dir) apply
to the REPL and the TUI; `undo`/`sessions` only need --dir. Each
also has an environment-variable equivalent (SEACODE_MODEL, SEACODE_HOST,
SEACODE_MAX_TOOLS, SEACODE_TRUST, SEACODE_DIR) and an seacode.toml key; the flag
wins, then env, then config file, then the default.
"""

import argparse
import os
import sys

import seacode.agent as seacode_module
import seacode.context as context_module
import seacode.hooks as hooks_module
import seacode.mcp as mcp_module
from seacode import __version__
from seacode import permissions as permissions_module
from seacode import project_trust as project_trust_module
from seacode import providers as providers_module
from seacode.config import (
    load_model_profiles,
    load_project_config,
    write_default_config,
)
from seacode.sessions import list_sessions, load_session, turns_label
from seacode.tools import (
    get_project_root,
    list_backups,
    set_extra_commands,
    undo_last_change,
)


def add_common_arguments(parser):
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Model to use (default: $SEACODE_MODEL, seacode.toml, "
            f"or '{seacode_module.DEFAULT_MODEL}')"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=providers_module.PROVIDERS,
        default=None,
        help=(
            "Model provider: ollama (local, default) or openai (any "
            "OpenAI-compatible endpoint) ($SEACODE_PROVIDER, seacode.toml)"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for --provider openai ($SEACODE_API_KEY or "
            "$OPENAI_API_KEY, seacode.toml). Local servers need none."
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Provider endpoint (default: $SEACODE_HOST, $OLLAMA_HOST for "
            "ollama, seacode.toml, or the provider default)"
        ),
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=None,
        help=(
            "Max tool calls per agent turn (default: $SEACODE_MAX_TOOLS, "
            f"seacode.toml, or {seacode_module.MAX_TOOL_CALLS})"
        ),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help=(
            "Context window requested from Ollama (default: "
            "$SEACODE_NUM_CTX, seacode.toml, or 16384; the runtime "
            "default of 4096 is too small for the system prompt)"
        ),
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=None,
        help=(
            "Context budget in estimated tokens; older history is "
            "trimmed from the wire copy when exceeded (default: "
            "$SEACODE_CONTEXT_TOKENS, seacode.toml, or 16000)"
        ),
    )
    parser.add_argument(
        "--trust",
        choices=permissions_module.TRUST_LEVELS,
        default=None,
        help=(
            "Permission policy: ask everything (default), auto-approve "
            "file edits, or auto-approve all ($SEACODE_TRUST, seacode.toml)"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Shorthand for --trust all.",
    )
    parser.add_argument(
        "--trust-project",
        action="store_true",
        help=(
            "Run this project's .mcp.json servers and seacode.toml hooks "
            "without asking (for scripts; $SEACODE_TRUST_PROJECT=1). "
            "--yes does not imply this."
        ),
    )
    parser.add_argument(
        "--dir",
        default=None,
        help=(
            "Project directory to operate on "
            "(default: $SEACODE_DIR or the current directory)"
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="SESSION",
        help=(
            "Resume a saved session: the latest, or the one matching "
            "this id prefix."
        ),
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for full replies instead of streaming tokens.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Dump each raw model response (debugging).",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Disable splash screen animation.",
    )


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
    add_common_arguments(common)

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

    sessions = subparsers.add_parser(
        "sessions",
        help="List saved conversation sessions.",
    )
    sessions.add_argument(
        "--dir",
        default=None,
        help=(
            "Project directory to operate on "
            "(default: $SEACODE_DIR or the current directory)"
        ),
    )
    sessions.set_defaults(func=run_sessions)

    eval_cmd = subparsers.add_parser(
        "eval",
        help="Benchmark a local model on canned tasks (live).",
    )
    eval_cmd.add_argument("--model", default=None)
    eval_cmd.add_argument(
        "--task", default=None, metavar="ID",
        help="Run only this task id (see: seacode eval --task x).",
    )
    eval_cmd.add_argument(
        "--provider", choices=providers_module.PROVIDERS, default=None
    )
    eval_cmd.add_argument("--host", default=None)
    eval_cmd.add_argument("--api-key", default=None)
    eval_cmd.add_argument("--num-ctx", type=int, default=None)
    eval_cmd.set_defaults(func=run_eval_command)

    models_cmd = subparsers.add_parser(
        "models",
        help="List local (Ollama) models, or pull one with --pull.",
    )
    models_cmd.add_argument(
        "--pull",
        metavar="MODEL",
        default=None,
        help="Pull this model first, then list.",
    )
    models_cmd.add_argument(
        "--host",
        default=None,
        help="Ollama host (default: $SEACODE_HOST / $OLLAMA_HOST).",
    )
    models_cmd.set_defaults(func=run_models)

    mcp = subparsers.add_parser(
        "mcp",
        help="List MCP servers and tools from .mcp.json.",
    )
    mcp.add_argument(
        "--dir",
        default=None,
        help=(
            "Project directory to operate on "
            "(default: $SEACODE_DIR or the current directory)"
        ),
    )
    mcp.add_argument(
        "--trust-project",
        action="store_true",
        help="Start the servers without asking.",
    )
    mcp.set_defaults(func=run_mcp)

    trust_cmd = subparsers.add_parser(
        "trust",
        help=(
            "Approve (or with --revoke, forget) this project's "
            ".mcp.json servers and seacode.toml hooks."
        ),
    )
    trust_cmd.add_argument(
        "--dir",
        default=None,
        help=(
            "Project directory to operate on "
            "(default: $SEACODE_DIR or the current directory)"
        ),
    )
    trust_cmd.add_argument(
        "--revoke",
        action="store_true",
        help="Forget stored approvals for this project.",
    )
    trust_cmd.set_defaults(func=run_trust)

    init = subparsers.add_parser(
        "init",
        help="Write a starter seacode.toml in the project.",
    )
    init.add_argument(
        "--dir",
        default=None,
        help=(
            "Project directory to operate on "
            "(default: $SEACODE_DIR or the current directory)"
        ),
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing seacode.toml.",
    )
    init.set_defaults(func=run_init)

    # Bare `seacode` behaves like `seacode repl`, accepting the same flags.
    # argparse can't attach one shared-flag group to two parser levels
    # at once, so the top-level parser carries its own hidden copies
    # for that case.
    hidden = argparse.ArgumentParser(add_help=False)
    add_common_arguments(hidden)

    for action in hidden._actions:
        if action.dest == "help":
            continue
        kwargs = {
            "default": action.default,
            "help": argparse.SUPPRESS,
        }
        if action.const is not None or action.nargs is not None:
            kwargs["nargs"] = action.nargs
            kwargs["const"] = action.const
            kwargs["metavar"] = action.metavar
        if isinstance(action, argparse._StoreTrueAction):
            parser.add_argument(
                *action.option_strings,
                action="store_true",
                help=argparse.SUPPRESS,
            )
        elif action.type is not None:
            parser.add_argument(
                *action.option_strings,
                type=action.type,
                **kwargs,
            )
        elif action.choices is not None:
            parser.add_argument(
                *action.option_strings,
                choices=action.choices,
                **kwargs,
            )
        else:
            parser.add_argument(*action.option_strings, **kwargs)

    parser.set_defaults(func=run_repl)

    return parser


_mcp_manager = None


def _mcp_trusted(root, opt_in=False) -> bool:
    """Ask before spawning the project's .mcp.json servers."""

    config = mcp_module.load_mcp_config(root)

    return project_trust_module.confirm(
        root,
        project_trust_module.KIND_MCP,
        config,
        project_trust_module.describe_mcp(config),
        opt_in=opt_in,
    )


def _hooks_trusted(root, hooks, opt_in=False) -> bool:
    """Ask before installing the project's seacode.toml hooks."""

    payload = {k: v for k, v in hooks.items() if v}

    return project_trust_module.confirm(
        root,
        project_trust_module.KIND_HOOKS,
        payload,
        project_trust_module.describe_hooks(hooks),
        opt_in=opt_in,
    )


def _start_mcp(root, opt_in=False) -> None:
    """(Re)start MCP servers for this project, stopping any previous
    manager so repeated apply_common_options calls don't leak
    subprocesses. Untrusted config is not started."""

    global _mcp_manager

    _stop_mcp()

    if not _mcp_trusted(root, opt_in):
        return

    _mcp_manager = mcp_module.MCPManager()
    _mcp_manager.start_all(root)


def _stop_mcp() -> None:
    global _mcp_manager

    if _mcp_manager is not None:
        _mcp_manager.stop_all()
        _mcp_manager = None


def apply_common_options(args) -> tuple:
    """Apply startup options; return (model, max_tools, config, stream,
    verbose).

    Applies --dir first (everything else is relative to the project),
    loads seacode.toml, then resolves --model/--host/--max-tools and
    applies --trust/--yes plus the config's extra commands. Raises
    ValueError on invalid trust levels or command names so a bad
    config fails fast instead of mid-run.
    """

    if getattr(args, "dir", None):
        os.environ["SEACODE_DIR"] = args.dir

    root = get_project_root()
    config = load_project_config(root)

    if getattr(args, "yes", False):
        permissions_module.set_trust_level("all")
    elif getattr(args, "trust", None):
        permissions_module.set_trust_level(args.trust)

    # Pin the effective level (flag > env > seacode.toml > ask) so every
    # permission prompt during the run sees it, and fail fast on a
    # bad SEACODE_TRUST / seacode.toml value.
    permissions_module.set_trust_level(
        permissions_module.get_trust_level(config)
    )

    set_extra_commands(config.get("extra_commands"))

    # Build the provider client for this run (ollama by default). This
    # replaces the agent module's import-time client, which tests and
    # the CLI both already treat as the patchable global.
    provider = providers_module.resolve_provider(
        getattr(args, "provider", None), config
    )
    host = providers_module.resolve_base_url(
        getattr(args, "host", None), config, provider
    )
    api_key = providers_module.resolve_api_key(
        getattr(args, "api_key", None),
        config,
    )
    seacode_module.client = providers_module.build_client(
        provider, host=host, api_key=api_key, config=config
    )

    context_module.set_context_limit(
        context_module.resolve_context_tokens(
            getattr(args, "context_tokens", None), config
        )
    )

    # Generation options for the local runtime: a real context
    # window instead of Ollama's 4096 default, plus any per-model
    # profile from seacode.toml's [models."..."] tables (flag wins).
    model = seacode_module.resolve_model(
        getattr(args, "model", None), config
    )
    profile = load_model_profiles(root).get(model, {})
    num_ctx = providers_module.resolve_num_ctx(
        getattr(args, "num_ctx", None),
        {**config, **profile},
    )
    options = {"num_ctx": num_ctx}

    if "temperature" in profile:
        options["temperature"] = profile["temperature"]

    seacode_module.set_model_options(options)

    # Hooks from seacode.toml's [hooks] table, and MCP servers from
    # .mcp.json (their tools register as dynamic tools). Both launch
    # programs named by the project, so each needs the user's OK first
    # (see seacode.project_trust); either is a no-op when the project
    # has none configured.
    opt_in = getattr(args, "trust_project", False)
    project_hooks = hooks_module.load_hooks(root)

    if _hooks_trusted(root, project_hooks, opt_in):
        hooks_module.set_hooks(project_hooks)
    else:
        hooks_module.clear_hooks()

    _start_mcp(root, opt_in)

    model = seacode_module.resolve_model(
        getattr(args, "model", None), config
    )
    max_tool_calls = seacode_module.resolve_max_tool_calls(
        getattr(args, "max_tools", None), config
    )
    stream = not getattr(args, "no_stream", False)
    verbose = getattr(args, "verbose", False)

    return model, max_tool_calls, config, stream, verbose


def load_resume_session(args):
    """Load the session --resume names, or None if not asked.

    Returns the session dict. Exits non-zero with a message when the
    request can't be satisfied.
    """

    if getattr(args, "resume", None) is None:
        return None

    try:
        session = load_session(get_project_root(), args.resume)
    except (FileNotFoundError, ValueError) as e:
        print(f"Cannot resume session: {e}")
        raise SystemExit(1)

    print(f"Resumed session {session['id']} "
          f"({len(session['messages'])} messages).")

    return session


def run_repl(args) -> int:
    model, max_tool_calls, config, stream, verbose = (
        apply_common_options(args)
    )
    session = load_resume_session(args)

    session_id = None
    messages = None
    initial_turns = 0

    if session is not None:
        session_id = session["id"]
        messages = session["messages"]
        initial_turns = session.get("turns", 0)

        # A resumed session remembers its own model unless --model
        # (or SEACODE_MODEL) explicitly overrides it for this run.
        if (
            session.get("model")
            and getattr(args, "model", None) is None
            and os.environ.get("SEACODE_MODEL") is None
        ):
            model = session["model"]

    try:
        seacode_module.run_repl(
            model=model,
            max_tool_calls=max_tool_calls,
            stream=stream,
            initial_messages=messages,
            session_id=session_id,
            initial_turns=initial_turns,
            verbose=verbose,
        )
    finally:
        _stop_mcp()
    return 0


def run_tui(args) -> int:
    from seacode.tui import SeaCodeApp

    model, max_tool_calls, config, stream, _ = apply_common_options(args)
    session = load_resume_session(args)

    session_id = None
    messages = None
    initial_turns = 0

    if session is not None:
        session_id = session["id"]
        messages = session["messages"]
        initial_turns = session.get("turns", 0)

    try:
        SeaCodeApp(
            model=model,
            max_tool_calls=max_tool_calls,
            stream=stream,
            initial_messages=messages,
            session_id=session_id,
            initial_turns=initial_turns,
            animate=not getattr(args, "no_anim", False),
            theme=config.get("theme") if isinstance(config, dict) else None,
        ).run()
    finally:
        _stop_mcp()
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


def run_sessions(args) -> int:
    if args.dir:
        os.environ["SEACODE_DIR"] = args.dir

    entries = list_sessions(get_project_root())

    if not entries:
        print("No saved sessions.")
        return 0

    print(f"Sessions in {get_project_root()} (newest first):")

    for entry in entries:
        print(f"  {entry['id']}  "
              f"{turns_label(entry['turns'])}  "
              f"{entry['model']}")
        if entry["preview"]:
            print(f"    {entry['preview']}")

    return 0


def format_model_list(models) -> list:
    """Display lines for ollama's model objects/dicts."""

    lines = []

    for entry in models or []:
        name = getattr(entry, "model", None) or (
            entry.get("model") if isinstance(entry, dict) else None
        ) or getattr(entry, "name", None) or "?"

        size = getattr(entry, "size", None) or (
            entry.get("size") if isinstance(entry, dict) else None
        )
        size_text = (
            f"{size / 1e9:.1f} GB" if isinstance(size, (int, float))
            else "?"
        )
        lines.append(f"  {name}  ({size_text})")

    return lines


def run_eval_command(args) -> int:
    import seacode.evals as evals_module

    provider = providers_module.resolve_provider(
        getattr(args, "provider", None), None
    )
    host = providers_module.resolve_base_url(
        getattr(args, "host", None), None, provider
    )
    seacode_module.client = providers_module.build_client(
        provider,
        host=host,
        api_key=providers_module.resolve_api_key(
            getattr(args, "api_key", None), None
        ),
    )
    seacode_module.set_model_options({
        "num_ctx": providers_module.resolve_num_ctx(
            getattr(args, "num_ctx", None), None
        )
    })

    model = seacode_module.resolve_model(
        getattr(args, "model", None), None
    )
    task_ids = [args.task] if getattr(args, "task", None) else None

    print(f"Evaluating {model} via {provider} ...")
    results = evals_module.run_eval(model=model, task_ids=task_ids)

    passed = 0

    for result in results:
        mark = "PASS" if result["passed"] else "FAIL"
        passed += result["passed"]
        print(
            f"  [{mark}] {result['id']:<14} "
            f"{result['seconds']:>6.1f}s  {result['title']}"
        )

    print(f"{passed}/{len(results)} tasks passed.")
    return 0 if passed == len(results) else 1


def run_models(args) -> int:
    import ollama

    host = providers_module.resolve_base_url(
        getattr(args, "host", None), None, "ollama"
    )
    client = ollama.Client(host=host)

    try:
        if getattr(args, "pull", None):
            print(f"Pulling {args.pull} ...")
            client.pull(args.pull)
            print("Done.")

        response = client.list()
    except Exception as e:
        print(
            f"Cannot reach Ollama at {host}: {e}\n"
            "Is the server running? Start it with: ollama serve"
        )
        return 1

    models = getattr(response, "models", None) or (
        response.get("models") if isinstance(response, dict) else []
    )

    if not models:
        print("No local models. Pull one, e.g.:")
        print("  seacode models --pull qwen2.5-coder:7b")
        return 0

    print("Local models:")
    for line in format_model_list(models):
        print(line)

    return 0


def run_mcp(args) -> int:
    if args.dir:
        os.environ["SEACODE_DIR"] = args.dir

    root = get_project_root()
    config = mcp_module.load_mcp_config(root)

    if not config:
        print(f"No MCP servers configured ({root}/.mcp.json).")
        return 0

    if not _mcp_trusted(root, getattr(args, "trust_project", False)):
        print("MCP servers not started (project not trusted).")
        return 1

    manager = mcp_module.MCPManager()

    try:
        registered = manager.start_all(root)
    finally:
        tools_by_server = {
            name: [t.get("name", "") for t in client.tools]
            for name, client in manager.clients.items()
        }
        manager.stop_all()

    for server_name in config:
        tools = tools_by_server.get(server_name)

        if tools is None:
            print(f"  {server_name}: (failed to start)")
        else:
            print(f"  {server_name}: {', '.join(tools) or '(no tools)'}")

    print(f"Registered dynamic tools: {', '.join(registered) or '(none)'}")
    return 0


def run_trust(args) -> int:
    if args.dir:
        os.environ["SEACODE_DIR"] = args.dir

    root = get_project_root()

    if args.revoke:
        if project_trust_module.forget(root):
            print(f"Forgot stored approvals for {root}.")
        else:
            print(f"No stored approvals for {root}.")
        return 0

    mcp_config = mcp_module.load_mcp_config(root)
    hooks = hooks_module.load_hooks(root)

    if not mcp_config and not any(hooks.values()):
        print(f"Nothing to trust: no MCP servers or hooks in {root}.")
        return 0

    approved = True

    for kind, payload, lines in (
        (
            project_trust_module.KIND_MCP,
            mcp_config,
            project_trust_module.describe_mcp(mcp_config),
        ),
        (
            project_trust_module.KIND_HOOKS,
            {k: v for k, v in hooks.items() if v},
            project_trust_module.describe_hooks(hooks),
        ),
    ):
        if not payload:
            continue

        if project_trust_module.is_trusted(root, kind, payload):
            print(f"Already trusted: {kind}.")
            continue

        print(f"{kind}:")

        for line in lines:
            print(f"    {line}")

        try:
            answer = input("  Trust this exact config? [y/N]: ")
        except EOFError:
            answer = ""

        if answer.strip().lower() in ("y", "yes"):
            project_trust_module.remember(root, kind, payload)
            print("  Trusted.")
        else:
            approved = False

    return 0 if approved else 1


def run_init(args) -> int:
    if args.dir:
        os.environ["SEACODE_DIR"] = args.dir

    path = write_default_config(get_project_root(), force=args.force)

    if path is None:
        print("seacode.toml already exists (use --force to overwrite).")
        return 1

    print(f"Wrote {path}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
