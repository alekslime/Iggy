# SeaCode

SeaCode is a local-first CLI AI programming tool. It runs a coding agent
in your terminal: it inspects your project, edits files with your
permission (showing the diff first), verifies its work, and can undo
its changes. By default it talks to your own Ollama model; any
OpenAI-compatible endpoint works too (see Providers).

## Quickstart

Requirements: Python 3.12+, [Ollama](https://ollama.com) running
locally, and a coding model, e.g.:

```sh
ollama pull qwen2.5-coder:3b
```

Install and run, from the project you want to work on:

```sh
git clone https://github.com/alekslime/SeaCode
cd SeaCode
python3 -m venv .venv
.venv/bin/pip install -e .
cd /path/to/your/project
seacode tui
```

SeaCode operates on the current directory (or `--dir`). File tools
refuse paths outside it, `run_command` refuses arguments that point
outside it, and every modification asks for permission first. (See
"Command safety" below for what that does and doesn't guarantee.)

## Commands

```sh
seacode                    # chat REPL (same as `seacode repl`)
seacode tui                # full-screen terminal UI
seacode undo               # restore the most recently backed-up file
seacode undo notes.txt     # restore one file's most recent backup
seacode undo --list        # list available backups
seacode sessions           # list saved conversation sessions
seacode models             # list local Ollama models (--pull MODEL)
seacode eval               # benchmark a local model on canned tasks
seacode mcp                # list MCP servers/tools from .mcp.json
seacode init               # write a starter seacode.toml in the project
seacode --version
```

Options (REPL and TUI):

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--model` | `SEACODE_MODEL` | `qwen2.5-coder:3b` | model name |
| `--provider` | `SEACODE_PROVIDER` | `ollama` | `ollama` or `openai` (OpenAI-compatible) |
| `--host` | `SEACODE_HOST` / `OLLAMA_HOST` | provider default | provider endpoint |
| `--api-key` | `SEACODE_API_KEY` / `OPENAI_API_KEY` | — | key for `--provider openai` |
| `--max-tools` | `SEACODE_MAX_TOOLS` | `10` | tool calls allowed per turn |
| `--context-tokens` | `SEACODE_CONTEXT_TOKENS` | `16000` | wire context budget (see below) |
| `--num-ctx` | `SEACODE_NUM_CTX` | `16384` | window requested from Ollama (its own default is only 4096) |
| `--trust` | `SEACODE_TRUST` | `ask` | `ask`, `edits`, or `all` |
| `--yes` | — | off | shorthand for `--trust all` |
| `--resume [ID]` | — | off | resume latest (or matching) session |
| `--no-stream` | — | off | wait for full replies |
| `--dir` | `SEACODE_DIR` | current directory | project to work on |

Every option also has an `seacode.toml` key (see below); precedence is
flag > env > config file > default.

## Project config

`seacode init` writes a starter `seacode.toml` in the project root:

```toml
[seacode]
model = "qwen2.5-coder:3b"
provider = "ollama"    # ollama | openai
host = "http://127.0.0.1:11434"
# api_key = "..."      # openai only; prefer SEACODE_API_KEY
max_tools = 10
context_tokens = 16000
trust = "ask"          # ask | edits | all
extra_commands = []    # appended to the run_command allowlist

[hooks]
pre_tool = []          # commands; non-zero exit blocks the tool call
post_tool = []         # commands; failures only warn
```

All keys are optional.

## Providers

Ollama is the default and needs nothing but a running server. For any
OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, llama.cpp,
LM Studio, …):

```sh
SEACODE_API_KEY=sk-... seacode repl --provider openai --model gpt-5.2 \
    --host https://api.openai.com/v1
```

Both providers use native function calling; models that ignore it
still work through the plain-text/legacy-JSON fallback in the agent
loop.

SeaCode is built for local models first: Ollama is asked for a real
context window (`--num-ctx`, since its 4096 default nearly fits the
system prompt alone), models of 7B or smaller get a compact system
prompt (same rules, a third of the tokens), and per-model profiles
can tune generation:

```toml
[models."qwen2.5-coder:3b"]
num_ctx = 8192
temperature = 0.2
```

`seacode models` lists what you have pulled; `seacode eval` runs
four canned tasks (typo fix, failing test, create file, explain)
through the real agent loop and scores them mechanically, so you
can compare local models on evidence instead of vibes.

## Project instructions and context

If the project root has a `SEACODE.md`, `AGENTS.md`, or `CLAUDE.md`,
its contents are appended to the system prompt (all three, in that
order, capped) — for new *and* resumed sessions.

Long transcripts are trimmed to the `--context-tokens` budget before
each model call: system + most recent messages are sent, older
history is replaced by a short note. The stored transcript and saved
sessions always keep everything; only the wire copy is trimmed.

## Trust levels

- `ask` — prompt for every file change and command (default).
- `edits` — auto-approve file changes, still prompt for commands.
- `all` — auto-approve everything (same as `--yes`).

Auto-approvals are always printed. Every file change shows its
unified diff *before* the y/N prompt (and in the transcript when
auto-approved), so you approve the actual edit, not just a filename.
In the TUI, Scout mode still denies every modification regardless of
trust.

## Command safety

`run_command` checks more than the command name. Before the permission
prompt, every argument is resolved against the project root and the
command is refused if any of them lands outside it (`../`, absolute
paths, `~`, `--opt=/path`, symlinks that escape). `find` options that
run programs or write files (`-exec`, `-delete`, `-fprint`, ...) and
git options that inject config or helper programs (`git -c`,
`--upload-pack`, `--ext-diff`, ...) are refused too. Refused commands
never prompt you.

Commands run with stdin closed and a 30-second timeout (a timeout
kills the process and reports whatever it had printed). Output sent
back to the model is capped at about 8,000 characters: the start and
the end are kept and the middle is replaced with a note saying how
much was dropped.

This is a guard rail, not a sandbox. `python`, `pytest` and `git` run
project code by design (a script, a `conftest.py`, a git hook), so
with `--trust all` / `--yes` a model can still run arbitrary code as
you. Use `ask` (the default) unless you trust the model and the
project. The checks err on the side of refusing, so a grep pattern
that is literally `/` is blocked.

## Sessions

Conversations autosave per turn under `.seacode_sessions/` (local-only;
add it and `.seacode_backups/` to your project's `.gitignore` so they
stay out of commits). Resume with `seacode repl --resume` (latest) or
`seacode repl --resume <id-prefix>`; the TUI takes the same flag. A
resumed session keeps its model unless `--model` overrides it.

## How it works

The agent talks to the provider in a tool loop: the model calls
tools (natively, or via the text fallback), SeaCode executes them,
and results go back until the model gives a final answer.

Tools:

- `list_files`, `read_file`, `search_files`, `git_status` — inspection
- `run_command` — an allowlisted set (`python`, `pytest`, `git`, `ls`,
  `cat`, `grep`, …), argument-checked, permission-gated
- `write_file`, `replace_in_file`, `delete_file` — modification, each
  permission-gated (with diff preview) and backed up to
  `.seacode_backups/` first
- `undo_last_change` — restores a file from its most recent backup
  (the agent itself can use this to revert a bad edit)
- `task` — delegates a self-contained subtask to a subagent with its
  own conversation and a smaller tool budget (subagents can't spawn
  subagents)
- `mcp_<server>_<tool>` — tools from MCP servers (below)

Guardrails (the M7 series in `seacode/verify.py`, enforced by the
loop in `seacode/agent.py`): failed edits force a retry instead of
quitting; malformed model output is corrected; edits that destroy
file content, land on the wrong file, leave invalid Python, miss a
named file, or exceed an explicit scope are reverted with an
explanation; blind edits of unread files are blocked; and final
answers may not claim verification that never happened.

Model replies stream token by token as they arrive
(`--no-stream` restores the old wait-for-full-reply behavior).

The TUI (`seacode tui`) opens on a splash screen with a centered prompt,
model/mode/project pills, recent sessions (type a number to resume
one), project stats, and a tip. Once you start chatting it becomes a
single-column transcript where each file change shows its diff inline
with a one-command revert hint. `Shift+Tab` toggles Scout (read-only:
every modification is auto-denied) and Operate mode. Slash commands
work from either screen: `/help`, `/undo`, `/sessions`, `/model`,
`/tools`, `/scout`, `/operate`, `/clear`.

## MCP servers

Servers from a standard `.mcp.json` in the project root are spawned
over stdio at startup and their tools registered alongside the
built-ins:

```json
{"mcpServers": {"notes": {"command": "python", "args": ["notes_mcp.py"]}}}
```

`seacode mcp` lists what would be registered. MCP servers are
user-configured and trusted: their tools aren't individually
permission-gated, and one bad server is skipped with a warning rather
than taking the session down.

## Hooks

`[hooks]` in `seacode.toml` runs commands around every tool call.
Each hook gets a JSON payload on stdin
(`{"event", "tool", "arguments", "result"}`). A `pre_tool` hook that
exits non-zero blocks the call and tells the model why; `post_tool`
hooks can only warn. Hook commands are your own configuration, so
they aren't subject to the `run_command` allowlist.

## Project layout

```text
seacode/agent.py         # the tool loop and recovery policy (run_agent)
seacode/actions.py       # model output -> tool actions, stream chunk parsing
seacode/prompts.py       # system prompt text, model-size detection
seacode/render.py        # transcript formatting, colors, spinner
seacode/verify.py        # M7 guardrail checks (pure, unit-testable)
seacode/providers.py     # ollama + OpenAI-compatible clients, tool schemas
seacode/context.py       # project instructions, context-window trimming
seacode/tools.py         # tool implementations, sandboxing, backups
seacode/dynamic_tools.py # runtime-registered tools (MCP, ...)
seacode/mcp.py           # MCP stdio client + server manager
seacode/hooks.py         # pre/post tool hooks from seacode.toml
seacode/permissions.py   # approval prompts (with diffs) and trust levels
seacode/sessions.py      # conversation save/resume
seacode/config.py        # seacode.toml loading
seacode/cli.py           # `seacode` entry point
seacode/tui.py           # Textual interface
tests/                   # mocked suites (no Ollama server needed)
```

## Development

```sh
.venv/bin/pip install -e .[dev]
.venv/bin/python -m pytest tests/ -q
```

Tests script the model's replies and run the tools against temp
directories, so the full suite runs offline in a few seconds.

## Limitations

- A 3B local model handles small, well-scoped tasks best — say what
  you want precisely, name the files, and keep requests to one or two
  files at a time for the most reliable results. Stronger hosted
  models via `--provider openai` lift this considerably.
- Syntax validation covers Python; other languages rely on the
  content/target checks.
- MCP support is stdio transport only (no remote/HTTP servers), and
  MCP tools bypass the M7 modification checks — only run servers you
  trust.
