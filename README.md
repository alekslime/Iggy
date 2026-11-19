# Iggy

Iggy is a local-first CLI AI programming tool. It runs a coding agent
against your own Ollama model, inside your terminal: it inspects your
project, edits files with your permission, verifies its work, and can
undo its changes.

## Quickstart

Requirements: Python 3.12+, [Ollama](https://ollama.com) running
locally, and a coding model, e.g.:

```sh
ollama pull qwen2.5-coder:3b
```

Install and run, from the project you want to work on:

```sh
git clone https://github.com/alekslime/Iggy
cd Iggy
/usr/bin/python3.14 -m venv .venv
.venv/bin/pip install -e .
cd /path/to/your/project
iggy tui
```

Iggy operates on the current directory (or `--dir`). It never touches
files outside it, and every modification asks for permission first.

## Commands

```sh
iggy                    # chat REPL (same as `iggy repl`)
iggy tui                # full-screen terminal UI
iggy undo               # restore the most recently backed-up file
iggy undo notes.txt     # restore one file's most recent backup
iggy undo --list        # list available backups
iggy --version
```

Options (REPL and TUI):

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--model` | `IGGY_MODEL` | `qwen2.5-coder:3b` | Ollama model |
| `--host` | `IGGY_HOST` / `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |
| `--max-tools` | `IGGY_MAX_TOOLS` | `10` | tool calls allowed per turn |
| `--dir` | `IGGY_DIR` | current directory | project to work on |

## How it works

The agent talks to Ollama in a tool loop: the model emits one JSON
tool call per turn, Iggy executes it, and the result goes back until
the model gives a final answer.

Tools:

- `list_files`, `read_file`, `search_files`, `git_status` — inspection
- `run_command` — an allowlisted set (`python`, `pytest`, `git`, `ls`,
  `cat`, `grep`, …), permission-gated
- `write_file`, `replace_in_file`, `delete_file` — modification, each
  permission-gated and backed up to `.iggy_backups/` first
- `undo_last_change` — restores a file from its most recent backup
  (the agent itself can use this to revert a bad edit)

Guardrails (the M7 series in `agent/agent.py`): failed edits force a
retry instead of quitting; malformed model output is corrected;
edits that destroy file content, land on the wrong file, leave
invalid Python, miss a named file, or exceed an explicit scope are
reverted with an explanation; blind edits of unread files are
blocked; and final answers may not claim verification that never
happened.

The TUI (`iggy tui`) opens on a splash screen with a centered prompt
and model/mode/project pills, then switches to a live feed plus diff
pane once you start chatting. `Shift+Tab` toggles Scout (read-only:
every modification is auto-denied) and Operate mode.

## Project layout

```text
agent/agent.py    # the tool loop, system prompt, M7 guardrails
agent/tools.py    # tool implementations, sandboxing, backups
agent/permissions.py  # y/N approval prompt
agent/cli.py      # `iggy` entry point (repl, tui, undo)
agent/tui.py      # Textual interface
tests/            # mocked suites (no Ollama server needed)
```

## Development

```sh
.venv/bin/pip install -e .[dev]
.venv/bin/python -m pytest tests/ -q
```

Tests script the model's replies and run the tools against temp
directories, so the full suite runs offline in a few seconds.

## Limitations

- Ollama only; no hosted-model providers yet.
- A 3B local model handles small, well-scoped tasks best — say what
  you want precisely, name the files, and keep requests to one or two
  files at a time for the most reliable results.
- Syntax validation covers Python; other languages rely on the
  content/target checks.
