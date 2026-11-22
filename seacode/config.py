"""Project configuration for SeaCode (`seacode.toml`).

A project root may contain an `seacode.toml` with an `[seacode]` table:

    [seacode]
    model = "qwen2.5-coder:3b"
    provider = "ollama"    # ollama | openai (any OpenAI-compatible API)
    host = "http://127.0.0.1:11434"
    api_key = "..."        # for provider = "openai" (prefer SEACODE_API_KEY)
    max_tools = 10
    trust = "ask"          # ask | edits | all
    extra_commands = []    # appended to the run_command allowlist

Precedence for every key is CLI flag > environment variable >
seacode.toml > built-in default. Unknown keys are ignored so older
releases keep working when new keys are added later.
"""

import sys
import tomllib
from pathlib import Path


CONFIG_FILENAME = "seacode.toml"

CONFIG_TABLE = "seacode"

KNOWN_KEYS = (
    "model",
    "provider",
    "host",
    "api_key",
    "max_tools",
    "context_tokens",
    "trust",
    "theme",
    "extra_commands",
)

DEFAULT_CONFIG_TEXT = """\
# SeaCode project configuration.
# Every key is optional; CLI flags and SEACODE_* env vars override these.

[seacode]
# model = "qwen2.5-coder:3b"
# provider = "ollama"  # ollama | openai (OpenAI/OpenRouter/vLLM/LM Studio/...)
# host = "http://127.0.0.1:11434"
# api_key = "..."  # only for provider = "openai"; prefer $SEACODE_API_KEY
# max_tools = 10
# context_tokens = 16000  # wire budget; older history is trimmed, not lost
# trust = "ask"  # ask | edits | all
# theme = "abyss"  # TUI theme: abyss | kelp | coral | mono
# extra_commands = []  # appended to the run_command allowlist

# [hooks]
# pre_tool = []   # commands; a non-zero exit blocks the tool call
# post_tool = []  # commands; failures only warn
"""


def config_path(project_root) -> Path:
    return Path(project_root) / CONFIG_FILENAME


def load_project_config(project_root) -> dict:
    """Read the project's seacode.toml, returning known keys only.

    A missing file means no configuration ({}). An unreadable or
    invalid file warns on stderr and is treated the same way rather
    than refusing to run.
    """

    path = config_path(project_root)

    if not path.is_file():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
        data = tomllib.loads(raw)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Warning: ignoring invalid {CONFIG_FILENAME}: {e}",
              file=sys.stderr)
        return {}

    table = data.get(CONFIG_TABLE, {})

    if not isinstance(table, dict):
        print(f"Warning: ignoring invalid [{CONFIG_TABLE}] table in "
              f"{CONFIG_FILENAME}: expected a table.",
              file=sys.stderr)
        return {}

    return {
        key: table[key] for key in KNOWN_KEYS if key in table
    }


def write_default_config(project_root, force=False):
    """Write a starter seacode.toml. Returns the path, or None if the
    file already exists and force is not set."""

    path = config_path(project_root)

    if path.exists() and not force:
        return None

    path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")

    return path
