# SeaCode project instructions

- Python 3.12+, no frameworks beyond the declared deps (ollama, httpx,
  textual, rich; pytest for dev). Keep new modules stdlib-first.
- Tests are unittest-style, run offline, and must never need an Ollama
  server or the network: script the model's replies, run tools against
  temp directories, use httpx MockTransport / fake stdio servers.
- The agent transcript formats printed by run_agent are parsed by the
  TUI (see tui.classify_line) -- change them in lockstep.
- Every file modification is permission-gated and backed up first;
  new tools must go through seacode.permissions.request_permission
  and stay inside the project root.
- Precedence for every setting: CLI flag > env var > seacode.toml >
  default.
