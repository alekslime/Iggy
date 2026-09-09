# TUI look & feel -- direction to move toward

Current `tui.py` is a Claude-Code-style split view: a bordered live feed
pane + diff pane side by side, docked chat input at the bottom, standard
Textual `Header`/`Footer`. Functionally solid, but the aesthetic isn't
landing.

Reference: OpenCode's welcome screen (screenshot provided 2026-09-09).
What to borrow from it:

- **Big blocky/pixel-art wordmark** ("OPENCODE") centered near the top
  of the idle screen, instead of a plain text `Header` title bar.
- **Centered, rounded-border input box** with placeholder copy like
  `Ask anything... "Fix a TODO in the codebase"` -- softer and more
  inviting than a full-width docked input.
- **A status line directly under the input**, made of a few short
  pill-like segments (in OpenCode's case: agent name, a nickname/tag,
  and the backend/provider) -- for Iggy this maps to something like
  model name / mode (Scout·Operate) / project name.
- **A row of keybinding hints below that**, small and muted (their
  example: `ctrl+t variants  tab agents  ctrl+p commands`) -- Iggy's
  equivalent would surface things like `shift+tab Scout/Operate`.
- **A slim bottom status bar**, not a full Textual `Footer` legend --
  shows cwd/project path, MCP/tool status, version, right-aligned.

Open question for later: does this replace the idle/pre-first-message
screen only (like OpenCode: splash while empty, then it becomes a
normal chat transcript once you start typing), or should the live
feed + diff panes get restyled to match this palette/chrome too?
Worth deciding before implementing, since it changes how much of
`tui.py` is a rewrite vs. a reskin.

Not implemented yet -- captured here so it doesn't get lost.
