"""Conversation persistence for SeaCode.

Transcripts are stored per project under `.seacode_sessions/` (see
SESSIONS_DIR_NAME) as one JSON file per session: the full message
list plus metadata. Sessions are local-only working state, like
`.seacode_backups/` -- never committed.

A session id is a UTC timestamp (`%Y%m%d-%H%M%S`); resume accepts an
exact id or a unique prefix. Writes are atomic (temp file + rename)
so a crash mid-save can't leave a corrupt session behind.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SESSIONS_DIR_NAME = ".seacode_sessions"

SESSION_VERSION = 1

VALID_ROLES = ("system", "user", "assistant")


def sessions_dir(project_root) -> Path:
    """Return the sessions directory for a project root."""

    return Path(project_root) / SESSIONS_DIR_NAME


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_session_id(existing_ids: set) -> str:
    """Return a fresh timestamp-based session id, suffixing on
    collision (two sessions started within the same second)."""

    base = utc_now().strftime("%Y%m%d-%H%M%S")

    if base not in existing_ids:
        return base

    counter = 2

    while f"{base}-{counter}" in existing_ids:
        counter += 1

    return f"{base}-{counter}"


def validate_messages(messages) -> list:
    """Check loaded data has the message-list shape run_agent needs."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("Session file is corrupt: no message list.")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Session file is corrupt: bad message.")
        if message.get("role") not in VALID_ROLES:
            raise ValueError("Session file is corrupt: bad role.")
        if not isinstance(message.get("content"), str):
            raise ValueError("Session file is corrupt: bad content.")

    return messages


def save_session(
    project_root, messages, model, session_id=None, turns=None
) -> str:
    """Save a transcript, returning the session id.

    With session_id, overwrites that session (keeping its original
    created timestamp); otherwise starts a new one. turns is the
    caller's count of human exchanges (tool-result followups share
    the user role, so it can't be derived reliably here); when
    omitted, the user-message count is stored instead.
    """

    if turns is None:
        turns = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        )

    directory = sessions_dir(project_root)
    directory.mkdir(parents=True, exist_ok=True)

    if session_id is None:
        existing = {path.stem for path in directory.glob("*.json")}
        session_id = generate_session_id(existing)

    session_file = directory / f"{session_id}.json"
    now = utc_now().isoformat(timespec="seconds")

    created = now

    if session_file.exists():
        try:
            previous = json.loads(session_file.read_text(encoding="utf-8"))
            created = previous.get("created", now)
        except (OSError, ValueError):
            pass

    payload = {
        "version": SESSION_VERSION,
        "model": model,
        "created": created,
        "updated": now,
        "turns": turns,
        "messages": messages,
    }

    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".tmp-",
        suffix=".json",
        delete=False,
    )

    try:
        json.dump(payload, tmp_file, indent=2)
        tmp_file.close()
        Path(tmp_file.name).replace(session_file)
    except BaseException:
        Path(tmp_file.name).unlink(missing_ok=True)
        raise

    return session_id


def load_session(project_root, session_id_or_prefix=None) -> dict:
    """Load a session by id, unique prefix, or (None) most recent.

    Returns {"id": ..., "model": ..., "messages": ...}. Raises
    FileNotFoundError when there is nothing to load and ValueError
    when a prefix is ambiguous or the file is corrupt.
    """

    entries = list_sessions(project_root)

    if session_id_or_prefix not in (None, "latest"):
        # An exact id whose file exists but wasn't listed is corrupt
        # (list_sessions skips unreadable files) -- report that
        # rather than "no such session".
        exact_file = (
            sessions_dir(project_root) / f"{session_id_or_prefix}.json"
        )

        if exact_file.is_file() and not any(
            entry["id"] == session_id_or_prefix for entry in entries
        ):
            try:
                payload = json.loads(
                    exact_file.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as e:
                raise ValueError(
                    f"Session file '{session_id_or_prefix}' is "
                    f"corrupt: {e}"
                )

            messages = validate_messages(payload.get("messages"))

            return {
                "id": session_id_or_prefix,
                "model": payload.get("model"),
                "turns": payload.get("turns", 0),
                "messages": messages,
            }

    if not entries:
        raise FileNotFoundError("No saved sessions in this project.")

    if session_id_or_prefix in (None, "latest"):
        chosen = entries[0]
    else:
        matches = [
            entry for entry in entries
            if entry["id"].startswith(session_id_or_prefix)
        ]

        if not matches:
            raise FileNotFoundError(
                f"No session matching '{session_id_or_prefix}'."
            )

        if len(matches) > 1:
            candidates = ", ".join(entry["id"] for entry in matches)
            raise ValueError(
                f"Session prefix '{session_id_or_prefix}' is ambiguous: "
                f"{candidates}"
            )

        chosen = matches[0]

    session_file = (
        sessions_dir(project_root) / f"{chosen['id']}.json"
    )

    try:
        payload = json.loads(session_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(
            f"Session file '{chosen['id']}' is corrupt: {e}"
        )

    messages = validate_messages(payload.get("messages"))

    return {
        "id": chosen["id"],
        "model": payload.get("model"),
        "turns": payload.get("turns", 0),
        "messages": messages,
    }


def turns_label(turns: int) -> str:
    """Human turn count with correct plural ("1 turn", "3 turns")."""

    if turns == 1:
        return "1 turn"

    return f"{turns} turns"


def session_preview(payload: dict) -> str:
    """One-line summary: the first user message, truncated."""

    messages = payload.get("messages", [])

    if not isinstance(messages, list):
        return ""

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            first_line = message.get("content", "").splitlines()
            text = first_line[0].strip() if first_line else ""
            if len(text) > 60:
                return text[:57] + "..."
            return text

    return ""


def list_sessions(project_root) -> list:
    """List saved sessions, newest first.

    Each entry has "id", "updated", "model", "preview", and "turns"
    (human exchanges, as recorded at save time). Corrupt files are
    skipped, not fatal.
    """

    directory = sessions_dir(project_root)

    if not directory.is_dir():
        return []

    entries = []

    for session_file in directory.glob("*.json"):
        try:
            payload = json.loads(
                session_file.read_text(encoding="utf-8")
            )
            messages = payload.get("messages", [])

            if "turns" in payload:
                turns = payload["turns"]
            else:
                turns = sum(
                    1 for m in messages
                    if isinstance(m, dict) and m.get("role") == "user"
                )

            entries.append({
                "id": session_file.stem,
                "updated": payload.get("updated", ""),
                "model": payload.get("model", ""),
                "preview": session_preview(payload),
                "turns": turns,
            })
        except (OSError, ValueError):
            continue

    entries.sort(
        key=lambda entry: (entry["updated"], entry["id"]),
        reverse=True,
    )

    return entries
