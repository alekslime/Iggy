from datetime import datetime
from pathlib import Path
import os
import shlex
import subprocess

from seacode.permissions import request_permission


BACKUP_DIR_NAME = ".seacode_backups"

IGNORED_DIRECTORIES = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    BACKUP_DIR_NAME,
}

ALLOWED_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "cat",
    "grep",
    "head",
    "tail",
    "wc",
    "diff",
    "python",
    "pytest",
    "git",
}


def get_project_root() -> Path:
    """Return the root directory of the project.

    This is the directory SeaCode operates on: the current working
    directory by default, or SEACODE_DIR when set (the `seacode --dir`
    flag sets it). Everything -- sandboxing, backups, file tools --
    is anchored here.
    """

    override = os.environ.get("SEACODE_DIR")

    if override:
        return Path(override).expanduser().resolve()

    return Path.cwd().resolve()


def is_safe_path(path: Path, root: Path) -> bool:
    """Check whether a path stays inside the project root."""

    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def git_status() -> str:
    """Return the current Git status of the project."""

    project_root = get_project_root()

    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Git status failed."
        )

    return result.stdout.strip()


def run_command(command: str) -> str:
    """Run an allowed command after requesting user permission."""

    if not command.strip():
        raise ValueError("Command cannot be empty.")

    args = shlex.split(command)

    if not args:
        raise ValueError("Command cannot be empty.")

    command_name = args[0]

    if command_name not in ALLOWED_COMMANDS:
        raise PermissionError(
            f"Command not allowed: {command_name}"
        )

    if not request_permission(f"run: {command}"):
        return "Command denied by user."

    project_root = get_project_root()

    result = subprocess.run(
        args,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )

    output = result.stdout

    if result.stderr:
        output += result.stderr

    return output.strip()


def list_files(root: str = ".") -> list[str]:
    """List relevant files inside the project directory."""

    project_root = get_project_root()

    requested_root = (project_root / root).resolve()

    if not is_safe_path(requested_root, project_root):
        raise PermissionError(
            "Access denied: directory is outside the project."
        )

    if not requested_root.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {root}"
        )

    if not requested_root.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {root}"
        )

    files = []

    for path in requested_root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        files.append(
            str(path.relative_to(project_root))
        )

    return sorted(files)


def read_file(path: str) -> str:
    """Read a text file inside the project."""

    project_root = get_project_root()
    file_path = (project_root / path).resolve()

    if not is_safe_path(file_path, project_root):
        raise PermissionError(
            "Access denied: file is outside the project directory."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    if not file_path.is_file():
        raise IsADirectoryError(
            f"Not a file: {path}"
        )

    return file_path.read_text(encoding="utf-8")


def is_backup_path(file_path: Path, project_root: Path) -> bool:
    """Check whether a path falls inside SeaCode's own backup directory."""

    try:
        relative_path = file_path.relative_to(project_root)
    except ValueError:
        return False

    return BACKUP_DIR_NAME in relative_path.parts


def backup_file(file_path: Path, project_root: Path) -> str:
    """Save a timestamped copy of a file's current content before it's
    overwritten.

    Backups are stored under .seacode_backups/, mirroring the project's
    directory structure, with one subfolder per file and one timestamped
    .bak per backup. Returns the backup's path relative to the project
    root.
    """

    relative_path = file_path.relative_to(project_root)

    backup_dir = project_root / BACKUP_DIR_NAME / relative_path
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_path = backup_dir / f"{timestamp}.bak"

    backup_path.write_bytes(file_path.read_bytes())

    return str(backup_path.relative_to(project_root))


def write_file(path: str, content: str) -> str:
    """Write a text file inside the project after requesting permission."""

    if not path.strip():
        raise ValueError("File path cannot be empty.")

    project_root = get_project_root()
    file_path = (project_root / path).resolve()

    if not is_safe_path(file_path, project_root):
        raise PermissionError(
            "Access denied: file is outside the project directory."
        )

    if is_backup_path(file_path, project_root):
        raise PermissionError(
            "Access denied: cannot modify SeaCode's own backup directory "
            f"({BACKUP_DIR_NAME}/)."
        )

    if file_path.exists() and not file_path.is_file():
        raise IsADirectoryError(
            f"Not a file: {path}"
        )

    if not request_permission(f"modify file: {path}"):
        return "File modification denied by user."

    if file_path.exists():
        backup_path = backup_file(file_path, project_root)
        print(f"Backed up existing file to: {backup_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    return f"File written successfully: {path}"


def restore_file_content(path: str, content: str) -> None:
    """Overwrite a file with previously-known content, without prompting
    for permission or creating a further backup.

    This is an internal-only helper for M7.6 semantic verification: when
    a modification tool reports success but the agent's post-hoc check
    determines the result doesn't match what was requested, the agent
    reverts the file to the content it captured immediately before the
    modification (which was already safely backed up by write_file /
    replace_in_file when the bad edit was made).

    Deliberately NOT registered in TOOLS: the model can never call this
    directly. It only runs from seacode.py's own recovery logic.
    """

    project_root = get_project_root()
    file_path = (project_root / path).resolve()

    if not is_safe_path(file_path, project_root):
        raise PermissionError(
            "Access denied: file is outside the project directory."
        )

    if is_backup_path(file_path, project_root):
        raise PermissionError(
            "Access denied: cannot modify SeaCode's own backup directory "
            f"({BACKUP_DIR_NAME}/)."
        )

    file_path.write_text(content, encoding="utf-8")


def remove_created_file(path: str) -> None:
    """Delete a file that write_file just created, when a post-creation
    check (M7.10 syntax validation) determines it should not be kept.

    Used specifically for the "brand-new file" case, where there is no
    prior content to fall back to via restore_file_content -- undoing
    the creation entirely is the only sensible revert.

    Deliberately NOT registered in TOOLS, like restore_file_content:
    the model can never call this directly. It only runs from
    seacode.py's own recovery logic. Silently no-ops if the file is
    already gone.
    """

    project_root = get_project_root()
    file_path = (project_root / path).resolve()

    if not is_safe_path(file_path, project_root):
        raise PermissionError(
            "Access denied: file is outside the project directory."
        )

    if is_backup_path(file_path, project_root):
        raise PermissionError(
            "Access denied: cannot modify SeaCode's own backup directory "
            f"({BACKUP_DIR_NAME}/)."
        )

    if file_path.exists():
        file_path.unlink()


def list_backups(path: str | None = None) -> list[dict]:
    """List SeaCode's timestamped backups, oldest first.

    With a path, lists backups for that file only; otherwise lists
    every backup in the project. Each entry has the project-relative
    file "path", the project-relative "backup" file, and the
    "timestamp" (the backup file's name without its .bak suffix).
    """

    project_root = get_project_root()
    backup_root = project_root / BACKUP_DIR_NAME

    if path is not None:
        if not path.strip():
            raise ValueError("File path cannot be empty.")

        file_path = (project_root / path).resolve()

        if not is_safe_path(file_path, project_root):
            raise PermissionError(
                "Access denied: file is outside the project directory."
            )

        search_roots = [backup_root / file_path.relative_to(project_root)]
    else:
        search_roots = [backup_root]

    if not backup_root.is_dir():
        return []

    entries = []

    for search_root in search_roots:
        if not search_root.is_dir():
            continue

        for backup_path in sorted(search_root.rglob("*.bak")):
            relative = backup_path.relative_to(backup_root)
            # Layout is <file path>/<timestamp>.bak, so the parent
            # chain is the original file's project-relative path.
            entries.append({
                "path": str(relative.parent),
                "backup": str(
                    backup_path.relative_to(project_root)
                ),
                "timestamp": backup_path.stem,
            })

    entries.sort(key=lambda entry: entry["timestamp"])

    return entries


def undo_last_change(path: str | None = None) -> str:
    """Restore a file from its most recent SeaCode backup.

    With a path, restores that file; otherwise restores whichever
    file was most recently backed up. The current content (if the
    file still exists) is backed up first, so an undo can itself be
    undone. Asks for user permission like any other modification.

    Deliberately NOT in MODIFICATION_TOOLS in seacode.py: the M7.6
    semantic check compares results against the request's desired
    text, which would fight a revert on principle. Undo is
    permission-gated and backup-preserving instead.
    """

    backups = list_backups(path)

    if not backups:
        if path is not None:
            return f"Nothing to undo: no backups found for {path}."
        return "Nothing to undo: no backups found in this project."

    latest = backups[-1]
    target_relpath = latest["path"]

    if not request_permission(f"restore file from backup: {target_relpath}"):
        return "File modification denied by user."

    project_root = get_project_root()
    backup_path = project_root / latest["backup"]
    target_path = (project_root / target_relpath).resolve()

    if not is_safe_path(target_path, project_root):
        raise PermissionError(
            "Access denied: file is outside the project directory."
        )

    if target_path.exists():
        if not target_path.is_file():
            raise IsADirectoryError(
                f"Not a file: {target_relpath}"
            )
        pre_undo_backup = backup_file(target_path, project_root)
        print(f"Backed up current content to: {pre_undo_backup}")
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    target_path.write_bytes(backup_path.read_bytes())

    return (
        f"Restored {target_relpath} from backup "
        f"{latest['backup']}."
    )


def replace_in_file(
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace an exact piece of text inside a project file."""

    if not path.strip():
        raise ValueError("File path cannot be empty.")

    if not old_text:
        raise ValueError("Old text cannot be empty.")

    project_root = get_project_root()
    file_path = (project_root / path).resolve()

    if not is_safe_path(file_path, project_root):
        raise PermissionError(
            "Access denied: file is outside the project directory."
        )

    if is_backup_path(file_path, project_root):
        raise PermissionError(
            "Access denied: cannot modify SeaCode's own backup directory "
            f"({BACKUP_DIR_NAME}/)."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    if not file_path.is_file():
        raise IsADirectoryError(
            f"Not a file: {path}"
        )

    content = file_path.read_text(encoding="utf-8")

    occurrences = content.count(old_text)

    if occurrences == 0:
        return (
            f"Replacement failed: exact text was not found in {path}."
        )

    if occurrences > 1:
        return (
            f"Replacement failed: exact text appears "
            f"{occurrences} times in {path}. "
            "The change was not applied."
        )

    if not request_permission(f"modify file: {path}"):
        return "File modification denied by user."

    backup_path = backup_file(file_path, project_root)
    print(f"Backed up existing file to: {backup_path}")

    updated_content = content.replace(
        old_text,
        new_text,
        1,
    )

    file_path.write_text(
        updated_content,
        encoding="utf-8",
    )

    return f"Replacement applied successfully: {path}"


def delete_file(path: str) -> str:
    """Delete a file inside the project after requesting permission.

    The file's content is backed up first (see backup_file), so a
    deletion can be recovered with undo_last_change.
    """

    if not path.strip():
        raise ValueError("File path cannot be empty.")

    project_root = get_project_root()
    file_path = (project_root / path).resolve()

    if not is_safe_path(file_path, project_root):
        raise PermissionError(
            "Access denied: file is outside the project directory."
        )

    if is_backup_path(file_path, project_root):
        raise PermissionError(
            "Access denied: cannot modify SeaCode's own backup directory "
            f"({BACKUP_DIR_NAME}/)."
        )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    if not file_path.is_file():
        raise IsADirectoryError(
            f"Not a file: {path}"
        )

    if not request_permission(f"delete file: {path}"):
        return "File modification denied by user."

    backup_path = backup_file(file_path, project_root)
    print(f"Backed up existing file to: {backup_path}")

    file_path.unlink()

    return f"File deleted successfully: {path}"


def search_files(query: str) -> list[dict]:
    """Search for text inside project files."""

    project_root = get_project_root()
    results = []

    if not query:
        raise ValueError("Search query cannot be empty.")

    for path in project_root.rglob("*"):

        if not path.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            if query.lower() in line.lower():

                results.append({
                    "file": str(path.relative_to(project_root)),
                    "line": line_number,
                    "text": line.strip(),
                })

    return results


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_files": search_files,
    "git_status": git_status,
    "run_command": run_command,
    "write_file": write_file,
    "replace_in_file": replace_in_file,
    "delete_file": delete_file,
    "undo_last_change": undo_last_change,
}
