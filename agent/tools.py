from pathlib import Path
import shlex
import subprocess

from agent.permissions import request_permission


IGNORED_DIRECTORIES = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}

ALLOWED_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "python",
    "pytest",
    "git",
}


def get_project_root() -> Path:
    """Return the root directory of the project."""

    return Path(__file__).resolve().parent.parent


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

    if file_path.exists() and not file_path.is_file():
        raise IsADirectoryError(
            f"Not a file: {path}"
        )

    if not request_permission(f"modify file: {path}"):
        return "File modification denied by user."

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    return f"File written successfully: {path}"


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
}