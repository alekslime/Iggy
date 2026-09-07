from pathlib import Path


IGNORED_DIRECTORIES = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
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
        raise FileNotFoundError(f"File does not exist: {path}")

    if not file_path.is_file():
        raise IsADirectoryError(f"Not a file: {path}")

    return file_path.read_text(encoding="utf-8")


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
}