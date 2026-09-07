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


def list_files(root: str) -> list[str]:
    """List relevant files inside a project directory."""

    root_path = Path(root).resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    files = []

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        files.append(str(path.relative_to(root_path)))

    return sorted(files)


TOOLS = {
    "list_files": list_files,
}