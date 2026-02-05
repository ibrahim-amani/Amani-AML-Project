from pathlib import Path

DEFAULT_IGNORE = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".vscode",
    "chrome_profile",
    "node_modules",
    ".idea",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
    "schema",
    "raw",
    "processed",
    "output",
    "logs",
    "results",
    ".dvc",
    ".ipynb_checkpoints",
    "final",
    "files",
    "pdfs",
    "error_logs"
    
}

def print_tree(
    root: Path,
    prefix: str = "",
    depth: int = 0,
    max_depth: int | None = None,
    ignore: set[str] = DEFAULT_IGNORE,
):
    if max_depth is not None and depth > max_depth:
        return

    try:
        entries = sorted(
            (e for e in root.iterdir() if e.name not in ignore),
            key=lambda x: (x.is_file(), x.name.lower()),
        )
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(
                entry,
                prefix + extension,
                depth + 1,
                max_depth=max_depth,
                ignore=ignore,
            )

if __name__ == "__main__":
    root = Path(".").resolve()
    print(root.name)
    print_tree(root, max_depth=4)
