"""
Built-in file system tools: read_file and list_directory.
"""


import os

from agent_harness.tools.base import tool


@tool(description=(
    "Read the text contents of a file at the given path. "
    "Returns the file contents as a string. "
    "Use encoding='utf-8' by default; pass a different encoding if needed."
))
def read_file(path: str, encoding: str = "utf-8") -> str:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return f"ERROR: File not found: {abs_path}"
    if not os.path.isfile(abs_path):
        return f"ERROR: Path is not a file: {abs_path}"
    try:
        with open(abs_path, encoding=encoding) as fh:
            return fh.read()
    except Exception as exc:
        return f"ERROR: Could not read {abs_path}: {exc}"


@tool(description=(
    "List the contents of a directory. "
    "Returns a newline-separated list of entries. "
    "Directories are suffixed with '/'. "
    "Pass recursive=True to walk subdirectories (max depth 3)."
))
def list_directory(path: str = ".", recursive: bool = False) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return f"ERROR: Directory not found: {abs_path}"
    if not os.path.isdir(abs_path):
        return f"ERROR: Path is not a directory: {abs_path}"

    lines: list[str] = []

    if recursive:
        for root, dirs, files in os.walk(abs_path):
            depth = root.replace(abs_path, "").count(os.sep)
            if depth >= 3:
                dirs.clear()
                continue
            indent = "  " * depth
            rel_root = os.path.relpath(root, abs_path)
            if rel_root != ".":
                lines.append(f"{indent}{rel_root}/")
            file_indent = "  " * (depth + 1)
            for fname in sorted(files):
                lines.append(f"{file_indent}{fname}")
    else:
        try:
            entries = sorted(os.listdir(abs_path))
        except PermissionError as exc:
            return f"ERROR: {exc}"
        for entry in entries:
            suffix = "/" if os.path.isdir(os.path.join(abs_path, entry)) else ""
            lines.append(f"{entry}{suffix}")

    return "\n".join(lines) if lines else "(empty directory)"
