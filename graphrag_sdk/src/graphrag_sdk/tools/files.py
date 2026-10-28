# GraphRAG SDK — Tools: files under a root folder
# The ingestion tools read only files below one root the server operator
# chose. Paths from the agent are relative; absolute paths, ".." and symlinks
# that leave the root are refused, and hidden files and folders are skipped.

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from graphrag_sdk.ingestion.ingestion_planner import match_path
from graphrag_sdk.tools.errors import ToolError

#: File types the ingestion tools read (tables have their own path).
SUPPORTED_SUFFIXES: tuple[str, ...] = (".txt", ".md", ".markdown", ".rst", ".pdf")
#: Most files one ingest call accepts.
MAX_FILES_PER_CALL = 50
_GLOB_CHARS = frozenset("*?[")


def resolve_root(root: str | os.PathLike[str]) -> Path:
    """``root`` as an absolute, symlink-free directory path."""
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"files_root {str(root)!r} is not a directory")
    return path


def _check_relative(entry: str) -> str:
    """``entry`` as a forward-slash relative path or glob, or a ToolError."""
    text = entry.strip().replace("\\", "/")
    if not text:
        raise ToolError("An empty path was given.", hint="Pass paths relative to the files root.")
    pure = PurePosixPath(text)
    if pure.is_absolute() or (len(text) > 1 and text[1] == ":") or ".." in pure.parts:
        raise ToolError(
            f"{entry!r} is outside the files root.",
            hint="Pass paths relative to the files root, without '..'.",
        )
    while text.startswith("./"):
        text = text[2:]
    return text or "*"


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _supported(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_SUFFIXES)


def list_files(root: Path, pattern: str = "**/*") -> list[str]:
    """Supported files under ``root`` matching ``pattern``, as sorted relative paths.

    Hidden files and folders are skipped, folder symlinks are not followed
    and a file symlink pointing outside the root is left out.
    """
    pattern = _check_relative(pattern)
    found: list[str] = []
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        base = Path(folder)
        for name in names:
            if name.startswith(".") or not _supported(name):
                continue
            path = base / name
            rel = path.relative_to(root).as_posix()
            if match_path(pattern, rel) and path.is_file() and _inside(root, path):
                found.append(rel)
    return sorted(found)


def resolve_files(
    root: Path, entries: Sequence[str], *, max_files: int = MAX_FILES_PER_CALL
) -> list[str]:
    """Expand relative paths and globs into at most ``max_files`` relative paths.

    Order follows ``entries`` (a glob's matches sorted), without repeats.
    """
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        text = _check_relative(entry)
        if _GLOB_CHARS & set(text):
            matches = list_files(root, text)
            if not matches:
                raise ToolError(
                    f"No supported file matches {entry!r}.",
                    hint="Call sample_docs to see the files and their paths.",
                )
        else:
            path = root / text
            if any(part.startswith(".") for part in PurePosixPath(text).parts):
                raise ToolError(f"{entry!r} is a hidden file.", hint="Hidden files are skipped.")
            if not path.is_file() or not _inside(root, path):
                raise ToolError(
                    f"No file {entry!r} under the files root.",
                    hint="Call sample_docs to see the files and their paths.",
                )
            if not _supported(text):
                raise ToolError(
                    f"{entry!r} is not a supported file type.",
                    hint=f"Supported: {', '.join(SUPPORTED_SUFFIXES)}.",
                )
            matches = [path.relative_to(root).as_posix()]
        for rel in matches:
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
    if len(out) > max_files:
        raise ToolError(
            f"{len(out)} files match; one call takes at most {max_files}.",
            hint="Split the files over several calls (narrower globs or explicit lists).",
        )
    return out


def middle_sample(text: str, max_chars: int) -> str:
    """About ``max_chars`` characters from the middle of ``text``, cut at spaces.

    The middle avoids the title page, table of contents and boilerplate that
    open many documents; whitespace runs are collapsed.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    start = (len(text) - max_chars) // 2
    end = start + max_chars
    space = text.find(" ", start, start + 80)
    if space != -1:
        start = space + 1
    space = text.rfind(" ", end - 80, end)
    if space > start:
        end = space
    return text[start:end]
