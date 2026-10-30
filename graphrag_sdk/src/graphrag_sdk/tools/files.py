# GraphRAG SDK — Tools: files under a root folder
# The ingestion tools read only files below one root the server operator
# chose. Paths from the agent are relative; absolute paths, ".." and symlinks
# that leave the root are refused, and hidden files and folders are skipped.

from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from graphrag_sdk.ingestion.ingestion_planner import match_path
from graphrag_sdk.tools.errors import ToolError

#: File types the ingestion tools read (tables have their own path).
SUPPORTED_SUFFIXES: tuple[str, ...] = (".txt", ".md", ".markdown", ".rst", ".pdf")
#: Most files one ingest call accepts.
MAX_FILES_PER_CALL = 50
#: Most folder entries one listing walks; a bigger tree needs a narrower pattern.
MAX_WALK_ENTRIES = 100_000
#: Default largest file ``ingest(files=...)`` reads (50 MiB).
MAX_FILE_BYTES = 50 * 1024 * 1024
#: A text sample reads this many bytes per sample character, from the middle.
_WINDOW_FACTOR = 4
_MIN_WINDOW_BYTES = 4096
#: Most PDF pages parsed for one sample.
_PDF_SAMPLE_PAGES = 3
_GLOB_CHARS = frozenset("*?[")


def resolve_root(root: str | os.PathLike[str]) -> Path:
    """``root`` as an absolute, symlink-free directory path.

    The filesystem root, the home folder and any folder holding it (such as
    ``/Users``) are refused: an agent could then read every document of the
    machine or the user.
    """
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"files_root {str(root)!r} is not a directory")
    path = Path(path.anchor).joinpath(*_respell(Path(path.anchor), path.parts[1:], {}))
    home = Path.home().resolve()
    if path == Path(path.anchor) or home == path or home.is_relative_to(path):
        raise ValueError(
            f"files_root {str(root)!r} is the filesystem root or the home folder; "
            "choose the folder that holds the documents"
        )
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


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _respell(base: Path, parts: Sequence[str], listings: dict[Path, set[str]]) -> list[str]:
    """``parts`` below ``base`` spelled as their folder entries.

    A case-insensitive filesystem (the macOS and Windows default) opens
    ``NOTES/A.MD`` as ``notes/a.md`` and ``resolve()`` keeps the given case,
    so one file would get two document ids; ``listings`` caches each folder's
    entries as a set, so a big folder costs one lookup per name.
    """
    out = []
    folder = base
    for name in parts:
        if folder not in listings:
            try:
                listings[folder] = set(os.listdir(folder))
            except OSError:
                listings[folder] = set()
        if name not in listings[folder]:
            same = [n for n in listings[folder] if _fold(n) == _fold(name)]
            if len(same) == 1:
                name = same[0]
        out.append(name)
        folder = folder / name
    return out


def _canonical(root: Path, path: Path, listings: dict[Path, set[str]]) -> str | None:
    """``path`` relative to ``root`` with symlinks resolved and names spelled as
    on disk, so each file has one spelling; ``None`` when outside ``root``."""
    try:
        rel = path.resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    return PurePosixPath(*_respell(root, rel.parts, listings)).as_posix()


def _supported(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_SUFFIXES)


def list_files(
    root: Path, pattern: str = "**/*", *, max_entries: int = MAX_WALK_ENTRIES
) -> list[str]:
    """Supported files under ``root`` matching ``pattern``, as sorted relative paths.

    Hidden files and folders are skipped, folder symlinks are not followed
    and a file symlink pointing outside the root, or at a hidden or
    unsupported file, is left out. The walk
    always starts at ``root`` (a pattern also matches at any depth) and stops
    with a ToolError after ``max_entries`` entries.
    Blocking: call it from a worker thread in async code.
    """
    pattern = _check_relative(pattern)
    found: list[str] = []
    listings: dict[Path, set[str]] = {}
    seen = 0
    for folder, dirs, names in os.walk(root, followlinks=False):
        seen += len(dirs) + len(names)
        if seen > max_entries:
            raise ToolError(
                f"The files root holds more than {max_entries} entries.",
                hint="Point files_root at the folder that holds the documents.",
            )
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        base = Path(folder)
        for name in names:
            if name.startswith(".") or not _supported(name):
                continue
            path = base / name
            rel = path.relative_to(root).as_posix()
            if not (match_path(pattern, rel) and path.is_file()):
                continue
            real = _canonical(root, path, listings)
            if (
                real is None
                or any(p.startswith(".") for p in PurePosixPath(real).parts)
                or not _supported(real)
            ):
                continue
            found.append(rel)
    return sorted(found)


def resolve_files(
    root: Path, entries: Sequence[str], *, max_files: int = MAX_FILES_PER_CALL
) -> list[str]:
    """Expand relative paths and globs into at most ``max_files`` relative paths.

    Order follows ``entries`` (a glob's matches sorted), without repeats.
    Each path is the file's canonical one (symlinks resolved, names spelled
    as on disk), and two names of one file (a hard link) count once.
    """
    out: list[str] = []
    seen: set[str | tuple[int, int]] = set()
    listings: dict[Path, set[str]] = {}
    for entry in entries:
        text = _check_relative(entry)
        # A file whose name has '[' or '*' in it is that file, not a glob.
        literal = (root / text).is_file() and _inside(root, root / text)
        if _GLOB_CHARS & set(text) and not literal:
            matches = list_files(root, text)
            if not matches:
                raise ToolError(
                    f"No supported file matches {entry!r}.",
                    hint="Call preview_files to see the files and their paths.",
                )
        else:
            path = root / text
            if any(part.startswith(".") for part in PurePosixPath(text).parts):
                raise ToolError(f"{entry!r} is a hidden file.", hint="Hidden files are skipped.")
            if not path.is_file() or not _inside(root, path):
                raise ToolError(
                    f"No file {entry!r} under the files root.",
                    hint="Call preview_files to see the files and their paths.",
                )
            if not _supported(text):
                raise ToolError(
                    f"{entry!r} is not a supported file type.",
                    hint=f"Supported: {', '.join(SUPPORTED_SUFFIXES)}.",
                )
            matches = [text]
        for given in matches:
            rel = _canonical(root, root / given, listings)
            if rel is None or any(p.startswith(".") for p in PurePosixPath(rel).parts):
                raise ToolError(
                    f"{given!r} links to a file that is hidden or outside the files root.",
                    hint="Pass the file itself.",
                )
            if not _supported(rel):
                raise ToolError(
                    f"{given!r} links to an unsupported file type.",
                    hint=f"Supported: {', '.join(SUPPORTED_SUFFIXES)}.",
                )
            stat = (root / rel).stat()
            if rel not in seen and (stat.st_dev, stat.st_ino) not in seen:
                seen.update((rel, (stat.st_dev, stat.st_ino)))
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


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def check_sizes(root: Path, rels: Sequence[str], max_bytes: int | None) -> None:
    """Raise a ToolError naming the files above ``max_bytes`` (``None``: no cap).

    Blocking: call it from a worker thread in async code.
    """
    if max_bytes is None:
        return
    big = []
    for rel in rels:
        try:
            size = (root / rel).stat().st_size
        except OSError:
            continue  # the loader reports a file that vanished
        if size > max_bytes:
            big.append(f"{rel} ({_megabytes(size)})")
    if big:
        raise ToolError(
            f"{len(big)} file(s) are larger than max_file_bytes ({_megabytes(max_bytes)}): "
            + ", ".join(big[:10])
            + (" ..." if len(big) > 10 else "")
            + ".",
            hint="Split the file or raise max_file_bytes; leave it out of this call.",
        )


def confined(root: Path, path: str | os.PathLike[str], max_bytes: int | None = None) -> Path:
    """``path`` resolved, if it is still a file inside ``root`` (and not above ``max_bytes``).

    Checked again right before a file is read, so a symlink swapped in after
    the paths were validated cannot point the loader outside the root.
    Blocking: call it from a worker thread in async code.
    """
    real = Path(path).resolve()
    if not _inside(root, real) or not real.is_file():
        raise ToolError(
            f"{Path(path).name!r} is no longer a file under the files root.",
            hint="Call preview_files to see the files and their paths.",
        )
    size = real.stat().st_size
    if max_bytes is not None and size > max_bytes:
        raise ToolError(
            f"{Path(path).name!r} is larger than max_file_bytes "
            f"({_megabytes(size)} > {_megabytes(max_bytes)}).",
            hint="Split the file or raise max_file_bytes.",
        )
    return real


def sample_file(path: Path, max_chars: int) -> tuple[str, int, bool]:
    """``(sample, chars, estimated)``: a middle sample of a file, read in part.

    Text files are read only in a window around their middle (about
    ``4 * max_chars`` bytes) and PDFs only in a few middle pages, so a huge
    file costs no more than a small one. ``chars`` is the text length,
    estimated from the part read when ``estimated`` is True.
    Blocking: call it from a worker thread in async code.
    """
    if path.suffix.lower() == ".pdf":
        return _pdf_sample(path, max_chars)
    size = path.stat().st_size
    window = max(_WINDOW_FACTOR * max_chars, _MIN_WINDOW_BYTES)
    with path.open("rb") as fh:
        if size <= window:
            text = fh.read().decode("utf-8", errors="replace")
            return middle_sample(text, max_chars), len(text), False
        fh.seek((size - window) // 2)
        raw = fh.read(window)
    text = raw.decode("utf-8", errors="replace")
    # The window starts and ends mid-word (maybe mid-character): drop both ends,
    # unless that leaves nothing (text with no whitespace is cut by characters).
    words = text.split()
    words = words[1:-1] if len(words) > 2 else [w.strip("\ufffd") for w in words]
    chars = round(size * len(text) / max(len(raw), 1))
    return middle_sample(" ".join(words), max_chars), chars, True


def _pdf_sample(path: Path, max_chars: int) -> tuple[str, int, bool]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(str(path)) as doc:
            return _middle_pages(len(doc), lambda i: doc[i].get_text(sort=True), max_chars)
    from pypdf import PdfReader

    # A file object, not a path: given a path pypdf reads the whole file.
    with path.open("rb") as fh:
        reader = PdfReader(fh)
        return _middle_pages(len(reader.pages), lambda i: reader.pages[i].extract_text(), max_chars)


def _middle_pages(
    count: int, page_text: Callable[[int], str | None], max_chars: int
) -> tuple[str, int, bool]:
    texts: list[str] = []
    first = count // 2
    for i in range(first, min(first + _PDF_SAMPLE_PAGES, count)):
        texts.append(page_text(i) or "")
        if sum(len(t) for t in texts) >= max_chars:
            break
    text = "\n\n".join(texts)
    estimated = len(texts) < count
    chars = round(len(text) / len(texts) * count) if texts and estimated else len(text)
    return middle_sample(text, max_chars), chars, estimated
