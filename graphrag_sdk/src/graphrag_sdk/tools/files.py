# GraphRAG SDK — Tools: files under a root folder
# The ingestion tools read only files below one root the server operator
# chose. Paths from the agent are relative; absolute paths, ".." and symlinks
# that leave the root are refused, and hidden files and folders are skipped.
# That a file is read from under the root even if the folder changes during
# the read is guaranteed on POSIX only (see open_confined).

from __future__ import annotations

import errno
import io
import os
import stat
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from graphrag_sdk.core.exceptions import LoaderError
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput
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
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_BINARY = getattr(os, "O_BINARY", 0)
#: Open each path component relative to its folder, never following a symlink.
_FD_WALK = bool(_NOFOLLOW and _DIRECTORY) and os.open in os.supports_dir_fd
#: What opening a path refused because it is gone, not a folder, or a symlink
#: (``O_NOFOLLOW`` fails with ``ELOOP``, or ``EMLINK`` on FreeBSD).
_GONE_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EMLINK})
#: ``(st_dev, st_ino)`` of each files root, recorded when it is resolved, so a
#: root folder replaced afterwards (or moved behind a symlink) is refused. A
#: later resolve of the same path overwrites it, so a toolkit keeps its own
#: record (:func:`resolve_root_with_id`) and passes it as ``root_id``; this one
#: is the fallback for callers that pass none.
_ROOT_IDS: dict[Path, tuple[int, int]] = {}


def resolve_root(root: str | os.PathLike[str]) -> Path:
    """``root`` as an absolute, symlink-free directory path.

    The filesystem root, the home folder and any folder holding it (such as
    ``/Users``) are refused: an agent could then read every document of the
    machine or the user.
    """
    return resolve_root_with_id(root)[0]


def resolve_root_with_id(root: str | os.PathLike[str]) -> tuple[Path, tuple[int, int]]:
    """:func:`resolve_root`, and the folder's ``(st_dev, st_ino)`` to pass as
    ``root_id`` to the confined reads, so the folder is checked against this
    caller's own record rather than the latest resolve of the same path."""
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
    info = path.stat()
    _ROOT_IDS[path] = ident = (info.st_dev, info.st_ino)
    return path, ident


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
                # The count only grows, so stop at the first file too many.
                if len(out) > max_files:
                    raise ToolError(
                        f"More than {max_files} files match; one call takes at most {max_files}.",
                        hint=(
                            "Split the files over several calls (narrower globs or explicit lists)."
                        ),
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


def canonical_file(root: Path, rel: str) -> str:
    """``rel`` as its file's canonical path under ``root`` (a ToolError if outside it).

    A listed file may be a symlink to another file under the root;
    :func:`open_confined` refuses symlinks, so it opens this path instead.
    Blocking: call it from a worker thread in async code.
    """
    real = _canonical(root, root / rel, {})
    if real is None:
        raise _gone(rel)
    # The link may have been pointed elsewhere since the listing: apply the
    # listing's rules to where it leads now.
    if any(p.startswith(".") for p in PurePosixPath(real).parts) or not _supported(real):
        raise ToolError(
            f"{rel!r} links to a file that is hidden or not a supported type.",
            hint="Pass the file itself.",
        )
    return real


def _gone(rel: str) -> ToolError:
    return ToolError(
        f"{rel!r} is no longer a file under the files root.",
        hint="Call preview_files to see the files and their paths.",
    )


def _open_failed(rel: str, exc: OSError) -> ToolError:
    if exc.errno in _GONE_ERRNOS:
        return _gone(rel)
    if exc.errno in (errno.EACCES, errno.EPERM):
        return ToolError(
            f"{rel!r} cannot be read: permission denied.",
            hint="Ask the server operator to make the file readable, or leave it out.",
        )
    return ToolError(f"{rel!r} could not be opened: {exc.strerror or exc}.")


def _root_replaced() -> ToolError:
    return ToolError(
        "The files root is no longer the folder it was when the server started; "
        "nothing is read from it.",
        hint="Ask the server operator to restart it with the intended files_root.",
    )


def _open_root(root: Path, root_id: tuple[int, int] | None = None) -> int:
    """``root`` opened as a folder, refused if it is a symlink now or not the
    folder ``root_id`` names (by default the one :func:`resolve_root` last
    recorded; a folder above it swapped for a symlink leads elsewhere)."""
    try:
        folder = os.open(root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    except OSError as exc:
        if exc.errno not in _GONE_ERRNOS:
            raise
        raise _root_replaced() from None
    try:
        info = os.fstat(folder)
        expected = root_id or _ROOT_IDS.setdefault(root, (info.st_dev, info.st_ino))
        if expected != (info.st_dev, info.st_ino):
            raise _root_replaced()
    except BaseException:
        os.close(folder)
        raise
    return folder


def open_confined(
    root: Path, rel: str, max_bytes: int | None = None, *, root_id: tuple[int, int] | None = None
) -> BinaryIO:
    """The file ``rel`` under ``root``, opened for reading after checking it is still there.

    On POSIX the root is opened without following a symlink and checked to be
    the folder :func:`resolve_root` recorded; each folder on the way is then
    opened relative to the one above it, again without following a symlink
    (``O_NOFOLLOW``), so a symlink swapped in after the paths were validated
    cannot lead outside the root. ``fstat`` on the open file then checks it is
    a regular file (not a FIFO or a device, opened with ``O_NONBLOCK`` so it
    cannot block) within ``max_bytes``. The caller reads that same open file.
    ``rel`` must be canonical (see :func:`canonical_file`).

    Without ``dir_fd`` support (Windows) the path is resolved, checked to be
    under the root and then reopened by name, so a symlink or junction
    swapped in between the check and the open is followed: there, the
    guarantee that nothing outside the root is read does not hold against a
    folder changed during the read. Blocking: call it from a worker thread.
    """
    pure = PurePosixPath(rel)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise _gone(rel)
    try:
        if _FD_WALK:
            folder = _open_root(root, root_id)
            try:
                for name in pure.parts[:-1]:
                    inner = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=folder)
                    os.close(folder)
                    folder = inner
                fd = os.open(pure.parts[-1], os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=folder)
            finally:
                os.close(folder)
        else:  # no dir_fd (Windows): check the resolved path, then open it
            real = (root / rel).resolve()
            if not _inside(root, real):
                raise _gone(rel)
            fd = os.open(real, os.O_RDONLY | _BINARY | _NONBLOCK)
    except OSError as exc:
        raise _open_failed(rel, exc) from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise _gone(rel)
        if max_bytes is not None and info.st_size > max_bytes:
            raise _too_large(rel, info.st_size, max_bytes)
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _too_large(rel: str, size: int, max_bytes: int) -> ToolError:
    return ToolError(
        f"{rel!r} is larger than max_file_bytes ({_megabytes(size)} > {_megabytes(max_bytes)}).",
        hint="Split the file or raise max_file_bytes.",
    )


def _read_capped(fh: BinaryIO, rel: str, max_bytes: int | None) -> bytes:
    """All of ``fh``, but never more than ``max_bytes`` (a file that grew
    after its size was checked is refused, not read whole)."""
    if max_bytes is None:
        return fh.read()
    data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise _too_large(rel, len(data), max_bytes)
    return data


def read_confined(
    root: Path, rel: str, max_bytes: int | None = None, *, root_id: tuple[int, int] | None = None
) -> bytes:
    """The bytes of ``rel`` under ``root``, read through :func:`open_confined`.

    At most ``max_bytes`` are read, even if the file grows while open.
    Blocking: call it from a worker thread in async code.
    """
    with open_confined(root, rel, max_bytes, root_id=root_id) as fh:
        return _read_capped(fh, rel, max_bytes)


def document_from_bytes(rel: str, data: bytes) -> DocumentOutput:
    """The document a loader would make of the file ``rel`` holding ``data``.

    The same rule as ``GraphRAG``'s default loaders: ``.pdf`` is parsed as a
    PDF, ``.md`` as Markdown (with its structure) and anything else as UTF-8
    text, with line endings normalised as reading a text file does.
    """
    lower = rel.lower()
    if lower.endswith(".pdf"):
        return _pdf_document(rel, data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LoaderError(
            f"{rel!r} is not UTF-8 text ({exc.reason} at byte {exc.start})."
        ) from None
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    meta = {"size_bytes": len(data), "suffix": PurePosixPath(rel).suffix}
    if lower.endswith(".md"):
        from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader

        return DocumentOutput(
            text=text,
            document_info=DocumentInfo(path=rel, metadata={**meta, "loader": "markdown"}),
            elements=MarkdownLoader()._parse_markdown(text),
        )
    return DocumentOutput(
        text=text, document_info=DocumentInfo(path=rel, metadata={**meta, "loader": "text"})
    )


def _pdf_document(rel: str, data: bytes) -> DocumentOutput:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    try:
        if fitz is not None:
            with fitz.open(stream=data, filetype="pdf") as doc:
                pages = [page.get_text(sort=True) for page in doc]
                count, backend = len(doc), "pymupdf"
            pages = [t for t in pages if t and t.strip()]
        else:
            try:
                from pypdf import PdfReader
            except ImportError:
                raise ImportError(
                    "A PDF library is required for PDF loading. Install one of:\n"
                    "  pip install graphrag-sdk[pdf]       # pypdf (default)\n"
                    "  pip install graphrag-sdk[pdf-fast]  # PyMuPDF, AGPL-3.0"
                ) from None
            reader = PdfReader(io.BytesIO(data))
            pages = [t for t in (page.extract_text() for page in reader.pages) if t]
            count, backend = len(reader.pages), "pypdf"
    except ImportError:
        raise
    except Exception as exc:
        raise LoaderError(f"Failed to read PDF {rel!r}: {exc}") from exc
    return DocumentOutput(
        text="\n\n".join(pages),
        document_info=DocumentInfo(
            path=rel,
            metadata={"page_count": count, "loader": "pdf", "pdf_backend": backend},
        ),
    )


def sample_confined(
    root: Path,
    rel: str,
    max_chars: int,
    max_bytes: int | None = None,
    *,
    root_id: tuple[int, int] | None = None,
) -> tuple[str, int, bool, int]:
    """:func:`sample_file` of ``rel`` under ``root``, read through :func:`open_confined`,
    and the file's size. A PDF is parsed from memory, so one above ``max_bytes``
    is refused. Blocking: call it from a worker thread in async code.
    """
    pdf = rel.lower().endswith(".pdf")
    cap = max_bytes if pdf else None
    with open_confined(root, rel, cap, root_id=root_id) as fh:
        size = os.fstat(fh.fileno()).st_size
        return (*_sample_open(fh, pdf, size, max_chars, rel=rel, max_bytes=cap), size)


def sample_file(path: Path, max_chars: int) -> tuple[str, int, bool]:
    """``(sample, chars, estimated)``: a middle sample of a file, read in part.

    Text files are read only in a window around their middle (about
    ``4 * max_chars`` bytes), so a huge one costs no more than a small one.
    PDFs are sampled from a few middle pages, but PyMuPDF loads the whole
    file into memory (pypdf reads only the pages asked for). This function
    sets no size cap; :func:`sample_confined`, which the tools use, refuses a
    PDF above ``max_file_bytes`` first, so its memory is unbounded only when
    that cap is ``None``. ``chars`` is the text
    length, estimated from the part read when ``estimated`` is True.
    Blocking: call it from a worker thread in async code.
    """
    with path.open("rb") as fh:
        return _sample_open(fh, path.suffix.lower() == ".pdf", path.stat().st_size, max_chars)


def _sample_open(
    fh: BinaryIO,
    pdf: bool,
    size: int,
    max_chars: int,
    *,
    rel: str = "",
    max_bytes: int | None = None,
) -> tuple[str, int, bool]:
    if pdf:
        return _pdf_sample(fh, max_chars, rel=rel, max_bytes=max_bytes)
    window = max(_WINDOW_FACTOR * max_chars, _MIN_WINDOW_BYTES)
    if size <= window:
        # Bounded by the window: a file that grew since fstat is not read whole.
        text = fh.read(window).decode("utf-8", errors="replace")
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


def _pdf_sample(
    fh: BinaryIO, max_chars: int, *, rel: str = "", max_bytes: int | None = None
) -> tuple[str, int, bool]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    if fitz is not None:
        with fitz.open(stream=_read_capped(fh, rel, max_bytes), filetype="pdf") as doc:
            return _middle_pages(len(doc), lambda i: doc[i].get_text(sort=True), max_chars)
    from pypdf import PdfReader

    # A file object: pypdf reads only the pages asked for.
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
