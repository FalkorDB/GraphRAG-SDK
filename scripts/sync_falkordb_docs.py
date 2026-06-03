"""Sync the canonical GraphRAG-SDK documentation into the FalkorDB/docs Jekyll site.

The canonical source lives in this repository under
``docs/site/falkordb/genai-tools/graphrag-sdk/``. This script mirrors that
subtree into a ``FalkorDB/docs`` working tree at
``genai-tools/graphrag-sdk/``, wires the index page, merges spellcheck
wordlist additions, and (optionally) commits + pushes a fresh branch.

The script is idempotent — running it twice produces no further diff when
nothing in the canonical source has changed.

Branch contract: a fresh branch is created off ``origin/main`` per run, named
``sync/graphrag-sdk-<source-sha>``. The script never pushes to ``main`` and
never reuses an existing branch.

Usage:
    # Preview locally (no commit / push):
    python scripts/sync_falkordb_docs.py \\
        --target /path/to/falkordb-docs --no-push

    # CI mode (commit + push, open PR):
    python scripts/sync_falkordb_docs.py \\
        --target /tmp/falkordb-docs --pr
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_SRC = REPO_ROOT / "docs" / "site" / "falkordb"
SDK_SUBTREE = "genai-tools/graphrag-sdk"
INDEX_FILE = "genai-tools/index.md"
WORDLIST_FILE = ".wordlist.txt"

# Bullet on `genai-tools/index.md` that points at the SDK section. The
# pre-sync version pointed at the flat `./graphrag-sdk.md`; we rewrite it to
# the directory. Idempotent — no-op when the bullet is already correct.
INDEX_BULLET_OLD = "- [GraphRAG-SDK](./graphrag-sdk.md):"
INDEX_BULLET_NEW = "- [GraphRAG-SDK](./graphrag-sdk/):"

# Flat page that the new directory replaces. Deleted on first run; subsequent
# runs are no-ops (file is already gone in `origin/main`).
FLAT_PAGE_TO_REMOVE = "genai-tools/graphrag-sdk.md"


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    """Run a subprocess, capture stdout, raise on non-zero exit."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(
            f"$ {' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}\n"
        )
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def source_sha() -> str:
    """Short hash of the current HEAD of the canonical source repo."""
    return run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT)


def fresh_branch_name(sha: str) -> str:
    return f"sync/graphrag-sdk-{sha}"


def prepare_target_branch(target: Path, branch: str) -> None:
    """Fetch origin/main and check out a fresh branch off it.

    Never touches any existing branch. ``git checkout -b`` is used with a
    name guaranteed unique (by source SHA), so this can never re-use someone
    else's branch.
    """
    run(["git", "fetch", "origin", "main"], cwd=target)
    # Refuse to clobber: if branch already exists locally, the SHA collided
    # (shouldn't happen). Fail loudly instead of silently overwriting.
    existing = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
        cwd=target,
        capture_output=True,
    )
    if existing.returncode == 0:
        raise SystemExit(
            f"branch {branch!r} already exists in target. Refusing to overwrite. "
            "Delete it first or rebase canonical source onto a new commit."
        )
    run(["git", "checkout", "-b", branch, "origin/main"], cwd=target)


def wipe_target_subtree(target: Path) -> None:
    """Delete the existing SDK subtree (and the old flat page) so syncs are full-replace.

    Without this, files renamed or removed in the canonical source would
    linger in the target. Idempotent — deletes silently when paths don't exist.
    """
    subtree = target / SDK_SUBTREE
    if subtree.exists():
        shutil.rmtree(subtree)
    flat = target / FLAT_PAGE_TO_REMOVE
    if flat.exists():
        flat.unlink()


def copy_subtree(target: Path) -> None:
    """Copy canonical source subtree 1:1 into the target."""
    src = CANONICAL_SRC / SDK_SUBTREE
    dst = target / SDK_SUBTREE
    if not src.exists():
        raise SystemExit(f"canonical source not found at {src}")
    shutil.copytree(src, dst)


def update_section_index(target: Path) -> None:
    """Rewrite the GraphRAG-SDK bullet on ``genai-tools/index.md``.

    The pre-sync index points at the flat page (`./graphrag-sdk.md`). The
    sync re-points it at the directory (`./graphrag-sdk/`). Idempotent.
    """
    path = target / INDEX_FILE
    if not path.exists():
        # Index doesn't exist in target — log and skip; the section index is
        # nice-to-have, not load-bearing.
        sys.stderr.write(f"warning: {INDEX_FILE} missing in target; skipping bullet rewrite\n")
        return
    text = path.read_text(encoding="utf-8")
    if INDEX_BULLET_OLD in text:
        text = text.replace(INDEX_BULLET_OLD, INDEX_BULLET_NEW)
        path.write_text(text, encoding="utf-8")


def merge_wordlist(target: Path) -> None:
    """Merge canonical wordlist additions into the target's ``.wordlist.txt``.

    Sorted-unique merge — the existing wordlist is preserved, additions are
    appended only when not already present. Idempotent.
    """
    additions_path = CANONICAL_SRC / ".wordlist-additions.txt"
    target_path = target / WORDLIST_FILE
    if not additions_path.exists():
        return
    additions = {
        line.strip()
        for line in additions_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    existing = (
        set(target_path.read_text(encoding="utf-8").splitlines())
        if target_path.exists()
        else set()
    )
    merged = sorted(existing | additions)
    target_path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def has_changes(target: Path) -> bool:
    """Return True when ``git status --porcelain`` reports any work in the target."""
    return bool(run(["git", "status", "--porcelain"], cwd=target))


def commit_and_optionally_push(
    target: Path,
    *,
    branch: str,
    source_sha_str: str,
    push: bool,
    open_pr: bool,
) -> None:
    """Stage everything, commit, and (when requested) push + open a PR."""
    if not has_changes(target):
        print("no changes — target is already in sync.")
        return

    run(["git", "add", "-A"], cwd=target)
    commit_msg = (
        f"docs(graphrag-sdk): sync from FalkorDB/GraphRAG-SDK@{source_sha_str}\n\n"
        f"Auto-generated by scripts/sync_falkordb_docs.py in the SDK repo. "
        f"Source commit: https://github.com/FalkorDB/GraphRAG-SDK/commit/{source_sha_str}\n"
    )
    run(["git", "commit", "-m", commit_msg], cwd=target)
    print(f"committed on branch {branch}")

    if not push:
        return

    # Fresh ref every run — never force, never reuse.
    run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], cwd=target)
    print(f"pushed origin/{branch}")

    if open_pr:
        gh_body = (
            f"Sync of the canonical GraphRAG-SDK docs into `genai-tools/graphrag-sdk/`.\n\n"
            f"**Source commit:** "
            f"https://github.com/FalkorDB/GraphRAG-SDK/commit/{source_sha_str}\n\n"
            f"This PR is auto-generated. Edits made here will be overwritten by "
            f"the next sync — to make a content change, edit the canonical source "
            f"under `docs/site/falkordb/` in the SDK repo and re-run the sync.\n"
        )
        run(
            [
                "gh",
                "pr",
                "create",
                "--title",
                f"docs(graphrag-sdk): sync from GraphRAG-SDK@{source_sha_str}",
                "--body",
                gh_body,
                "--head",
                branch,
                "--base",
                "main",
            ],
            cwd=target,
        )
        print("opened PR")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="Working tree of FalkorDB/docs (must be a git checkout).",
    )
    parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        default=True,
        help="Commit locally; do not push or open a PR. Default in CI is push + PR.",
    )
    parser.add_argument(
        "--pr",
        dest="open_pr",
        action="store_true",
        default=False,
        help="After pushing, open a PR against FalkorDB/docs main via gh CLI.",
    )
    args = parser.parse_args()

    target: Path = args.target.resolve()
    if not (target / ".git").exists():
        raise SystemExit(f"target {target} is not a git checkout")

    sha = source_sha()
    branch = fresh_branch_name(sha)
    print(f"source @ {sha} → target branch {branch}")

    prepare_target_branch(target, branch)
    wipe_target_subtree(target)
    copy_subtree(target)
    update_section_index(target)
    merge_wordlist(target)
    commit_and_optionally_push(
        target,
        branch=branch,
        source_sha_str=sha,
        push=args.push,
        open_pr=args.open_pr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
