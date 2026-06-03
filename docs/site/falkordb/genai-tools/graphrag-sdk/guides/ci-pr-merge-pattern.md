---
title: "CI / PR-merge pattern"
nav_order: 6
parent: "Guides"
grand_parent: "GraphRAG-SDK"
description: "Re-sync a documentation knowledge graph on every PR merge — added, modified, and deleted files in one apply_changes call."
---

# CI / PR-merge pattern

The canonical incremental-update use case: you have a documentation graph backing a "talk to your docs" RAG endpoint, and you want every PR merge to keep the graph in sync — without rebuilding from scratch.

## The shape

```
PR merges to main
       │
       ▼
GitHub Action triggers
       │
       ▼
git diff main HEAD~1 --name-status     <-- find changed files
       │
       ▼
rag.apply_changes(added=A, modified=M, deleted=D)
       │
       ▼
rag.finalize()                          <-- exactly once per run
```

## Runnable script

Save as `scripts/sync_docs_graph.py`:

```python
import asyncio
import os
import subprocess
import sys
from pathlib import Path

from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder


def changed_docs(base_ref: str, head_ref: str) -> dict[str, list[str]]:
    """Return added / modified / deleted .md files between two refs."""
    diff = subprocess.check_output(
        ["git", "diff", "--name-status", f"{base_ref}..{head_ref}", "--", "docs/**/*.md"],
        text=True,
    ).strip().splitlines()
    added, modified, deleted = [], [], []
    for line in diff:
        status, _, path = line.partition("\t")
        # Renames show up as "R100\told\tnew" — treat as delete+add.
        if status.startswith("R") and "\t" in path:
            old, new = path.split("\t", 1)
            deleted.append(old)
            added.append(new)
        elif status == "A":
            added.append(path)
        elif status == "M":
            modified.append(path)
        elif status == "D":
            deleted.append(path)
    return {"added": added, "modified": modified, "deleted": deleted}


async def main() -> int:
    base_ref = os.environ.get("BASE_REF", "main~1")
    head_ref = os.environ.get("HEAD_REF", "HEAD")
    changes = changed_docs(base_ref, head_ref)

    if not any(changes.values()):
        print("No doc changes — nothing to sync.")
        return 0

    print(f"+{len(changes['added'])} ~{len(changes['modified'])} "
          f"-{len(changes['deleted'])}")

    async with GraphRAG(
        connection=ConnectionConfig.from_url(os.environ["FALKORDB_URL"]),
        llm=LiteLLM(model="openai/gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
    ) as rag:
        result = await rag.apply_changes(
            added=changes["added"],
            modified=changes["modified"],
            deleted=changes["deleted"],
        )
        await rag.finalize()

    # Surface per-file failures — apply_changes never raises.
    failed = [
        e for e in (result.added + result.modified + result.deleted)
        if not e.is_success
    ]
    for entry in failed:
        print(f"FAIL: {entry.error_type}: {entry.error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

## GitHub Actions wrapper

```yaml
name: Sync docs graph
on:
  push:
    branches: [main]
    paths: ["docs/**/*.md"]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2          # so HEAD~1 is reachable
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install graphrag-sdk[litellm]
      - run: python scripts/sync_docs_graph.py
        env:
          BASE_REF: ${{ github.event.before }}
          HEAD_REF: ${{ github.event.after }}
          FALKORDB_URL: ${{ secrets.FALKORDB_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Why this is cheap

Three properties make this pattern affordable:

1. **`update()` short-circuits unchanged content.** Each modified file's SHA-256 is compared to the stored hash; matching files exit in one Cypher query with `no_op=True`. A "touch" commit that didn't change a file is effectively free.
2. **Per-file errors don't blow up the batch.** Each file's outcome is a `BatchEntry` with either `.result` or `.error` / `.error_type`. A flaky network on one file doesn't lose the other 99.
3. **`finalize()` is called once.** Its cost is O(graph size), independent of how many files changed — calling it per file would be the easy mistake that makes the workflow O(files × graph size). The script puts it after the loop deliberately.

## Operational notes

- **Set a deploy lock** if multiple PRs can merge in flight. `apply_changes` is safe in isolation, but two concurrent runs can race on `finalize()` and end up with duplicate entities to merge. A simple `concurrency: { group: docs-sync, cancel-in-progress: false }` on the workflow is usually enough.
- **Cache the model and embedder providers.** Don't reinstantiate `GraphRAG` in a loop — connection pool warmup and config validation are amortised across calls.
- **Log `BatchEntry` failures.** The default surfacing — `error_type` plus formatted `error` — is enough for triage. Surface the failure count to your CI metrics so a regression in the LLM provider doesn't silently degrade the graph.

## See also

- [Concepts → Incremental updates](../concepts/incremental-updates) — the mental model and crash-safety guarantees.
- [API Reference → GraphRAG](../api-reference/graphrag#apply_changes) — `apply_changes`, `update`, `delete_document`, `finalize`.
- [API Reference → Result types](../api-reference/result-types) — `BatchEntry`, `ApplyChangesResult`, `UpdateResult`.
