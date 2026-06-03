# Falkordb.com docs — canonical source

These pages are the source of truth for the GraphRAG-SDK section of
`https://docs.falkordb.com`. The published site repo (`FalkorDB/docs`)
gets a **generated copy** of this tree via `scripts/sync_falkordb_docs.py`.

## Layout

```
docs/site/falkordb/
├── genai-tools/
│   └── graphrag-sdk/        ← mirrors `genai-tools/graphrag-sdk/` on the live site
│       ├── index.md
│       ├── quickstart.md
│       ├── concepts/        ← mental-model pages, no API tables
│       ├── guides/          ← task-oriented how-tos
│       ├── api-reference/   ← per-symbol OpenAI-style reference
│       └── changelog.md
└── .wordlist-additions.txt  ← merged into FalkorDB/docs/.wordlist.txt at sync time
```

## Editing rules

1. **Author here**, never directly in `FalkorDB/docs/genai-tools/graphrag-sdk/`.
   Edits there are wiped on the next sync.
2. **Match signatures exactly** to the `__all__` surface in
   `graphrag_sdk/src/graphrag_sdk/__init__.py`. Anything documented here
   must be importable from `graphrag_sdk` directly. If a signature changes
   in the SDK, update the matching page in the same PR.
3. **Front matter** uses the `just-the-docs` Jekyll theme conventions:
   `title`, `nav_order`, `parent`, `grand_parent`, `description`,
   `has_children`, `redirect_from`. No MDX, no React components.
4. **Code samples** use the existing `_includes/code_tabs.html` helper.
   Python is mandatory; other languages are optional (the SDK is Python-only).

## Local preview

```bash
git clone https://github.com/FalkorDB/docs /tmp/falkordb-docs
cd /tmp/falkordb-docs
git checkout -b docs/graphrag-sdk-preview

# From this repo's root:
python scripts/sync_falkordb_docs.py --target /tmp/falkordb-docs --no-push

cd /tmp/falkordb-docs
bundle install
bundle exec jekyll serve
# open http://127.0.0.1:4000/genai-tools/graphrag-sdk/
```

## Publishing

The `sync-docs` GitHub Action in this repo opens a PR against
`FalkorDB/docs` on release-tag pushes. Manual runs are available via
`workflow_dispatch`. The action always works on a fresh
`sync/graphrag-sdk-<sha>` branch in the target repo — it never pushes to
`main` and never reuses an existing branch.
