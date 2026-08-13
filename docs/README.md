# GraphRAG SDK documentation

Published with [Mintlify](https://mintlify.com) at
<https://docs.graphrag.falkordb.com>, and surfaced as a product in the FalkorDB
docs product switcher alongside FalkorDB, FalkorDB Cloud, and FalkorDB
Enterprise.

This directory is the Mintlify content root — `docs.json` lives here, and every
page path in `navigation` is relative to it.

## Authoring rules

- Every page is an `.mdx` file with YAML front matter containing at least
  `title`, plus `description` where it adds value.
- **Do not write an H1 in the body.** Mintlify renders `title` as the H1. Start
  the body at `##`.
- Register every new page in `docs.json`. A page that is not in `navigation` is
  not reachable.
- Internal links are root-relative and omit the extension:
  `[Configuration](/configuration)`.

## MDX gotchas

MDX parses `{`, `}` and `<` as JSX. Outside fenced code blocks and inline code
spans you must escape them, self-close void elements (`<br />`, `<img ... />`),
use `className` instead of `class`, and pass `style` as an object. Use MDX
comments (`{/* ... */}`) rather than HTML comments.

## Components

Prefer Mintlify components over hand-rolled HTML:

- `<CodeGroup>` for the same example in multiple languages.
- `<AccordionGroup>` / `<Accordion>` for FAQs and collapsible argument lists.
- `<Note>`, `<Tip>`, `<Warning>`, `<Info>` for callouts.

## Local development

```bash
npm i -g mint
cd docs
mint dev            # preview at http://localhost:3000
mint broken-links   # verify every internal link resolves
```

`mint` requires an LTS release of Node.

To validate MDX syntax across every page in one pass:

```bash
npm install --no-save @mdx-js/mdx remark-gfm
node scripts/check_mdx.mjs
```

Both checks run in CI via `.github/workflows/docs.yml`.

## Product switcher entry

The `FalkorDB/docs` repository owns the shared product switcher. GraphRAG SDK is
registered there as an external product in `navigation.products`:

```json
{
  "product": "GraphRAG SDK",
  "description": "Build knowledge graphs from documents and query them with natural language.",
  "icon": "share-nodes",
  "href": "https://docs.graphrag.falkordb.com"
}
```

Keep that entry and the mirrored product list in this repo's `docs.json` in
sync whenever a product is added or renamed.
