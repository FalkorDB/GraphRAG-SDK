"""Generate the bundled Schema.org subset used by ``SchemaOrgCatalog``.

Downloads the latest Schema.org JSON-LD vocabulary, walks ``@graph``, and
extracts a *curated subset* of types — their properties (typed for the
SDK) and the relations between them — into a single JSON file the
catalog can load lazily at runtime.

Why a generator rather than hand-typed data
-------------------------------------------
The previous ``Ontology.from_schema_org()`` literal was misleading: it
claimed Schema.org provenance while actually being labels I typed. This
script earns the provenance — the output is real Schema.org data, the
SDK only chooses which subset to bundle. Regenerate after a Schema.org
release or whenever the curated subset changes.

Output
------
``graphrag_sdk/src/graphrag_sdk/data/schema_org.json`` — committed to the
repo. The SDK reads this via ``importlib.resources``; no network at
runtime.

Run::

    python scripts/generate_schema_org_data.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

SCHEMA_ORG_URL = "https://schema.org/version/latest/schemaorg-current-https.jsonld"

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "graphrag_sdk"
    / "src"
    / "graphrag_sdk"
    / "data"
    / "schema_org.json"
)

# Curated subset — the small set of common types we ship. Tracks what
# barakb/text-to-rdf's system prompt hardcodes plus a few obvious
# additions. Expanding the subset is one line.
CURATED_TYPES = frozenset(
    {
        "Person",
        "Organization",
        "EducationalOrganization",
        "Place",
        "Country",
        "City",
        "Event",
        "CreativeWork",
        "Article",
        "Book",
    }
)

# Schema.org's primitive range types -> SDK Attribute types. Property
# whose range is one of these is an attribute; range that's another
# entity-class is a relation.
RANGE_TO_SDK_TYPE = {
    "Text": "STRING",
    "URL": "STRING",
    "Date": "DATE",
    "DateTime": "DATE",
    "Time": "STRING",
    "Integer": "INTEGER",
    "Number": "FLOAT",
    "Boolean": "BOOLEAN",
}


def schema_id_to_name(id_str: str) -> str:
    """Convert ``"schema:Person"`` or ``"https://schema.org/Person"`` -> ``"Person"``."""
    if id_str.startswith("schema:"):
        return id_str[len("schema:") :]
    if id_str.startswith("https://schema.org/"):
        return id_str[len("https://schema.org/") :]
    return id_str


def camel_to_snake(name: str) -> str:
    """``birthDate`` -> ``birth_date``."""
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _as_list(value):
    """Schema.org's JSON-LD uses single dict or list-of-dicts — normalise."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _comment_string(value) -> str:
    """``rdfs:comment`` can be a string, a list, or a ``{@value, @language}`` dict."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("@value", "")).strip()
    if isinstance(value, list):
        return _comment_string(value[0]) if value else ""
    return str(value).strip()


def main() -> int:
    print(f"Downloading {SCHEMA_ORG_URL}...")
    try:
        with urllib.request.urlopen(SCHEMA_ORG_URL, timeout=60) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1

    graph = payload.get("@graph", [])
    print(f"received {len(graph)} entries")

    types: dict[str, dict] = {}
    properties_by_domain: dict[str, list[dict]] = {}

    for item in graph:
        item_id = item.get("@id", "")
        item_types = _as_list(item.get("@type"))

        if "rdfs:Class" in item_types:
            name = schema_id_to_name(item_id)
            if name in CURATED_TYPES:
                types[name] = {
                    "label": name,
                    "canonical_uri": f"https://schema.org/{name}",
                    "description": _comment_string(item.get("rdfs:comment")),
                    "properties": [],
                }
            continue

        if "rdf:Property" in item_types:
            domains = [
                schema_id_to_name(d.get("@id", ""))
                for d in _as_list(item.get("schema:domainIncludes"))
            ]
            ranges = [
                schema_id_to_name(r.get("@id", ""))
                for r in _as_list(item.get("schema:rangeIncludes"))
            ]
            if not (domains and ranges):
                continue
            for domain in domains:
                if domain in CURATED_TYPES:
                    properties_by_domain.setdefault(domain, []).append(
                        {
                            "schema_name": schema_id_to_name(item_id),
                            "description": _comment_string(item.get("rdfs:comment")),
                            "domains": domains,
                            "ranges": ranges,
                        }
                    )

    # Build attributes + relations.
    relations: list[dict] = []
    relation_seen: set[tuple[str, str, str]] = set()

    for type_name in CURATED_TYPES:
        if type_name not in types:
            continue
        for prop in properties_by_domain.get(type_name, []):
            primitive_ranges = [r for r in prop["ranges"] if r in RANGE_TO_SDK_TYPE]
            entity_ranges = [r for r in prop["ranges"] if r in CURATED_TYPES]

            if primitive_ranges:
                sdk_type = RANGE_TO_SDK_TYPE[primitive_ranges[0]]
                snake = camel_to_snake(prop["schema_name"])
                if any(a["name"] == snake for a in types[type_name]["properties"]):
                    continue
                types[type_name]["properties"].append(
                    {
                        "name": snake,
                        "type": sdk_type,
                        "schema_name": prop["schema_name"],
                        "description": (prop["description"] or "")[:200],
                    }
                )

            if entity_ranges:
                for target in entity_ranges:
                    label = camel_to_snake(prop["schema_name"]).upper()
                    key = (label, type_name, target)
                    if key in relation_seen:
                        continue
                    relation_seen.add(key)
                    relations.append(
                        {
                            "label": label,
                            "schema_name": prop["schema_name"],
                            "source": type_name,
                            "target": target,
                            "description": (prop["description"] or "")[:200],
                        }
                    )

    output = {
        "source_url": SCHEMA_ORG_URL,
        "curated_types": sorted(CURATED_TYPES),
        "types": dict(sorted(types.items())),
        "relations": sorted(relations, key=lambda r: (r["label"], r["source"], r["target"])),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n")

    print(f"wrote {OUTPUT_PATH}")
    print(f"  {len(types)} types")
    total_props = sum(len(t["properties"]) for t in types.values())
    print(f"  {total_props} attributes")
    print(f"  {len(relations)} relations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
