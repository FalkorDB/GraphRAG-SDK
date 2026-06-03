# GraphRAG SDK — Discovery: ontology catalogs
#
# A Catalog is the source of truth for ontology vocabulary, queried by
# the grounded discovery pipeline. It answers two questions:
#
#   - link_entity(name)        : what type(s) is this entity?
#   - lookup(type_name)        : what's this type's schema?
#   - relations_among(types)   : what relations exist among these types?
#
# The grounded pipeline calls these to turn "names NER found in the
# corpus" into a typed Ontology. The catalog stays agnostic about NER —
# the pipeline runs GLiNER (or any EntityExtractor) and passes the
# resulting mention strings here for resolution.
#
# Built-in implementation: ``DBpediaCatalog`` queries DBpedia's SPARQL
# endpoint for entity→type lookups and fetches Schema.org's JSON-LD
# vocabulary for type→schema lookups. Both live, both per-request
# cached, no bundled snapshots.

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from graphrag_sdk.core.models import Attribute, Entity, Relation

logger = logging.getLogger(__name__)


class Catalog(ABC):
    """Source-of-truth ontology vocabulary queried by the grounded pipeline.

    Concrete implementations decide *how* to resolve the three queries.
    Built-in :class:`DBpediaCatalog` uses DBpedia SPARQL + Schema.org;
    custom catalogs can plug in any KB / SPARQL endpoint / domain
    vocabulary.
    """

    @abstractmethod
    def link_entity(self, name: str) -> list[str]:
        """Return the type names this catalog thinks ``name`` belongs to.

        Empty list if no match found. Types should be normalised to
        whatever vocabulary the catalog's :py:meth:`lookup` understands
        (so ``DBpediaCatalog`` returns Schema.org-style local names
        like ``"Person"`` rather than full URIs).
        """

    @abstractmethod
    def lookup(self, type_name: str) -> Entity | None:
        """Return the catalog's schema for a type, or ``None`` if unknown.

        The returned :class:`Entity` carries the type's attributes
        (typed for the SDK) and a description that records provenance
        (canonical URI of the source).
        """

    @abstractmethod
    def relations_among(self, type_names: Iterable[str]) -> list[Relation]:
        """All relations whose source AND target labels are in ``type_names``.

        Called once per discovery run after the corpus has been linked,
        to pull only the relations relevant to the detected subset.
        """


# ── DBpediaCatalog ─────────────────────────────────────────────────


class DBpediaFetchError(RuntimeError):
    """Raised when :class:`DBpediaCatalog` cannot reach DBpedia or Schema.org."""


def _default_cache_path() -> Path:
    """``$XDG_CACHE_HOME/graphrag-sdk/schema_org.json`` (or ``~/.cache/...``)."""
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(cache_home) / "graphrag-sdk" / "schema_org.json"


def _http_get_json(url: str, *, headers: dict | None = None, timeout: int = 30) -> dict:
    """Minimal HTTP GET that returns parsed JSON. Raises on any failure."""
    req = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class DBpediaCatalog(Catalog):
    """Live catalog: DBpedia SPARQL for entity→type, Schema.org for type→schema.

    On ``link_entity(name)``:
      - SPARQL query DBpedia for entities whose ``rdfs:label`` matches
        ``name`` (with the ``@en`` language tag), filtered to
        ``http://dbpedia.org/ontology/`` types.
      - Return the local names of those types (so ``dbo:Person`` becomes
        ``"Person"``).

    On ``lookup(type_name)`` / ``relations_among(...)``:
      - First call lazily fetches the full Schema.org JSON-LD vocabulary
        from ``schema.org`` and processes it (subClassOf inheritance
        applied so e.g. ``Article`` inherits ``CreativeWork``'s
        properties).
      - Processed result is cached on disk under
        ``$XDG_CACHE_HOME/graphrag-sdk/schema_org.json``. TTL defaults
        to 30 days.

    No bundled data files. No hardcoded type list. Schema.org's full
    vocabulary is the type space; DBpedia decides which subset of it
    the corpus contains.

    Args:
        sparql_endpoint: Override the DBpedia SPARQL URL. Defaults to
            the canonical ``https://dbpedia.org/sparql``.
        schema_org_url: Override the Schema.org JSON-LD URL.
        cache_path: Where to read/write the processed Schema.org cache.
        cache_ttl_days: Schema.org cache TTL. ``None`` = never expire;
            ``0`` = re-fetch every construction.

    Raises:
        DBpediaFetchError: If DBpedia or Schema.org is unreachable and
            the required data isn't cached.

    Example::

        from graphrag_sdk.discovery.catalog import DBpediaCatalog

        catalog = DBpediaCatalog()
        catalog.link_entity("Albert Einstein")
        # ['Person', 'Scientist', ...]   ← real DBpedia types

        catalog.lookup("Person")
        # Entity(label='Person', properties=[...])  ← from Schema.org

        catalog.relations_among({"Person", "Organization"})
        # [Relation(label='WORKS_FOR', ...), ...]   ← from Schema.org
    """

    DEFAULT_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    DEFAULT_SCHEMA_ORG_URL = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
    DEFAULT_CACHE_TTL_DAYS: int | None = 30
    DBPEDIA_ONTOLOGY_PREFIX = "http://dbpedia.org/ontology/"

    _RANGE_TO_SDK_TYPE: dict[str, str] = {
        "Text": "STRING",
        "URL": "STRING",
        "Date": "DATE",
        "DateTime": "DATE",
        "Time": "STRING",
        "Integer": "INTEGER",
        "Number": "FLOAT",
        "Boolean": "BOOLEAN",
    }

    def __init__(
        self,
        *,
        sparql_endpoint: str | None = None,
        schema_org_url: str | None = None,
        cache_path: str | Path | None = None,
        cache_ttl_days: int | None = DEFAULT_CACHE_TTL_DAYS,
    ) -> None:
        self._sparql_endpoint = sparql_endpoint or self.DEFAULT_SPARQL_ENDPOINT
        self._schema_org_url = schema_org_url or self.DEFAULT_SCHEMA_ORG_URL
        self._cache_path = Path(cache_path) if cache_path is not None else _default_cache_path()
        self._cache_ttl_days = cache_ttl_days
        self._schema_org: dict | None = None
        self._entity_cache: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    # ── Catalog API ────────────────────────────────────────────────

    def link_entity(self, name: str) -> list[str]:
        if not name:
            return []
        with self._lock:
            cached = self._entity_cache.get(name)
        if cached is not None:
            return cached
        types = self._query_dbpedia_types(name)
        with self._lock:
            self._entity_cache[name] = types
        return types

    def lookup(self, type_name: str) -> Entity | None:
        type_record = self._load_schema_org()["types"].get(type_name)
        if type_record is None:
            return None
        properties: list[Attribute] = []
        for p in type_record["properties"]:
            schema_name = p.get("schema_name", p["name"])
            base_desc = p.get("description") or ""
            attr_desc = f"Schema.org {schema_name}"
            if base_desc:
                attr_desc = f"{attr_desc} — {base_desc}"
            properties.append(Attribute(name=p["name"], type=p["type"], description=attr_desc))
        description = type_record.get("description") or ""
        uri = type_record.get("canonical_uri")
        if uri:
            description = (
                f"{description} (Schema.org: {uri})" if description else f"Schema.org: {uri}"
            )
        return Entity(
            label=type_record["label"],
            description=description or None,
            properties=properties,
        )

    def relations_among(self, type_names: Iterable[str]) -> list[Relation]:
        wanted = set(type_names)
        raw = self._load_schema_org()
        by_label: dict[str, dict] = {}
        for rec in raw["relations"]:
            source, target = rec["source"], rec["target"]
            if source not in wanted or target not in wanted:
                continue
            entry = by_label.setdefault(
                rec["label"],
                {
                    "patterns": [],
                    "schema_name": rec.get("schema_name"),
                    "description": rec.get("description") or "",
                },
            )
            pattern = (source, target)
            if pattern not in entry["patterns"]:
                entry["patterns"].append(pattern)
        relations: list[Relation] = []
        for label, entry in sorted(by_label.items()):
            schema_name = entry["schema_name"] or label
            base_desc = entry["description"]
            desc = f"Schema.org {schema_name}"
            if base_desc:
                desc = f"{desc} — {base_desc}"
            relations.append(Relation(label=label, description=desc, patterns=entry["patterns"]))
        return relations

    # ── DBpedia SPARQL: link_entity helper ────────────────────────

    def _query_dbpedia_types(self, name: str) -> list[str]:
        """SPARQL DBpedia for an entity matching ``name`` and return
        the local names of its ``dbo:`` types.

        We map ``http://dbpedia.org/ontology/Person`` → ``"Person"`` —
        the local name. Schema.org and DBpedia ontology types share the
        same names for the common entity types (Person, Organization,
        Place, Event, etc.), so the same string is used to look up the
        Schema.org schema in :py:meth:`lookup`.
        """
        # ``rdfs:label "name"@en`` is the canonical English-label query.
        # Filtering to ``dbpedia.org/ontology/`` keeps the result small
        # and avoids tags like ``wikidata:Q5`` that the Schema.org
        # lookup couldn't use anyway.
        query = (
            "SELECT DISTINCT ?type WHERE { "
            f'  ?entity rdfs:label "{self._sparql_escape(name)}"@en . '
            "  ?entity a ?type . "
            f'  FILTER(STRSTARTS(STR(?type), "{self.DBPEDIA_ONTOLOGY_PREFIX}")) '
            "} LIMIT 50"
        )
        url = (
            f"{self._sparql_endpoint}?"
            f"query={urllib.parse.quote(query)}&format=application/sparql-results%2Bjson"
        )
        try:
            payload = _http_get_json(url, headers={"Accept": "application/sparql-results+json"})
        except Exception as exc:
            logger.warning(
                "DBpediaCatalog: SPARQL lookup for %r failed: %s — returning no types",
                name,
                exc,
            )
            return []
        bindings = payload.get("results", {}).get("bindings", [])
        types: list[str] = []
        seen: set[str] = set()
        for b in bindings:
            uri = b.get("type", {}).get("value", "")
            if not uri.startswith(self.DBPEDIA_ONTOLOGY_PREFIX):
                continue
            local = uri[len(self.DBPEDIA_ONTOLOGY_PREFIX) :]
            if local and local not in seen:
                seen.add(local)
                types.append(local)
        return types

    @staticmethod
    def _sparql_escape(value: str) -> str:
        """Escape a string for inclusion in a SPARQL double-quoted literal."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    # ── Schema.org: load / fetch / process / cache ────────────────

    def _load_schema_org(self) -> dict:
        if self._schema_org is not None:
            return self._schema_org
        with self._lock:
            if self._schema_org is not None:
                return self._schema_org
            if self._is_cache_fresh():
                try:
                    self._schema_org = json.loads(self._cache_path.read_text(encoding="utf-8"))
                    return self._schema_org
                except Exception as exc:
                    logger.warning(
                        "DBpediaCatalog: Schema.org cache at %s is corrupt (%s); refetching",
                        self._cache_path,
                        exc,
                    )
            self._schema_org = self._fetch_and_process_schema_org()
            self._write_cache(self._schema_org)
            return self._schema_org

    def _is_cache_fresh(self) -> bool:
        if not self._cache_path.is_file():
            return False
        if self._cache_ttl_days is None:
            return True
        if self._cache_ttl_days <= 0:
            return False
        age_seconds = time.time() - self._cache_path.stat().st_mtime
        return age_seconds < self._cache_ttl_days * 86400

    def _write_cache(self, payload: dict) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "DBpediaCatalog: failed to write Schema.org cache %s: %s",
                self._cache_path,
                exc,
            )

    def _fetch_and_process_schema_org(self) -> dict:
        logger.info(
            "DBpediaCatalog: fetching Schema.org vocabulary from %s",
            self._schema_org_url,
        )
        try:
            payload = _http_get_json(self._schema_org_url, timeout=60)
        except Exception as exc:
            raise DBpediaFetchError(
                f"Failed to fetch Schema.org vocabulary from {self._schema_org_url}: {exc}. "
                "There is no offline fallback — pre-warm the cache by running with "
                "network connectivity, or pass a pre-populated ``cache_path=``."
            ) from exc
        graph = payload.get("@graph", [])
        if not graph:
            raise DBpediaFetchError(
                f"Schema.org response from {self._schema_org_url} had no @graph"
            )
        return self._process_schema_org_graph(graph)

    def _process_schema_org_graph(self, graph: list[dict]) -> dict:
        """Walk Schema.org's ``@graph`` once and build queryable tables.

        Applies ``rdfs:subClassOf`` inheritance so subclasses inherit
        base-class members. No curated-type filter: every Schema.org
        rdfs:Class becomes a queryable type. DBpedia decides which
        subset is actually used per discovery run.
        """
        types: dict[str, dict] = {}
        parents_by_class: dict[str, set[str]] = {}
        properties_by_domain: dict[str, list[dict]] = {}

        for item in graph:
            item_id = item.get("@id", "")
            item_types = self._as_list(item.get("@type"))

            if "rdfs:Class" in item_types:
                name = self._schema_id_to_name(item_id)
                parents_by_class[name] = {
                    self._schema_id_to_name(parent.get("@id", ""))
                    for parent in self._as_list(item.get("rdfs:subClassOf"))
                    if isinstance(parent, dict) and parent.get("@id")
                }
                types[name] = {
                    "label": name,
                    "canonical_uri": f"https://schema.org/{name}",
                    "description": self._comment_string(item.get("rdfs:comment")),
                    "properties": [],
                }
                continue

            if "rdf:Property" in item_types:
                domains = [
                    self._schema_id_to_name(d.get("@id", ""))
                    for d in self._as_list(item.get("schema:domainIncludes"))
                ]
                ranges = [
                    self._schema_id_to_name(r.get("@id", ""))
                    for r in self._as_list(item.get("schema:rangeIncludes"))
                ]
                if not (domains and ranges):
                    continue
                for domain in domains:
                    properties_by_domain.setdefault(domain, []).append(
                        {
                            "schema_name": self._schema_id_to_name(item_id),
                            "description": self._comment_string(item.get("rdfs:comment")),
                            "domains": domains,
                            "ranges": ranges,
                        }
                    )

        def _ancestors(class_name: str) -> set[str]:
            seen: set[str] = set()
            stack = list(parents_by_class.get(class_name, set()))
            while stack:
                parent = stack.pop()
                if parent in seen:
                    continue
                seen.add(parent)
                stack.extend(parents_by_class.get(parent, set()))
            return seen

        all_type_names = set(types.keys())
        relations: list[dict] = []
        relation_seen: set[tuple[str, str, str]] = set()

        for type_name in all_type_names:
            effective_domains = {type_name, *_ancestors(type_name)}
            for domain in effective_domains:
                for prop in properties_by_domain.get(domain, []):
                    primitive_ranges = [r for r in prop["ranges"] if r in self._RANGE_TO_SDK_TYPE]
                    entity_ranges = [r for r in prop["ranges"] if r in all_type_names]

                    if primitive_ranges:
                        sdk_type = self._RANGE_TO_SDK_TYPE[primitive_ranges[0]]
                        snake = self._camel_to_snake(prop["schema_name"])
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
                            label = self._camel_to_snake(prop["schema_name"]).upper()
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

        return {
            "source_url": self._schema_org_url,
            "fetched_at": int(time.time()),
            "types": dict(sorted(types.items())),
            "relations": sorted(relations, key=lambda r: (r["label"], r["source"], r["target"])),
        }

    # ── JSON-LD parsing helpers ───────────────────────────────────

    @staticmethod
    def _schema_id_to_name(id_str: str) -> str:
        if id_str.startswith("schema:"):
            return id_str[len("schema:") :]
        if id_str.startswith("https://schema.org/"):
            return id_str[len("https://schema.org/") :]
        return id_str

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        out: list[str] = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0 and not name[i - 1].isupper():
                out.append("_")
            out.append(ch.lower())
        return "".join(out)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _comment_string(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return str(value.get("@value", "")).strip()
        if isinstance(value, list):
            return DBpediaCatalog._comment_string(value[0]) if value else ""
        return str(value).strip()
