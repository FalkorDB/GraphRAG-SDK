# GraphRAG SDK — Discovery: ontology catalogs
#
# A Catalog is a *source of truth* for ontology vocabulary: given a
# type name (``"Person"``), it returns that type's schema — attributes,
# relations to other types, canonical URI. The catalog decouples
# discovery from any particular vocabulary; the grounded discovery
# path (``Ontology.from_sources(method="grounded")``) takes a catalog
# as a parameter, runs NER to learn which types the corpus contains,
# and asks the catalog for the schema of those types.
#
# :class:`SchemaOrgCatalog` fetches the published Schema.org JSON-LD
# vocabulary live from https://schema.org on first use and caches the
# *processed* result (already filtered to the curated subset and with
# ``subClassOf`` inheritance applied) under
# ``~/.cache/graphrag-sdk/schema_org.json``. Subsequent instantiations
# in the same process or on the same machine hit the cache. There is
# no offline fallback — if both the network and the cache are
# unavailable, ``SchemaOrgFetchError`` is raised. Users who need
# offline behaviour pre-warm the cache by running discovery once with
# connectivity, or supply a pre-populated ``cache_path``.

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from graphrag_sdk.core.models import Attribute, Entity, Relation

logger = logging.getLogger(__name__)


class Catalog(ABC):
    """A source-of-truth ontology vocabulary the grounded pipeline can query.

    Concrete implementations may consult a live web endpoint, a SPARQL
    store, a domain catalog the user wrote — the abstraction lets
    ``Ontology.from_sources(method="grounded")`` work against any
    vocabulary without knowing where it came from.
    """

    @abstractmethod
    def known_types(self) -> set[str]:
        """Every type name this catalog can answer about."""

    @abstractmethod
    def lookup(self, type_name: str) -> Entity | None:
        """Return the catalog's view of ``type_name``, or ``None`` if unknown.

        The returned :class:`Entity` includes the type's attributes
        (translated to the SDK's typed-property model) and its
        ``description`` field carries the catalog's natural-language
        description. The canonical source URI, when the catalog has
        one, is recorded in ``description`` for traceability.
        """

    @abstractmethod
    def relations_among(self, type_names: Iterable[str]) -> list[Relation]:
        """All relations whose source AND target labels are in ``type_names``.

        The grounded pipeline calls this after collecting the set of
        types the corpus contains, to pull only the relations relevant
        to that subset. Patterns are merged by relation label.
        """


# ── SchemaOrgCatalog ───────────────────────────────────────────────


class SchemaOrgFetchError(RuntimeError):
    """Raised when ``SchemaOrgCatalog`` cannot fetch from Schema.org and has no fresh cache."""


def _default_cache_path() -> Path:
    """Return ``$XDG_CACHE_HOME/graphrag-sdk/schema_org.json`` (or ``~/.cache/...``)."""
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(cache_home) / "graphrag-sdk" / "schema_org.json"


class SchemaOrgCatalog(Catalog):
    """Catalog backed by a live fetch of the Schema.org JSON-LD vocabulary.

    On first call to :py:meth:`known_types`, :py:meth:`lookup`, or
    :py:meth:`relations_among`, the catalog either reads its on-disk
    cache (if fresh) or downloads the live Schema.org vocabulary from
    ``https://schema.org``, processes it into the SDK's typed shape,
    and writes the processed result to the cache. There is **no
    bundled snapshot** — the source of truth is always Schema.org.

    Args:
        url: Override the Schema.org JSON-LD endpoint. The published
            URL of the latest vocabulary by default.
        cache_path: Where to read/write the processed result. Defaults
            to ``$XDG_CACHE_HOME/graphrag-sdk/schema_org.json``.
        cache_ttl_days: Re-fetch from Schema.org when the on-disk cache
            is older than this. ``None`` disables expiry (cache forever
            once written). ``0`` forces re-fetch on every constructor.
        curated_types: The Schema.org type names to include in the
            output. Defaults to a curated subset of 10 common types
            (Person, Organization, EducationalOrganization, Place,
            Country, City, Event, CreativeWork, Article, Book).
            ``subClassOf`` inheritance applies regardless: ``Article``
            inherits ``CreativeWork``'s attributes.

    Raises:
        SchemaOrgFetchError: If the network fetch fails AND the cache
            is either missing or stale beyond the TTL. There is no
            offline fallback.

    Example::

        from graphrag_sdk import Ontology
        from graphrag_sdk.discovery.catalog import SchemaOrgCatalog

        catalog = SchemaOrgCatalog()        # uses default cache path
        catalog.known_types()                # {"Person", "Organization", ...}
        catalog.lookup("Person")             # Entity(label="Person", properties=[...])
        catalog.relations_among({"Person", "Organization"})
        #   → [Relation(label="WORKS_FOR", patterns=[("Person", "Organization")]), ...]
    """

    DEFAULT_URL = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
    DEFAULT_CACHE_TTL_DAYS: int | None = 30
    DEFAULT_CURATED_TYPES: frozenset[str] = frozenset(
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

    # Schema.org primitive ranges → SDK ``Attribute.type``.
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
        url: str | None = None,
        cache_path: str | Path | None = None,
        cache_ttl_days: int | None = DEFAULT_CACHE_TTL_DAYS,
        curated_types: Iterable[str] | None = None,
    ) -> None:
        self._url = url or self.DEFAULT_URL
        self._cache_path = Path(cache_path) if cache_path is not None else _default_cache_path()
        self._cache_ttl_days = cache_ttl_days
        self._curated_types = (
            frozenset(curated_types) if curated_types is not None else self.DEFAULT_CURATED_TYPES
        )
        self._loaded: dict | None = None
        self._lock = threading.Lock()

    # ── Public Catalog API ─────────────────────────────────────────

    def known_types(self) -> set[str]:
        return set(self._load()["types"].keys())

    def lookup(self, type_name: str) -> Entity | None:
        type_record = self._load()["types"].get(type_name)
        if type_record is None:
            return None

        properties: list[Attribute] = []
        for p in type_record["properties"]:
            schema_name = p.get("schema_name", p["name"])
            base_desc = p.get("description") or ""
            attr_desc = f"Schema.org {schema_name}"
            if base_desc:
                attr_desc = f"{attr_desc} — {base_desc}"
            properties.append(
                Attribute(
                    name=p["name"],
                    type=p["type"],
                    description=attr_desc,
                )
            )

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
        # Group by label so a relation with multiple patterns becomes one Relation.
        by_label: dict[str, dict] = {}
        for rec in self._load()["relations"]:
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
            relations.append(
                Relation(
                    label=label,
                    description=desc,
                    patterns=entry["patterns"],
                )
            )
        return relations

    # ── Load + fetch ───────────────────────────────────────────────

    def _load(self) -> dict:
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            if self._is_cache_fresh():
                try:
                    self._loaded = json.loads(self._cache_path.read_text(encoding="utf-8"))
                    logger.debug("SchemaOrgCatalog: served from cache %s", self._cache_path)
                    return self._loaded
                except Exception as exc:
                    logger.warning(
                        "SchemaOrgCatalog: cache at %s is corrupt (%s); refetching",
                        self._cache_path,
                        exc,
                    )
            self._loaded = self._fetch_and_process()
            self._write_cache(self._loaded)
            return self._loaded

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
            logger.debug("SchemaOrgCatalog: wrote processed cache to %s", self._cache_path)
        except Exception as exc:
            # A cache write failure is non-fatal — the in-process load
            # still succeeded — but worth surfacing for debug.
            logger.warning(
                "SchemaOrgCatalog: failed to write cache %s: %s",
                self._cache_path,
                exc,
            )

    def _fetch_and_process(self) -> dict:
        """Download Schema.org's JSON-LD vocabulary and process it."""
        logger.info("SchemaOrgCatalog: fetching Schema.org vocabulary from %s", self._url)
        try:
            with urllib.request.urlopen(self._url, timeout=60) as response:
                payload = json.load(response)
        except Exception as exc:
            raise SchemaOrgFetchError(
                f"Failed to fetch Schema.org vocabulary from {self._url}: {exc}. "
                "There is no offline fallback for SchemaOrgCatalog — pre-warm "
                "the cache by running with network connectivity, or pass a "
                "pre-populated ``cache_path=`` to the constructor."
            ) from exc
        graph = payload.get("@graph", [])
        if not graph:
            raise SchemaOrgFetchError(
                f"Schema.org response from {self._url} did not contain an @graph"
            )
        return self._process_graph(graph)

    # ── JSON-LD processing ────────────────────────────────────────

    @staticmethod
    def _schema_id_to_name(id_str: str) -> str:
        """``"schema:Person"`` or ``"https://schema.org/Person"`` -> ``"Person"``."""
        if id_str.startswith("schema:"):
            return id_str[len("schema:") :]
        if id_str.startswith("https://schema.org/"):
            return id_str[len("https://schema.org/") :]
        return id_str

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """``"birthDate"`` -> ``"birth_date"``."""
        out: list[str] = []
        for i, ch in enumerate(name):
            if ch.isupper() and i > 0 and not name[i - 1].isupper():
                out.append("_")
            out.append(ch.lower())
        return "".join(out)

    @staticmethod
    def _as_list(value):
        """Schema.org's JSON-LD uses single dict or list-of-dicts — normalise."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _comment_string(value) -> str:
        """``rdfs:comment`` may be string, list, or ``{@value, @language}`` dict."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return str(value.get("@value", "")).strip()
        if isinstance(value, list):
            return SchemaOrgCatalog._comment_string(value[0]) if value else ""
        return str(value).strip()

    def _process_graph(self, graph: list[dict]) -> dict:
        """Walk Schema.org's ``@graph`` and produce the processed lookup tables.

        For each curated type, pull properties whose ``domainIncludes``
        names the type itself OR any of its ``rdfs:subClassOf``
        ancestors. So ``Article`` inherits ``CreativeWork``'s attributes;
        ``EducationalOrganization`` inherits ``Organization``'s; etc.
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
                if name in self._curated_types:
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

        # Transitive ancestors of each curated type for subClassOf inheritance.
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

        # Build attributes + relations.
        relations: list[dict] = []
        relation_seen: set[tuple[str, str, str]] = set()

        for type_name in self._curated_types:
            if type_name not in types:
                continue
            effective_domains = {type_name, *_ancestors(type_name)}
            for domain in effective_domains:
                for prop in properties_by_domain.get(domain, []):
                    primitive_ranges = [r for r in prop["ranges"] if r in self._RANGE_TO_SDK_TYPE]
                    entity_ranges = [r for r in prop["ranges"] if r in self._curated_types]

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
            "source_url": self._url,
            "fetched_at": int(time.time()),
            "curated_types": sorted(self._curated_types),
            "types": dict(sorted(types.items())),
            "relations": sorted(relations, key=lambda r: (r["label"], r["source"], r["target"])),
        }
