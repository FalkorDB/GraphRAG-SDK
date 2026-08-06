"""Shared harness for the structured-ingestion spikes.

Importing this module pins *this worktree's* ``src`` ahead of any installed
graphrag_sdk, so spikes exercise the branch under review rather than whatever
``pip install -e`` happens to point at.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# poc/structured-ingestion/_harness/env.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
SDK_SRC = REPO_ROOT / "graphrag_sdk" / "src"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

FALKOR_HOST = os.getenv("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKOR_PORT", "6379"))


def sdk_is_local() -> bool:
    """True when ``graphrag_sdk`` resolves inside this worktree."""
    import graphrag_sdk

    return str(SDK_SRC) in str(Path(graphrag_sdk.__file__).resolve())


def falkor_available() -> bool:
    """Cheap reachability probe so DB spikes can skip instead of exploding."""
    import socket

    try:
        with socket.create_connection((FALKOR_HOST, FALKOR_PORT), timeout=2):
            return True
    except OSError:
        return False


def connection(graph_name: str):
    """A FalkorDBConnection against a throwaway ``poc_``-prefixed graph."""
    from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection

    if not graph_name.startswith("poc_"):
        raise ValueError(f"refusing non-throwaway graph name {graph_name!r}; use a poc_ prefix")
    return FalkorDBConnection(
        ConnectionConfig(host=FALKOR_HOST, port=FALKOR_PORT, graph_name=graph_name)
    )


async def reset_graph(conn) -> None:
    """Drop everything in the connected graph. Only ever called on poc_ graphs."""
    await conn.query("MATCH (n) DETACH DELETE n")


class FakeEmbedder:
    """Deterministic, dependency-free embedder.

    Spikes care about *what gets written and traversed*, never about vector
    quality, so a hash-derived vector is sufficient and keeps the whole folder
    runnable with no API keys.
    """

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    def _vec(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dimension)]

    async def aembed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vec(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self._vec(t) for t in texts]


# ── tiny reporting helpers ───────────────────────────────────────


class Report:
    """Collects PASS/FAIL checks so a spike ends with a verdict, not a wall of prints."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.lines: list[str] = []
        self.failures = 0
        print(f"\n=== {title} ===")

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        line = f"[{mark}] {label}" + (f" — {detail}" if detail else "")
        self.lines.append(line)
        print(line)
        return ok

    def note(self, text: str) -> None:
        self.lines.append(f"       {text}")
        print(f"       {text}")

    def verdict(self) -> int:
        status = "OK" if self.failures == 0 else f"{self.failures} FAILED"
        print(f"--- {self.title}: {status}")
        return 1 if self.failures else 0
