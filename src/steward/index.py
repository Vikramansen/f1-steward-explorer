"""SQLite FTS5 index over cached passages."""

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from steward.extract import Passage, extract_source
from steward.fetch import sha256_file
from steward.manifest import Manifest, Source

STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from get got had has have he her him his how i "
    "in into is it its me my of on or our she so than that the their them then there they this "
    "to was we were what when where which while who whom why will with would you your".split()
)

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE documents (
    id TEXT PRIMARY KEY, source_type TEXT NOT NULL, status TEXT NOT NULL,
    event_key TEXT, season INTEGER, json TEXT NOT NULL
);
CREATE TABLE passages (
    evidence_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id),
    ord INTEGER NOT NULL, page INTEGER, kind TEXT NOT NULL, label TEXT NOT NULL,
    text TEXT NOT NULL, quotable INTEGER NOT NULL
);
CREATE VIRTUAL TABLE passages_fts USING fts5(
    evidence_id UNINDEXED, doc_id UNINDEXED, text, title, tokenize = 'porter unicode61'
);
"""


@dataclass(frozen=True)
class DocHit:
    """Best BM25 passage score for one document."""

    doc_id: str
    bm25: float
    best_evidence_id: str


def fts_query(text: str) -> str:
    """Turn free text into an FTS5 OR-query of quoted, non-stopword terms."""
    terms = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def build_index(
    manifest: Manifest, cache_dir: Path, index_path: Path, exclude: Iterable[str] = ()
) -> Path:
    """Build the FTS5 index from the cached PDFs.

    Args:
        manifest: Parsed manifest.
        cache_dir: Directory with cached PDFs.
        index_path: Output SQLite file. It is replaced if it exists.
        exclude: Source IDs to leave out (used for ablation checks).

    Raises:
        FileNotFoundError: If a cached PDF is missing.
        ValueError: If a cached PDF does not match its pinned checksum.
    """
    excluded = set(exclude)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.unlink(missing_ok=True)
    con = sqlite3.connect(index_path)
    con.executescript(SCHEMA)
    for src in manifest.sources:
        if src.id in excluded:
            continue
        if src.cache_file:
            path = cache_dir / src.cache_file
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `steward fetch` first")
            if sha256_file(path) != src.sha256:
                raise ValueError(f"{src.id}: cached file does not match manifest checksum")
        passages, _ = extract_source(src, cache_dir)
        _insert(con, src, passages)
    meta = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "excluded": json.dumps(sorted(excluded)),
        "manifest_version": str(manifest.manifest_version),
    }
    con.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    con.commit()
    con.close()
    logger.info("Built index {} (excluded: {})", index_path, sorted(excluded) or "none")
    return index_path


def _insert(con: sqlite3.Connection, src: Source, passages: list[Passage]) -> None:
    con.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
        (src.id, src.source_type, src.status, src.event_key, src.season, src.model_dump_json()),
    )
    for ord_, p in enumerate(passages):
        con.execute(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (p.evidence_id, p.doc_id, ord_, p.page, p.kind, p.label, p.text, int(p.quotable)),
        )
        con.execute(
            "INSERT INTO passages_fts VALUES (?, ?, ?, ?)",
            (p.evidence_id, p.doc_id, p.text, src.title),
        )


class Index:
    """Read-only view of the built index."""

    def __init__(self, path: Path) -> None:
        """Open an index built by build_index."""
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `steward build` first")
        self.path = path
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.con.row_factory = sqlite3.Row

    def meta(self) -> dict[str, str]:
        """Return build metadata (time, exclusions, version)."""
        return {r["key"]: r["value"] for r in self.con.execute("SELECT key, value FROM meta")}

    def doc_ids(self) -> list[str]:
        """Return the IDs of all indexed documents."""
        return [r["id"] for r in self.con.execute("SELECT id FROM documents ORDER BY id")]

    def source(self, doc_id: str) -> Source | None:
        """Return the manifest entry for a document, or None if it is not indexed."""
        row = self.con.execute("SELECT json FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return Source.model_validate_json(row["json"]) if row else None

    def sources(self, source_types: Iterable[str] | None = None) -> list[Source]:
        """Return indexed sources, optionally of the given types only."""
        rows = self.con.execute("SELECT json, source_type FROM documents ORDER BY id").fetchall()
        wanted = set(source_types) if source_types else None
        return [
            Source.model_validate_json(r["json"])
            for r in rows
            if wanted is None or r["source_type"] in wanted
        ]

    def passages(self, doc_id: str) -> list[sqlite3.Row]:
        """Return all passages of a document in reading order."""
        return self.con.execute(
            "SELECT * FROM passages WHERE doc_id = ? ORDER BY ord", (doc_id,)
        ).fetchall()

    def passage(self, evidence_id: str) -> sqlite3.Row | None:
        """Return one passage by evidence ID."""
        return self.con.execute(
            "SELECT * FROM passages WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()

    def search_docs(self, text: str, doc_ids: Iterable[str] | None = None) -> list[DocHit]:
        """Rank documents by their best-matching passage (BM25, higher is better)."""
        query = fts_query(text)
        if not query:
            return []
        rows = self.con.execute(
            "SELECT doc_id, evidence_id, bm25(passages_fts, 0.0, 0.0, 1.0, 0.5) AS score "
            "FROM passages_fts WHERE passages_fts MATCH ? ORDER BY score",
            (query,),
        ).fetchall()
        allowed = set(doc_ids) if doc_ids is not None else None
        best: dict[str, DocHit] = {}
        for r in rows:
            if allowed is not None and r["doc_id"] not in allowed:
                continue
            if r["doc_id"] not in best:
                best[r["doc_id"]] = DocHit(r["doc_id"], -r["score"], r["evidence_id"])
        return sorted(best.values(), key=lambda h: h.bm25, reverse=True)
