"""Offline corpus checks: cache integrity, fact quotes, and source links."""

from pathlib import Path

from steward.extract import extract_source, normalize
from steward.fetch import sha256_file
from steward.manifest import Manifest


def verify_corpus(manifest: Manifest, cache_dir: Path) -> list[str]:
    """Check cached files and manifest annotations against the document text.

    Returns:
        Problems found. An empty list means every check passed.
    """
    problems: list[str] = []
    for src in manifest.sources:
        if src.status == "recalled":
            if src.canonical_url or src.cache_file:
                problems.append(f"{src.id}: recalled doc must not have a URL or cache file")
            if not src.superseded_by:
                problems.append(f"{src.id}: recalled doc needs superseded_by")
            continue
        path = cache_dir / (src.cache_file or "")
        if not path.is_file():
            problems.append(f"{src.id}: cache file missing ({path.name})")
            continue
        if sha256_file(path) != src.sha256:
            problems.append(f"{src.id}: checksum mismatch")
            continue
        if not (src.canonical_url or "").startswith("https://www.fia.com/"):
            problems.append(f"{src.id}: canonical URL is not on fia.com")
        passages, full = extract_source(src, cache_dir)
        if not passages:
            problems.append(f"{src.id}: no passages extracted")
        for p in passages:
            if p.page is None or not 1 <= p.page <= src.page_count:
                problems.append(f"{p.evidence_id}: page {p.page} out of range")
        for fact, quote in src.fact_quotes.items():
            if normalize(quote) not in full:
                problems.append(f"{src.id}: quote for {fact} not in document: {quote!r}")
        for fact, value in src.facts.items():
            if value != "unknown" and fact not in src.fact_quotes:
                problems.append(f"{src.id}: fact {fact}={value} has no supporting quote")
    return problems
