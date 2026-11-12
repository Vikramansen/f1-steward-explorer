"""Download official FIA documents into the local cache and verify them."""

import hashlib
import html
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

from steward.manifest import Manifest

USER_AGENT = "Mozilla/5.0 (f1-steward-explorer; research cache)"


@dataclass
class FetchResult:
    """Outcome of downloading one source."""

    source_id: str
    path: Path
    status: str


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_sources(manifest: Manifest, cache_dir: Path, force: bool = False) -> list[FetchResult]:
    """Download every source that has a canonical URL and check its SHA-256.

    Args:
        manifest: Parsed manifest.
        cache_dir: Target directory for PDFs.
        force: Download again even if a valid cached copy exists.

    Returns:
        One result per downloadable source.

    Raises:
        ValueError: If a downloaded file does not match the pinned checksum.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
    ) as client:
        for src in manifest.sources:
            if not (src.canonical_url and src.cache_file and src.sha256):
                continue
            path = cache_dir / src.cache_file
            if path.exists() and not force and sha256_file(path) == src.sha256:
                results.append(FetchResult(src.id, path, "cached"))
                continue
            logger.info("Downloading {} from {}", src.id, src.canonical_url)
            resp = client.get(src.canonical_url)
            resp.raise_for_status()
            digest = hashlib.sha256(resp.content).hexdigest()
            if digest != src.sha256:
                raise ValueError(
                    f"{src.id}: checksum mismatch (got {digest}, manifest {src.sha256}). "
                    "The FIA may have replaced the file; check it before updating the manifest."
                )
            path.write_bytes(resp.content)
            results.append(FetchResult(src.id, path, "downloaded"))
    return results


def verify_listings(manifest: Manifest) -> list[str]:
    """Check event listings online: titles, recall marks, and missing download links.

    Returns:
        Human-readable problems. An empty list means all checks passed.
    """
    problems: list[str] = []
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
    ) as client:
        for key, event in manifest.events.items():
            page = html.unescape(client.get(event.listing_url).text)
            rows = re.split(r'<li class="document-row', page)[1:]
            titles: dict[str, bool] = {}
            for row in rows:
                title = re.search(
                    r'field-name-title-field.*?field-item even">(.*?)</div>', row, re.S
                )
                if title:
                    has_link = "/sites/default/files/decision-document/" in row
                    titles[title.group(1).strip()] = has_link
            for src in manifest.sources:
                if src.event_key != key:
                    continue
                if src.title not in titles:
                    problems.append(f"{src.id}: title not found on listing: {src.title!r}")
                elif src.status == "recalled" and titles[src.title]:
                    problems.append(f"{src.id}: listing now has a download link for recalled doc")
                elif src.status != "recalled" and not titles[src.title]:
                    problems.append(f"{src.id}: listing has no download link")
    return problems
