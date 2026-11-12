"""Turn cached PDFs into short, page-referenced passages."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from steward.manifest import Source

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[”\"]?\s+(?=[A-Z“\"])")

DECISION_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("driver", r"No / Driver ", r" Competitor "),
    ("fact", r" Fact ", r" Infringement "),
    ("infringement", r" Infringement ", r" Decision "),
    ("decision", r" Decision ", r" Reason "),
    ("reason", r" Reason ", r" Competitors are reminded"),
)


@dataclass(frozen=True)
class Passage:
    """One indexed text unit with a stable evidence ID and page number."""

    evidence_id: str
    doc_id: str
    page: int | None
    kind: str
    label: str
    text: str
    quotable: bool = True


def normalize(text: str) -> str:
    """Normalize PDF text: NFKC form and single spaces."""
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def read_pages(path: Path) -> list[str]:
    """Return normalized text for each page of a PDF."""
    return [normalize(page.extract_text() or "") for page in PdfReader(path).pages]


def _page_for_offset(offsets: list[int], pos: int) -> int:
    page = 1
    for i, start in enumerate(offsets):
        if pos >= start:
            page = i + 1
    return page


def extract_decision(src: Source, pages: list[str]) -> list[Passage]:
    """Split a steward decision into field and reason-sentence passages.

    Raises:
        ValueError: If an expected field is missing from the document text.
    """
    offsets: list[int] = []
    full = ""
    for text in pages:
        offsets.append(len(full))
        full += text + " "
    passages: list[Passage] = []
    labels = {
        "driver": "Driver",
        "fact": "Fact",
        "infringement": "Infringement",
        "decision": "Decision",
        "reason": "Reason",
    }
    for name, start_pat, end_pat in DECISION_FIELDS:
        m = re.search(start_pat + r"(.+?)" + end_pat, full)
        if not m:
            raise ValueError(f"{src.id}: field {name!r} not found")
        body, pos = m.group(1).strip(), m.start(1)
        if name != "reason":
            page = _page_for_offset(offsets, pos)
            passages.append(
                Passage(f"{src.id}:p{page}:{name}", src.id, page, name, labels[name], body)
            )
            continue
        cursor = pos
        for n, sentence in enumerate(SENTENCE_SPLIT.split(body), start=1):
            sentence = sentence.strip()
            spos = full.find(sentence, cursor)
            cursor = max(cursor, spos)
            page = _page_for_offset(offsets, spos)
            passages.append(
                Passage(
                    f"{src.id}:p{page}:reason-{n}",
                    src.id,
                    page,
                    "reason",
                    f"Reason, sentence {n}",
                    sentence,
                )
            )
    return passages


def extract_anchored(src: Source, pages: list[str]) -> list[Passage]:
    """Cut regulation or guideline passages between manifest start/end markers.

    Raises:
        ValueError: If a marker is not found on its page.
    """
    passages: list[Passage] = []
    for anchor in src.passages:
        text = pages[anchor.page - 1]
        start = text.find(normalize(anchor.start))
        end_marker = normalize(anchor.end)
        end = text.find(end_marker, start + 1) if start >= 0 else -1
        if start < 0 or end < 0:
            raise ValueError(f"{src.id}:{anchor.key}: markers not found on page {anchor.page}")
        body = text[start : end + len(end_marker)]
        passages.append(
            Passage(
                f"{src.id}:p{anchor.page}:{anchor.key}",
                src.id,
                anchor.page,
                "article",
                anchor.label,
                body,
            )
        )
    return passages


def extract_source(src: Source, cache_dir: Path) -> tuple[list[Passage], str]:
    """Extract passages and the full normalized text for one source.

    Recalled documents without published text get one listing passage that is
    searchable but not quotable.
    """
    if not src.text_available or not src.cache_file:
        listing = Passage(
            f"{src.id}:listing",
            src.id,
            None,
            "listing",
            "FIA listing title",
            src.title,
            quotable=False,
        )
        return [listing], src.title
    pages = read_pages(cache_dir / src.cache_file)
    if src.is_decision:
        return extract_decision(src, pages), " ".join(pages)
    return extract_anchored(src, pages), " ".join(pages[a.page - 1] for a in src.passages)
