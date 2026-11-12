"""Typed access to the source manifest."""

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["steward_decision", "sporting_regulation", "isc_appendix", "driving_guideline"]
FactValue = str

FACT_FIELDS: tuple[str, ...] = (
    "overtake_side",
    "overtaker_at_apex",
    "room_at_exit",
    "advantage_retained",
    "penalized_role",
)


class Event(BaseModel):
    """One race weekend, with its aliases and date window."""

    name: str
    short_name: str
    season: int
    start_date: date
    race_date: date
    circuit: str
    aliases: list[str]
    listing_url: str


class Driver(BaseModel):
    """One driver, keyed by car number."""

    name: str
    aliases: list[str]


class PassageAnchor(BaseModel):
    """Text markers that cut one passage out of a regulation or guideline page."""

    key: str
    label: str
    page: int
    start: str
    end: str


class ScopeVerification(BaseModel):
    """How well the edition in force on the event date is verified."""

    level: Literal["verified", "unverified"]
    note: str


class Release(BaseModel):
    """Publication record of a document on the FIA listing."""

    date: date
    url: str


class Source(BaseModel):
    """One document: decision, regulation, or guideline, with scope and version data."""

    id: str
    source_type: SourceType
    title: str
    status: Literal["current", "recalled", "superseded"]
    canonical_url: str | None
    cache_file: str | None
    sha256: str | None
    page_count: int
    listing_url: str | None = None
    note: str | None = None
    superseded_by: str | None = None
    supersedes: list[str] = Field(default_factory=list)
    supersession_basis: str | None = None
    text_available: bool = True
    # Decision fields
    event_key: str | None = None
    doc_number: int | None = None
    document_date: date | None = None
    document_time_local: str | None = None
    listing_published: str | None = None
    session: str | None = None
    turns: list[int] = Field(default_factory=list)
    penalized_car: int | None = None
    cars_involved: list[int] = Field(default_factory=list)
    offence: str | None = None
    outcome: str | None = None
    penalty_seconds: int | None = None
    facts: dict[str, FactValue] = Field(default_factory=dict)
    fact_quotes: dict[str, str] = Field(default_factory=dict)
    # Authority fields
    season: int | None = None
    published_on: date | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    scope_verification: ScopeVerification | None = None
    first_public_release: Release | None = None
    passages: list[PassageAnchor] = Field(default_factory=list)

    @property
    def is_decision(self) -> bool:
        """Return True for steward decisions."""
        return self.source_type == "steward_decision"


class UnavailableSource(BaseModel):
    """A source that applied but has no public text, with the reason."""

    id: str
    source_type: SourceType
    season: int
    title: str
    reason: str
    evidence_url: str


class Manifest(BaseModel):
    """All events, drivers, and sources of the corpus."""

    manifest_version: int
    retrieved_on: date
    events: dict[str, Event]
    drivers: dict[str, Driver]
    sources: list[Source]
    unavailable_sources: list[UnavailableSource]

    def source(self, source_id: str) -> Source:
        """Return a source by ID.

        Raises:
            KeyError: If the ID is not in the manifest.
        """
        for src in self.sources:
            if src.id == source_id:
                return src
        raise KeyError(source_id)

    def driver_name(self, car: int | None) -> str | None:
        """Return the full driver name for a car number."""
        if car is None:
            return None
        drv = self.drivers.get(str(car))
        return drv.name if drv else f"Car {car}"


@lru_cache(maxsize=4)
def load_manifest(path: Path) -> Manifest:
    """Load and validate the manifest file."""
    return Manifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
