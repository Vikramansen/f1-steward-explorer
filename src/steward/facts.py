"""Rule-based parsing of incident facts and constraints from question text.

The parser sets a fact only when the text states it. Everything else stays
"unknown" and does not affect ranking.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from steward.manifest import FACT_FIELDS, Manifest

FACT_VALUES: dict[str, set[str]] = {
    "overtake_side": {"inside", "outside"},
    "overtaker_at_apex": {"ahead", "level", "behind"},
    "room_at_exit": {"given", "not_given"},
    "advantage_retained": {"yes", "no"},
    "penalized_role": {"overtaker", "defender", "none"},
}

NUMBER_WORDS = {"five": 5, "ten": 10, "twenty": 20, "5": 5, "10": 10, "20": 20}

APEX_BEHIND = re.compile(
    r"not (?:quite )?(?:fully )?(?:level|alongside)|never (?:fully )?alongside|behind|"
    r"had not (?:got|drawn) (?:level|alongside)"
)
APEX_AHEAD = re.compile(r"\bahead\b|\bin front\b|\bfront axle ahead\b")
APEX_LEVEL = re.compile(r"\blevel\b|\balongside\b")
ROOM_NOT_GIVEN = re.compile(
    r"forc\w* (?:\w+ )?off|push\w* (?:\w+ )?off|r[au]n (?:\w+ )?off (?:the )?track|squeez\w*|"
    r"no room|did(?: not|n't) (?:leave|give) (?:\w+ )?(?:any )?room"
)
ROOM_GIVEN = re.compile(r"\b(?:left|gave|leave|give)s? (?:\w+ )?(?:just )?(?:enough )?room\b")
ADV_YES = re.compile(
    r"kept (?:the )?(?:position|place|advantage|lasting advantage)|stayed ahead|"
    r"retain\w* (?:the )?(?:position|place)|gain\w* (?:a )?(?:lasting )?advantage|lasting advantage"
)
ADV_NO = re.compile(
    r"gave (?:the )?(?:position|place|it) back|hand\w* (?:the )?(?:position|place) back|"
    r"conced\w*|give (?:the )?(?:position|place) back"
)
ROLE_DEFENDER = re.compile(r"\bdefend(?:ing|er)\b[^.?!]*\b(?:get|got|penal\w*|punish\w*)")
ROLE_OVERTAKER = re.compile(
    r"\b(?:overtaking|attacking) (?:driver|car)\b[^.?!]*\b(?:get|got|penal\w*)"
)
CLAUSE_SPLIT = re.compile(r"\b(?:while|whereas|compared (?:to|with)|versus|vs\.?)\b|;")


class IncidentQuery(BaseModel):
    """Constraints and facts read from one incident description."""

    text: str
    event_keys: list[str] = Field(default_factory=list)
    season: int | None = None
    drivers: list[int] = Field(default_factory=list)
    turns: list[int] = Field(default_factory=list)
    session: str | None = None
    doc_numbers: list[int] = Field(default_factory=list)
    penalty_seconds: list[int] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    fact_sources: dict[str, str] = Field(default_factory=dict)

    def known_facts(self) -> dict[str, str]:
        """Return only the facts that are not unknown."""
        return {k: v for k, v in self.facts.items() if v != "unknown"}


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])")


def find_events(text: str, manifest: Manifest) -> list[str]:
    """Return the keys of events named in the text, in order of appearance."""
    low = text.lower()
    found: list[str] = []
    for key, event in manifest.events.items():
        if any(_alias_pattern(a).search(low) for a in event.aliases) and key not in found:
            found.append(key)
    return found


def find_drivers(text: str, manifest: Manifest) -> list[int]:
    """Return the car numbers of drivers named in the text."""
    low = text.lower()
    cars: list[int] = []
    for num, drv in manifest.drivers.items():
        names = [drv.name.lower(), *drv.aliases]
        if any(_alias_pattern(n).search(low) for n in names):
            cars.append(int(num))
    for m in re.finditer(r"(?<!doc )(?<!document )\bcar (\d{1,2})\b", low):
        if int(m.group(1)) not in cars:
            cars.append(int(m.group(1)))
    return cars


def _apex_fact(low: str) -> str | None:
    for m in re.finditer(r"apex", low):
        window = low[max(0, m.start() - 70) : m.end() + 15]
        if APEX_BEHIND.search(window):
            return "behind"
        if APEX_AHEAD.search(window):
            return "ahead"
        if APEX_LEVEL.search(window):
            return "level"
    return None


def parse_incident(
    text: str, manifest: Manifest, default_season: int | None = None
) -> IncidentQuery:
    """Parse one incident description into constraints and explicitly stated facts."""
    low = text.lower()
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", low)]
    q = IncidentQuery(text=text, season=years[0] if years else default_season)
    q.event_keys = find_events(text, manifest)
    q.drivers = find_drivers(text, manifest)
    q.turns = [int(t) for t in re.findall(r"\b(?:turn|t)\s?(\d{1,2})\b", low)]
    q.doc_numbers = [
        int(d) for d in re.findall(r"\bdoc(?:ument)?\.?\s*(?:no\.?\s*)?(\d{1,3})\b", low)
    ]
    if re.search(r"\bsprint\b", low):
        q.session = "Sprint"
    q.penalty_seconds = sorted(
        {
            NUMBER_WORDS[m.group(1)]
            for m in re.finditer(
                r"\b(five|ten|twenty|5|10|20)[- ]?(?:s\b|sec\b|secs\b|seconds?\b)", low
            )
        }
    )
    facts: dict[str, str] = {}
    has_out, has_in = bool(re.search(r"\boutside\b", low)), bool(re.search(r"\binside\b", low))
    if has_out != has_in:
        facts["overtake_side"] = "outside" if has_out else "inside"
    if apex := _apex_fact(low):
        facts["overtaker_at_apex"] = apex
    if ROOM_NOT_GIVEN.search(low):
        facts["room_at_exit"] = "not_given"
    elif ROOM_GIVEN.search(low):
        facts["room_at_exit"] = "given"
    if ADV_NO.search(low):
        facts["advantage_retained"] = "no"
    elif ADV_YES.search(low):
        facts["advantage_retained"] = "yes"
    if ROLE_DEFENDER.search(low):
        facts["penalized_role"] = "defender"
    elif ROLE_OVERTAKER.search(low):
        facts["penalized_role"] = "overtaker"
    q.facts = {f: facts.get(f, "unknown") for f in FACT_FIELDS}
    q.fact_sources = {f: ("query_text" if f in facts else "unknown") for f in FACT_FIELDS}
    return q


def split_incidents(text: str, manifest: Manifest) -> list[str]:
    """Split a comparison question into one clause per incident.

    Splits only when the question names two or more events.
    """
    if len(find_events(text, manifest)) < 2:
        return [text]
    parts = [p.strip(" ,.?") for p in CLAUSE_SPLIT.split(text) if p.strip(" ,.?")]
    return parts if len(parts) > 1 else [text]


def ground_to_question(q: IncidentQuery, question: str, manifest: Manifest) -> list[str]:
    """Drop constraints and facts that the user's question does not state.

    Only the question clauses about the same event are used, so a fact from one
    incident cannot leak into the search for another.

    Returns:
        Warnings that name each dropped value.
    """
    years = re.findall(r"\b(20\d{2})\b", question)
    clauses = [
        parse_incident(c, manifest, int(years[0]) if years else None)
        for c in split_incidents(question, manifest)
    ]
    warnings: list[str] = []
    stated_events = {e for c in clauses for e in c.event_keys}
    if dropped := [e for e in q.event_keys if e not in stated_events]:
        warnings.append(f"dropped event {dropped}: not named in the question")
        q.event_keys = [e for e in q.event_keys if e in stated_events]
    relevant = [c for c in clauses if set(c.event_keys) & set(q.event_keys)] or clauses
    for f in FACT_FIELDS:
        val = q.facts.get(f, "unknown")
        if val != "unknown" and not any(c.facts.get(f) == val for c in relevant):
            warnings.append(f"dropped {f}={val}: not stated in the question")
            q.facts[f], q.fact_sources[f] = "unknown", "unknown"
    stated_turns = {t for c in relevant for t in c.turns}
    if dropped := [t for t in q.turns if t not in stated_turns]:
        warnings.append(f"dropped turn {dropped}: not stated in the question")
        q.turns = [t for t in q.turns if t in stated_turns]
    stated_drivers = {d for c in relevant for d in c.drivers}
    if dropped := [d for d in q.drivers if d not in stated_drivers]:
        warnings.append(
            f"dropped drivers {[manifest.driver_name(d) for d in dropped]}: "
            "not named in the question"
        )
        q.drivers = [d for d in q.drivers if d in stated_drivers]
    return warnings


def apply_model_facts(
    q: IncidentQuery, model_facts: dict[str, Any], manifest: Manifest
) -> list[str]:
    """Merge facts supplied by the model into a parsed query.

    Values outside the allowed vocabulary are ignored and reported.

    Returns:
        Warnings about ignored values.
    """
    warnings: list[str] = []
    for key, value in (model_facts or {}).items():
        if key in FACT_VALUES:
            val = str(value).strip().lower().replace(" ", "_")
            if val in ("", "unknown", "none_known", "null"):
                continue
            if val not in FACT_VALUES[key]:
                warnings.append(f"ignored {key}={value!r}: allowed {sorted(FACT_VALUES[key])}")
                continue
            q.facts[key] = val
            q.fact_sources[key] = "model"
        elif key == "event" and value:
            events = find_events(str(value), manifest)
            if events:
                q.event_keys = events
            else:
                warnings.append(f"ignored event={value!r}: not a known event")
        elif key == "season" and value:
            try:
                q.season = int(value)
            except (TypeError, ValueError):
                warnings.append(f"ignored season={value!r}")
        elif key == "turn" and value not in (None, "", "unknown"):
            try:
                q.turns = [int(str(value).lower().removeprefix("turn").strip())]
            except ValueError:
                warnings.append(f"ignored turn={value!r}")
        elif key == "drivers" and value:
            names = value if isinstance(value, list) else [value]
            cars = find_drivers(" ".join(map(str, names)), manifest)
            if cars:
                q.drivers = cars
        elif key == "session" and str(value).lower() == "sprint":
            q.session = "Sprint"
    return warnings
