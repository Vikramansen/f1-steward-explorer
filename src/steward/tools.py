"""The three server-side tools: resolve_authority, search_decisions, get_passages.

Scope checks (season, in-force dates, recall and supersession) run in code
before any ranking. The model cannot turn them off.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from steward.config import Settings
from steward.facts import (
    IncidentQuery,
    apply_model_facts,
    find_events,
    ground_to_question,
    parse_incident,
    split_incidents,
)
from steward.index import Index
from steward.manifest import FACT_FIELDS, Manifest, Source

AUTHORITY_TYPES = ("sporting_regulation", "isc_appendix", "driving_guideline")
TYPE_KEYWORDS: dict[str, re.Pattern[str]] = {
    "driving_guideline": re.compile(r"guideline|driving standard"),
    "sporting_regulation": re.compile(r"sporting regulation|\barticle\b|\bregulation"),
    "isc_appendix": re.compile(r"appendix l|sporting code|\bisc\b"),
}
REASON_KEYWORDS: dict[str, int] = {
    "apex": 3,
    "outside": 2,
    "inside": 2,
    "level": 2,
    "ahead": 2,
    "alongside": 2,
    "forc": 2,
    "advantage": 2,
    "penalty": 2,
    "instead of": 2,
    "guideline": 2,
    "right": 1,
    "room": 2,
    "exit": 1,
}


def page_url(src: Source, page: int | None) -> str | None:
    """Return the official URL with a #page=N fragment."""
    if not src.canonical_url:
        return src.listing_url
    return f"{src.canonical_url}#page={page}" if page else src.canonical_url


def _event_label(manifest: Manifest, key: str | None) -> str | None:
    return manifest.events[key].name if key and key in manifest.events else None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = str(value).strip()
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        pass
    if m := re.search(r"\b(20\d{2})\b", value):
        return date(int(m.group(1)), 12, 31)
    return None


def apex_spans(text: str) -> list[list[int]]:
    """Return [start, end] spans of clauses that describe position at the apex."""
    spans: list[list[int]] = []
    for m in re.finditer(r"\bapex\b", text, re.I):
        start = 0
        for sep in (". ", ", but ", "; "):
            i = text.rfind(sep, 0, m.start())
            if i >= 0:
                start = max(start, i + len(sep))
        stops = [
            i
            for i in (
                text.find(". ", m.end()),
                text.find(" when ", m.end()),
                text.find(", however", m.end()),
            )
            if i >= 0
        ]
        end = min(stops) if stops else len(text.rstrip(". "))
        if not spans or start > spans[-1][1]:
            spans.append([start, end])
    return spans


def _merge_outputs(outputs: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Interleave per-clause search outputs so each incident keeps its best hits."""
    merged: list[dict[str, Any]] = []
    for rank in range(top_k):
        for out in outputs:
            if rank < len(out["results"]) and out["results"][rank]["id"] not in {
                r["id"] for r in merged
            }:
                merged.append(out["results"][rank])

    def union(key: str) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for out in outputs:
            for item in out[key]:
                seen.setdefault(item["id"], item)
        return list(seen.values())

    statuses = [o["status"] for o in outputs]
    return {
        "status": "ok"
        if all(s == "ok" for s in statuses)
        else ("partial" if "ok" in statuses else "insufficient_evidence"),
        "split_into_incidents": len(outputs),
        "parsed_query": [o["parsed_query"] for o in outputs],
        "status_by_incident": statuses,
        "warnings": [w for o in outputs for w in o["warnings"]],
        "results": merged[:top_k],
        "ambiguity": " ".join(o["ambiguity"] for o in outputs if o["ambiguity"]) or None,
        "excluded_before_ranking": union("excluded_before_ranking"),
        "near_misses": sorted(
            union("near_misses"), key=lambda r: (r["hard_failures"], -r["text_score"])
        )[:3],
        "comparisons": [
            c for c in union("comparisons") if c["id"] not in {r["id"] for r in merged}
        ],
    }


@dataclass
class RunContext:
    """Per-run state used to check claims after the model answers."""

    question: str | None = None
    seen_evidence: dict[str, str] = field(default_factory=dict)
    seen_text: dict[str, str] = field(default_factory=dict)
    doc_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    searched_events: set[str] = field(default_factory=set)
    rejected_sources: dict[str, str] = field(default_factory=dict)
    target_decisions: set[str] = field(default_factory=set)
    comparison_decisions: set[str] = field(default_factory=set)
    applicable_authority: set[str] = field(default_factory=set)
    authority_checked: bool = False


class Toolbox:
    """Tool implementations bound to one index and manifest."""

    def __init__(self, index: Index, manifest: Manifest, settings: Settings) -> None:
        """Bind the tools to one index, manifest, and settings."""
        self.index = index
        self.manifest = manifest
        self.settings = settings

    # ---- resolve_authority -------------------------------------------------

    def resolve_authority(
        self,
        event: str | None = None,
        date: str | None = None,
        issue: str = "",
        ctx: RunContext | None = None,
    ) -> dict[str, Any]:
        """Find rule and guideline candidates and check scope against the event date."""
        events = find_events(event or "", self.manifest) or find_events(issue, self.manifest)
        ev = self.manifest.events[events[0]] if events else None
        op_date = _parse_date(date) or (ev.race_date if ev else None)
        if op_date is None:
            return {"error": "Need a known event or a date to check which rules were in force."}
        warnings: list[str] = []
        if ev and not ev.start_date <= op_date <= ev.race_date:
            warnings.append(
                f"date {op_date} is outside {ev.name} ({ev.start_date} to "
                f"{ev.race_date}); used the race date {ev.race_date}"
            )
            op_date = ev.race_date
        season = ev.season if ev else op_date.year
        requested = [t for t, pat in TYPE_KEYWORDS.items() if pat.search(issue.lower())]

        candidates = self.index.sources(AUTHORITY_TYPES)
        relevance = {
            h.doc_id: h.bm25 for h in self.index.search_docs(issue, [c.id for c in candidates])
        }
        applicable: list[dict[str, Any]] = []
        unverified: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for src in candidates:
            checks, failure = self._scope_checks(src, season, op_date)
            entry = {
                "id": src.id,
                "title": src.title,
                "source_type": src.source_type,
                "relevance": round(relevance.get(src.id, 0.0), 3),
                "checks": checks,
                "url": src.canonical_url,
            }
            if failure:
                entry["failure_mode"], entry["reason"] = failure
                rejected.append(entry)
                if ctx is not None:
                    ctx.rejected_sources[src.id] = failure[1]
            else:
                verified = src.scope_verification and src.scope_verification.level == "verified"
                late = src.published_on is not None and src.published_on > op_date
                if src.scope_verification:
                    entry["scope_note"] = src.scope_verification.note
                if verified and not late:
                    entry["verdict"] = "applicable"
                    applicable.append(entry)
                else:
                    # Not citable: the edition in force on the date is not established.
                    entry["verdict"] = "unverified"
                    unverified.append(entry)
                    if ctx is not None:
                        ctx.rejected_sources[src.id] = "edition in force on the date not verified"
        applicable.sort(key=lambda e: e["relevance"], reverse=True)
        unverified.sort(key=lambda e: e["relevance"], reverse=True)
        rejected.sort(key=lambda e: e["relevance"], reverse=True)
        if ctx is not None:
            ctx.authority_checked = True
            ctx.applicable_authority.update(a["id"] for a in applicable)

        unavailable = [
            self._unavailable_entry(u, events[0] if events else None, op_date)
            for u in self.manifest.unavailable_sources
            if u.season == season
        ]
        status: dict[str, str] = {}
        for t in requested or list(AUTHORITY_TYPES):
            if any(a["source_type"] == t for a in applicable):
                status[t] = "ok"
            elif any(a["source_type"] == t for a in unverified):
                status[t] = "unverified"
            else:
                status[t] = "insufficient_evidence"
        return {
            "event": ev.name if ev else None,
            "operative_date": op_date.isoformat(),
            "warnings": warnings,
            "season": season,
            "requested_types": requested,
            "status_by_type": status,
            "applicable": applicable,
            "unverified": unverified,
            "rejected": rejected,
            "unavailable": unavailable,
        }

    def _scope_checks(
        self, src: Source, season: int, op_date: date
    ) -> tuple[list[dict[str, Any]], tuple[str, str] | None]:
        checks: list[dict[str, Any]] = []
        failure: tuple[str, str] | None = None
        if src.status == "recalled":
            failure = ("recalled", "Recalled by the FIA; not usable as authority.")
        season_ok = src.season == season
        checks.append(
            {
                "check": "season",
                "passed": season_ok,
                "detail": f"source season {src.season}, event season {season}",
            }
        )
        in_force = (src.valid_from is None or src.valid_from <= op_date) and (
            src.valid_to is None or op_date <= src.valid_to
        )
        checks.append(
            {
                "check": "in_force_on_date",
                "passed": in_force,
                "detail": f"valid {src.valid_from} to {src.valid_to or 'open'}; "
                f"operative date {op_date}",
            }
        )
        if not in_force and failure is None:
            failure = (
                "not_in_force_on_date",
                f"{src.season} season source, in force from {src.valid_from}; "
                f"not in force on {op_date}.",
            )
        if not season_ok and failure is None:
            failure = (
                "wrong_season",
                f"Written for the {src.season} season; the incident is in {season}.",
            )
        if src.published_on:
            checks.append(
                {
                    "check": "published_before_date",
                    "passed": src.published_on <= op_date,
                    "detail": f"published {src.published_on}",
                }
            )
        if failure and src.published_on and src.published_on > op_date:
            failure = (failure[0], failure[1] + f" Published {src.published_on}, after the event.")
        return checks, failure

    def _unavailable_entry(self, u: Any, event_key: str | None, op_date: date) -> dict[str, Any]:
        cited: list[dict[str, Any]] = []
        for src in self.index.sources(["steward_decision"]):
            if src.status != "current" or (event_key and src.event_key != event_key):
                continue
            if src.document_date and src.document_date > op_date:
                continue
            for row in self.index.passages(src.id):
                if re.search(r"guideline", row["text"], re.I) and row["quotable"]:
                    cited.append(
                        {
                            "evidence_id": row["evidence_id"],
                            "doc_id": src.id,
                            "page": row["page"],
                            "url": page_url(src, row["page"]),
                        }
                    )
        return {
            "id": u.id,
            "title": u.title,
            "source_type": u.source_type,
            "reason": u.reason,
            "evidence_url": u.evidence_url,
            "what_rulings_establish": cited[: self.settings.max_guideline_citations],
        }

    # ---- search_decisions --------------------------------------------------

    def search_decisions(
        self,
        query: str,
        incident_facts: dict[str, Any] | None = None,
        ctx: RunContext | None = None,
        use_facts: bool = True,
    ) -> dict[str, Any]:
        """Retrieve rulings by text and incident facts, after scope filtering.

        A question that names two events is split into one search per clause.
        """
        clauses = split_incidents(query, self.manifest)
        years = re.findall(r"\b(20\d{2})\b", query)
        outputs = []
        for clause in clauses:
            q = parse_incident(clause, self.manifest, int(years[0]) if years else None)
            facts = incident_facts if len(clauses) == 1 else {}
            warnings = apply_model_facts(q, facts or {}, self.manifest)
            if ctx is not None and ctx.question:
                warnings += ground_to_question(q, ctx.question, self.manifest)
            if q.season is None and q.event_keys:
                q.season = self.manifest.events[q.event_keys[0]].season
            outputs.append(self._rank(q, warnings, ctx, use_facts=use_facts))
        return outputs[0] if len(outputs) == 1 else _merge_outputs(outputs, self.settings.top_k)

    def _rank(
        self, q: IncidentQuery, warnings: list[str], ctx: RunContext | None, use_facts: bool = True
    ) -> dict[str, Any]:
        w = self.settings.weights
        decisions = self.index.sources(["steward_decision"])
        excluded: list[dict[str, Any]] = []
        pool: list[Source] = []
        for src in decisions:
            if src.status != "current":
                excluded.append(
                    {
                        "id": src.id,
                        "title": src.title,
                        "failure_mode": "superseded_version",
                        "reason": f"Status '{src.status}'. Replaced by {src.superseded_by}. "
                        f"{src.supersession_basis or ''}".strip(),
                        "superseded_by": src.superseded_by,
                        "url": src.listing_url,
                    }
                )
                if ctx is not None:
                    ctx.rejected_sources[src.id] = f"{src.status}; use {src.superseded_by}"
                continue
            pool.append(src)

        target_events = set(q.event_keys)
        target_dates = [self.manifest.events[k].race_date for k in target_events]
        text_hits = {h.doc_id: h.bm25 for h in self.index.search_docs(q.text, [s.id for s in pool])}
        max_bm25 = max(text_hits.values(), default=0.0) or 1.0

        results: list[dict[str, Any]] = []
        near_misses: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        for src in pool:
            ev = self.manifest.events.get(src.event_key or "")
            hard: list[str] = []
            if q.season is not None and ev and ev.season != q.season:
                hard.append(f"season {ev.season}, query asks about {q.season}")
            other_event = bool(target_events) and src.event_key not in target_events
            if q.drivers and not set(q.drivers) <= set(src.cars_involved):
                missing = [
                    self.manifest.driver_name(c) for c in q.drivers if c not in src.cars_involved
                ]
                hard.append(f"does not involve {', '.join(map(str, missing))}")
            if q.turns and not set(q.turns) & set(src.turns):
                hard.append(
                    f"Turn {'/'.join(map(str, src.turns))}, query asks about Turn "
                    f"{'/'.join(map(str, q.turns))}"
                )
            if q.session and src.session != q.session:
                hard.append(f"{src.session} session, query asks about {q.session}")

            text_score = text_hits.get(src.id, 0.0) / max_bm25
            matched, mismatched = [], []
            if use_facts:
                for f in FACT_FIELDS:
                    qv, dv = q.facts.get(f, "unknown"), src.facts.get(f, "unknown")
                    if qv == "unknown" or dv == "unknown":
                        continue
                    (matched if qv == dv else mismatched).append(f"{f}={dv}")
                if q.penalty_seconds:
                    if src.penalty_seconds in q.penalty_seconds:
                        matched.append(f"penalty={src.penalty_seconds}s")
                    else:
                        mismatched.append(f"penalty={src.penalty_seconds or 'none'}")
                if q.doc_numbers and src.doc_number in q.doc_numbers:
                    matched.append(f"doc_number={src.doc_number}")
                if q.doc_numbers and set(src.supersedes) & {
                    e["id"]
                    for e in excluded
                    if self.index.source(e["id"]).doc_number in q.doc_numbers
                }:
                    matched.append("supersedes_requested_doc")
            score = (
                w.text * text_score
                + w.fact_match * len(matched)
                - w.fact_mismatch * len(mismatched)
            )
            row = {
                "id": src.id,
                "event": _event_label(self.manifest, src.event_key),
                "document_date": src.document_date.isoformat() if src.document_date else None,
                "doc_number": src.doc_number,
                "session": src.session,
                "turns": src.turns,
                "penalized": self.manifest.driver_name(src.penalized_car) or "no penalty",
                "outcome": src.outcome,
                "score": round(score, 3),
                "text_score": round(text_score, 3),
                "matched_facts": matched,
                "mismatched_facts": mismatched,
            }
            if other_event and not hard:
                later = any(src.document_date and src.document_date > d for d in target_dates)
                row["role"] = "comparison"
                row["note"] = "Different event; shown for comparison only" + (
                    ", dated after the target event." if later else "."
                )
                if matched:
                    comparisons.append(row)
            elif hard or other_event:
                row["failure_mode"] = "different_incident" if not other_event else "other_event"
                row["reason"] = "; ".join(hard) or "different event"
                row["hard_failures"] = len(hard)
                if not other_event and text_score > 0:
                    near_misses.append(row)
            elif score > 0:
                row["role"] = "target"
                results.append(row)

        results.sort(key=lambda r: r["score"], reverse=True)
        near_misses.sort(key=lambda r: (r["hard_failures"], -r["text_score"]))
        comparisons.sort(key=lambda r: r["score"], reverse=True)
        top = results[: self.settings.top_k]
        if ctx is not None:
            ctx.target_decisions.update(r["id"] for r in top)
            ctx.comparison_decisions.update(r["id"] for r in comparisons[:2])
            ctx.searched_events.update(q.event_keys)

        ambiguity = None
        if not q.turns and len(top) > 1 and q.drivers:
            same = [
                r
                for r in top
                if self.index.source(r["id"])
                and set(q.drivers) <= set(self.index.source(r["id"]).cars_involved)
            ]
            turns = {tuple(r["turns"]) for r in same}
            if len(same) > 1 and len(turns) > 1:
                ambiguity = (
                    "Several rulings at this event involve the same drivers: "
                    + ", ".join(f"Turn {'/'.join(map(str, r['turns']))} ({r['id']})" for r in same)
                    + ". Name the turn to choose one."
                )
        return {
            "status": "ok" if top else "insufficient_evidence",
            "parsed_query": {
                "events": [_event_label(self.manifest, k) for k in q.event_keys],
                "season": q.season,
                "drivers": [self.manifest.driver_name(c) for c in q.drivers],
                "turns": q.turns,
                "session": q.session,
                "doc_numbers": q.doc_numbers,
                "penalty_seconds": q.penalty_seconds,
                "facts": {
                    f: {"value": q.facts[f], "source": q.fact_sources[f]} for f in FACT_FIELDS
                },
            },
            "warnings": warnings,
            "results": top,
            "ambiguity": ambiguity,
            "excluded_before_ranking": excluded,
            "near_misses": near_misses[:3],
            "comparisons": comparisons[:2],
        }

    # ---- get_passages ------------------------------------------------------

    def get_passages(
        self, document_ids: list[str], ctx: RunContext | None = None
    ) -> dict[str, Any]:
        """Return short excerpts with page references, URLs, and evidence IDs."""
        if isinstance(document_ids, str):
            document_ids = [document_ids]
        doc_ids = [str(d).split(":")[0].strip() for d in document_ids]
        out: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for doc_id in list(dict.fromkeys(doc_ids))[: self.settings.max_docs_per_passage_call]:
            src = self.index.source(doc_id)
            if src is None:
                errors.append({"id": doc_id, "error": "not in index"})
                continue
            if src.status == "recalled" or not src.text_available:
                errors.append(
                    {
                        "id": doc_id,
                        "error": f"recalled; no published text. Use {src.superseded_by}.",
                    }
                )
                continue
            rows = [r for r in self.index.passages(doc_id) if r["quotable"]]
            if src.is_decision:
                rows = self._pick_decision_rows(rows)
            if ctx is not None and doc_id not in (
                ctx.target_decisions | ctx.comparison_decisions | ctx.applicable_authority
            ):
                why = ctx.rejected_sources.get(doc_id, "not returned as a result in this run")
                errors.append(
                    {
                        "id": doc_id,
                        "warning": f"Not usable as evidence: {why}. "
                        "Claims citing it will be removed.",
                    }
                )
            for r in rows:
                text = r["text"]
                if len(text) > self.settings.excerpt_chars:
                    text = text[: self.settings.excerpt_chars - 1].rstrip() + "…"
                item = {
                    "evidence_id": r["evidence_id"],
                    "doc_id": doc_id,
                    "label": r["label"],
                    "page": r["page"],
                    "text": text,
                    "source_url": page_url(src, r["page"]),
                }
                spans = apex_spans(text)
                if spans:
                    item["apex_spans"] = spans
                out.append(item)
                if ctx is not None:
                    ctx.seen_evidence[r["evidence_id"]] = doc_id
                    ctx.seen_text[r["evidence_id"]] = r["text"]
            if ctx is not None:
                ctx.doc_meta[doc_id] = {
                    "title": src.title,
                    "event_key": src.event_key,
                    "turns": src.turns,
                    "penalty_seconds": src.penalty_seconds,
                    "cars": src.cars_involved,
                }
        return {"passages": out, "errors": errors}

    def _pick_decision_rows(self, rows: list[Any]) -> list[Any]:
        fields = [r for r in rows if r["kind"] in ("driver", "fact", "infringement", "decision")]
        reasons = [r for r in rows if r["kind"] == "reason"]
        scored = []
        for r in reasons:
            low = r["text"].lower()
            scored.append((sum(v for k, v in REASON_KEYWORDS.items() if k in low), r["ord"], r))
        keep = sorted(sorted(scored, key=lambda s: -s[0])[:4], key=lambda s: s[1])
        return fields + [s[2] for s in keep if s[0] > 0]


@dataclass
class ToolCall:
    """Record of one tool execution, shown in the trace."""

    n: int
    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float
    status: str


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "resolve_authority",
            "description": (
                "Find sporting regulations, ISC Appendix L, and driving guidelines for an "
                "incident, and check each source's season and in-force dates against the event. "
                "Returns applicable and rejected sources with reasons."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event": {
                        "type": "string",
                        "description": "Event, e.g. '2024 United States Grand Prix'",
                    },
                    "date": {"type": "string", "description": "Incident date YYYY-MM-DD, if known"},
                    "issue": {
                        "type": "string",
                        "description": "Rule topic, e.g. 'leaving the track, outside overtake'",
                    },
                },
                "required": ["event", "issue"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_decisions",
            "description": (
                "Search FIA steward decisions for ONE incident. Uses text plus incident facts. "
                "Recalled or superseded documents are removed before ranking. Call once per "
                "incident."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The incident in plain words, with event, drivers, turn",
                    },
                    "incident_facts": {
                        "type": "object",
                        "description": "Only facts the user stated. Leave others out.",
                        "properties": {
                            "event": {"type": "string"},
                            "turn": {"type": "integer"},
                            "drivers": {"type": "array", "items": {"type": "string"}},
                            "overtake_side": {
                                "type": "string",
                                "enum": ["inside", "outside", "unknown"],
                            },
                            "overtaker_at_apex": {
                                "type": "string",
                                "enum": ["ahead", "level", "behind", "unknown"],
                            },
                            "room_at_exit": {
                                "type": "string",
                                "enum": ["given", "not_given", "unknown"],
                            },
                            "advantage_retained": {
                                "type": "string",
                                "enum": ["yes", "no", "unknown"],
                            },
                        },
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_passages",
            "description": (
                "Get short excerpts from cached official documents. Each excerpt has an "
                "evidence_id, page, and source URL. Cite evidence_ids in the answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs from search_decisions or resolve_authority",
                    },
                },
                "required": ["document_ids"],
            },
        },
    },
]


TOOL_PARAMS: dict[str, set[str]] = {
    spec["function"]["name"]: set(spec["function"]["parameters"]["properties"])
    for spec in TOOL_SPECS
}


class ToolRunner:
    """Executes tool calls with a hard per-run budget and records each call."""

    def __init__(self, toolbox: Toolbox, max_calls: int, question: str | None = None) -> None:
        """Start a run with an empty context and call log."""
        self.toolbox = toolbox
        self.max_calls = max_calls
        self.calls: list[ToolCall] = []
        self.ctx = RunContext(question=question)

    def execute(self, name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
        """Run one tool call. Calls past the budget are refused without running."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        n = len(self.calls) + 1
        if n > self.max_calls:
            output = {
                "error": f"Tool budget exhausted ({self.max_calls} calls per run). "
                "Answer now with the evidence you have."
            }
            self.calls.append(ToolCall(n, name, arguments, output, 0.0, "refused_budget"))
            return output
        start = time.perf_counter()
        try:
            fn = {
                "resolve_authority": self.toolbox.resolve_authority,
                "search_decisions": self.toolbox.search_decisions,
                "get_passages": self.toolbox.get_passages,
            }[name]
            args = {k: v for k, v in arguments.items() if k in TOOL_PARAMS[name]}
            output = fn(**args, ctx=self.ctx)
            if dropped := sorted(set(arguments) - set(args)):
                output = {**output, "ignored_arguments": dropped}
            status = "error" if "error" in output else "ok"
        except KeyError:
            output, status = {"error": f"Unknown tool {name!r}"}, "error"
        except TypeError as exc:
            output, status = {"error": f"Bad arguments: {exc}"}, "error"
        latency = (time.perf_counter() - start) * 1000
        self.calls.append(ToolCall(n, name, arguments, output, round(latency, 2), status))
        return output

    @property
    def executed(self) -> int:
        """Number of calls that ran (refused calls excluded)."""
        return sum(1 for c in self.calls if c.status != "refused_budget")
