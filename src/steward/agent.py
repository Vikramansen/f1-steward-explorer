"""Model loop: one small model, three tools, evidence-checked claims."""

import json
import re
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import resources
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from steward.config import Settings
from steward.facts import find_drivers, find_events, split_incidents
from steward.index import Index
from steward.manifest import Manifest
from steward.tools import TOOL_SPECS, RunContext, ToolRunner

MAX_REPAIRS = 2

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


class Claim(BaseModel):
    """One answer claim, with its evidence IDs and check result."""

    text: str
    evidence: list[str] = Field(default_factory=list)
    supported: bool = False
    problems: list[str] = Field(default_factory=list)


class Incident(BaseModel):
    """One search_decisions result, shown as one column in the UI."""

    query: str
    status: str
    top_doc: str | None
    event: str | None
    ambiguity: str | None = None
    # Rulings the user must choose between. No claim is shown for this incident until then.
    choices: list[dict[str, Any]] = Field(default_factory=list)
    # Closest rejected rulings, shown when no ruling is eligible.
    rejected: list[dict[str, Any]] = Field(default_factory=list)


class RunRecord(BaseModel):
    """Full record of one run. Saved as a replay file and returned by the API."""

    run_id: str
    question: str
    mode: str
    model: str
    model_base_url: str
    started_at: str
    duration_s: float
    status: str
    claims: list[Claim]
    incidents: list[Incident]
    tool_calls: list[dict[str, Any]]
    transcript: list[dict[str, Any]]
    budget: dict[str, int]
    index_meta: dict[str, str]
    raw_final: str | None = None
    error: str | None = None


def system_prompt(settings: Settings) -> str:
    """Load the system prompt template and fill in run limits."""
    text = resources.files("steward").joinpath("prompts/system.txt").read_text(encoding="utf-8")
    return text.format(max_tool_calls=settings.max_tool_calls)


def parse_final(content: str) -> tuple[str, list[dict[str, Any]], str | None]:
    """Extract status and claims from the model's final JSON message.

    Returns:
        (status, raw claims, parse error or None).
    """
    text = THINK_BLOCK.sub("", content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return "insufficient_evidence", [], "final message had no JSON object"
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return "insufficient_evidence", [], f"final JSON did not parse: {exc}"
    claims = data.get("claims") or []
    if not isinstance(claims, list):
        return "insufficient_evidence", [], "claims is not a list"
    return str(data.get("status", "answered")), [c for c in claims if isinstance(c, dict)], None


PENALTY_MENTION = re.compile(r"\b(five|ten|5|10|20)[- ]?(?:s\b|sec|second)", re.I)
TURN_MENTION = re.compile(r"\bturn (\d{1,2})\b", re.I)
INSTRUMENTS: tuple[tuple[str, str], ...] = (
    (r"sporting regulations?", "the Sporting Regulations"),
    (r"appendix l\b", "Appendix L"),
    (r"driving standards? guidelines?", "the Driving Standards Guidelines"),
    (r"international sporting code", "the International Sporting Code"),
)


def consistency_problems(
    text: str, evidence: list[str], ctx: RunContext, manifest: Manifest
) -> list[str]:
    """Check that events, drivers, turns, and penalties in a claim match the cited documents."""
    docs = {ctx.seen_evidence[e] for e in evidence if e in ctx.seen_evidence}
    metas = [ctx.doc_meta[d] for d in docs if d in ctx.doc_meta]
    if not metas:
        return []
    cited_text = " ".join(ctx.seen_text.get(e, "") for e in evidence).lower()
    problems: list[str] = []
    decision_metas = [m for m in metas if m["event_key"]]
    if decision_metas:
        cited_events = {m["event_key"] for m in decision_metas}
        for ev in find_events(text, manifest):
            if ev not in cited_events:
                problems.append(f"names {manifest.events[ev].name} but cites no ruling from it")
        cited_cars = {c for m in decision_metas for c in m["cars"]}
        for car in find_drivers(text, manifest):
            if car not in cited_cars:
                problems.append(
                    f"names {manifest.driver_name(car)}, who is not in the cited rulings"
                )
        cited_turns = {t for m in decision_metas for t in m["turns"]}
        for t in {int(x) for x in TURN_MENTION.findall(text)}:
            if t not in cited_turns and f"turn {t}" not in cited_text:
                problems.append(f"names Turn {t}, which the cited rulings do not cover")
    cited_scope = cited_text + " " + " ".join(m.get("title", "") for m in metas).lower()
    low = text.lower()
    for pattern, name in INSTRUMENTS:
        if re.search(pattern, low) and not re.search(pattern, cited_scope):
            problems.append(f"names {name}, which the cited evidence does not mention")
    for ref in re.findall(r"\b(article \d+(?:\.\d+)*|issue \d+)\b", low):
        if ref not in cited_scope:
            problems.append(f"names {ref.title()}, which the cited evidence does not mention")
    numbers = {"five": 5, "ten": 10}
    cited_pen = {m["penalty_seconds"] for m in metas} | {
        int(n) for n in re.findall(r"\b(\d{1,2}) second", cited_text)
    }
    for p in PENALTY_MENTION.findall(text):
        val = numbers.get(p.lower(), int(p) if p.isdigit() else 0)
        if val not in cited_pen:
            problems.append(f"states a {val} second penalty that the cited evidence does not")
    return problems


def validate_claims(
    raw: list[dict[str, Any]], ctx: RunContext, manifest: Manifest | None = None
) -> list[Claim]:
    """Mark each claim supported only if every cited ID came from get_passages in this run.

    The cited document must also be a search_decisions result (target or comparison)
    or an applicable source from resolve_authority. Claims that cite rejected or
    excluded sources, or cite nothing, are unsupported.
    """
    claims: list[Claim] = []
    for item in raw:
        evidence = item.get("evidence") or item.get("evidence_ids") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        claim = Claim(text=str(item.get("text", "")).strip(), evidence=[str(e) for e in evidence])
        if not claim.evidence:
            claim.problems.append("no evidence ID")
        for ev in claim.evidence:
            doc = ctx.seen_evidence.get(ev)
            if doc is None:
                claim.problems.append(f"{ev}: not returned by get_passages in this run")
            elif doc in ctx.rejected_sources:
                claim.problems.append(f"{ev}: source rejected ({ctx.rejected_sources[doc]})")
            elif doc not in (
                ctx.target_decisions | ctx.comparison_decisions | ctx.applicable_authority
            ):
                claim.problems.append(
                    f"{ev}: {doc} was not a search_decisions result or applicable authority "
                    "in this run"
                )
        if manifest is not None and not claim.problems:
            claim.problems += consistency_problems(claim.text, claim.evidence, ctx, manifest)
        claim.supported = not claim.problems and bool(claim.text)
        claims.append(claim)
    return claims


def incidents_from_calls(calls: list[dict[str, Any]]) -> list[Incident]:
    """Build one incident per searched clause from search_decisions outputs."""
    incidents: list[Incident] = []
    for call in calls:
        if call["name"] != "search_decisions" or "results" not in call["output"]:
            continue
        out = call["output"]
        parsed = (
            out["parsed_query"] if isinstance(out["parsed_query"], list) else [out["parsed_query"]]
        )
        statuses = out.get("status_by_incident", [out["status"]])
        query = call["arguments"].get("query", "")
        for i, (pq, status) in enumerate(zip(parsed, statuses, strict=True)):
            event = (pq["events"] or [None])[0]
            on_side = [r for r in out["results"] if len(parsed) == 1 or r["event"] == event]
            amb = out.get("ambiguity")
            choices = [_choice(r) for r in on_side if amb and f"({r['id']})" in amb]
            near = [
                {"id": r["id"], "reason": r["reason"]}
                for r in out.get("near_misses", [])
                if len(parsed) == 1 or r.get("event") == event
            ]
            incidents.append(
                Incident(
                    query=query if len(parsed) == 1 else f"{query} [part {i + 1}]",
                    status=status,
                    top_doc=on_side[0]["id"] if on_side else None,
                    event=event,
                    ambiguity=amb if len(choices) > 1 else None,
                    choices=choices if len(choices) > 1 else [],
                    rejected=near if not on_side else [],
                )
            )
    return incidents


def _choice(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "turns": row["turns"],
        "doc_number": row["doc_number"],
        "penalized": row["penalized"],
        "session": row["session"],
    }


def hold_ambiguous(claims: list[Claim], incidents: list[Incident], ctx: RunContext) -> None:
    """Withhold claims about an incident that matches several rulings until the user chooses."""
    held = {c["id"]: c for i in incidents for c in i.choices}
    if not held:
        return
    options = ", ".join(f"Turn {'/'.join(map(str, c['turns']))} ({d})" for d, c in held.items())
    for claim in claims:
        docs = {ctx.seen_evidence.get(e) for e in claim.evidence}
        if claim.supported and docs & set(held):
            claim.supported = False
            claim.problems.append(
                f"the question matches more than one ruling: {options}. Choose one first."
            )


def run_status(claims: list[Claim], incidents: list[Incident]) -> str:
    """Return needs_choice, insufficient_evidence, partial, or answered.

    needs_choice means search matched several rulings for one incident, and the user must
    pick one before any claim about that incident is shown. partial means at least one
    incident has no eligible ruling.
    """
    if any(i.choices for i in incidents):
        return "needs_choice"
    if not any(c.supported for c in claims):
        return "insufficient_evidence"
    if any(i.top_doc is None for i in incidents):
        return "partial"
    return "answered"


def finalize(
    question: str,
    mode: str,
    settings: Settings,
    runner: ToolRunner,
    index: Index,
    transcript: list[dict[str, Any]],
    raw_final: str | None,
    started: float,
    error: str | None = None,
) -> RunRecord:
    """Validate claims and assemble the run record."""
    _, raw_claims, parse_error = parse_final(raw_final or "")
    claims = validate_claims(raw_claims, runner.ctx, runner.toolbox.manifest)
    calls = [asdict(c) for c in runner.calls]
    incidents = incidents_from_calls(calls)
    hold_ambiguous(claims, incidents, runner.ctx)
    status = run_status(claims, incidents)
    return RunRecord(
        run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        question=question,
        mode=mode,
        model=settings.model_name,
        model_base_url=settings.model_base_url,
        started_at=datetime.fromtimestamp(started, UTC).isoformat(timespec="seconds"),
        duration_s=round(time.time() - started, 2),
        status=status,
        claims=claims,
        incidents=incidents,
        tool_calls=calls,
        transcript=transcript,
        budget={
            "max_tool_calls": runner.max_calls,
            "executed": runner.executed,
            "refused": len(runner.calls) - runner.executed,
        },
        index_meta=index.meta(),
        raw_final=raw_final,
        error=error or parse_error,
    )


def repair_feedback(content: str, runner: ToolRunner) -> str | None:
    """Return one corrective message if the answer has no supported claim, else None.

    The run allows one such message. It names the missing step or the bad IDs.
    """
    _, raw, parse_error = parse_final(content)
    manifest = runner.toolbox.manifest
    claims = validate_claims(raw, runner.ctx, manifest)
    left = runner.max_calls - runner.executed
    notes: list[str] = []
    for clause in split_incidents(runner.ctx.question or "", manifest):
        events = find_events(clause, manifest)
        if events and not set(events) & runner.ctx.searched_events:
            notes.append(
                f"This incident was not searched: {clause!r}. Call search_decisions for it."
            )
    if any(c.problems for c in claims if c.evidence) and runner.ctx.seen_evidence:
        bad = [f"{c.text[:60]!r}: {c.problems[0]}" for c in claims if c.problems][:4]
        notes.append(f"Some claims will be removed: {bad}.")
    if not any(c.supported for c in claims):
        if not runner.ctx.seen_evidence:
            searched = [
                r["id"]
                for c in runner.calls
                if c.name == "search_decisions"
                for r in c.output.get("results", [])[:1]
            ]
            notes.append(
                "No claim is supported because get_passages was not called. Call "
                "get_passages with the top result ID of each incident"
                + (f" ({searched})" if searched else "")
                + "."
            )
        else:
            problems = [p for c in claims for p in c.problems][:6]
            notes.append(
                f"No claim is supported. Problems: {problems or [parse_error]}. Cite only "
                f"evidence IDs from get_passages output."
            )
    if not runner.ctx.authority_checked:
        notes.append(
            "resolve_authority was not called. Call it for the event of the first "
            "incident to check which rules were in force."
        )
    if not notes:
        return None
    if left <= 0:
        return (
            "Server check: " + " ".join(notes) + " No tool calls left. Reply with the final JSON."
        )
    return (
        f"Server check: {' '.join(notes)} You have {left} tool calls left. Then reply with "
        "the final JSON."
    )


class ModelClient:
    """Minimal OpenAI-compatible chat client (works with `mlx_lm.server`)."""

    def __init__(self, settings: Settings) -> None:
        """Create an HTTP client for the configured endpoint."""
        self.settings = settings
        headers = (
            {"Authorization": f"Bearer {settings.model_api_key}"} if settings.model_api_key else {}
        )
        self.http = httpx.Client(
            base_url=settings.model_base_url, headers=headers, timeout=settings.model_timeout_s
        )

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Send one chat request with the tool specs.

        Returns:
            The assistant message, plus the choice's finish_reason."""
        body = {
            "model": self.settings.model_name,
            "messages": messages,
            "tools": TOOL_SPECS,
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 1200,
            **json.loads(self.settings.model_extra_body or "{}"),
        }
        resp = self.http.post("/chat/completions", json=body)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        return {**choice["message"], "finish_reason": choice.get("finish_reason")}


def run_agent(
    question: str,
    index: Index,
    runner: ToolRunner,
    settings: Settings,
    client: ModelClient | None = None,
) -> RunRecord:
    """Answer a question with the live model, executing its tool calls server-side."""
    client = client or ModelClient(settings)
    started = time.time()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(settings)},
        {"role": "user", "content": question},
    ]
    transcript: list[dict[str, Any]] = [dict(m) for m in messages]
    raw_final: str | None = None
    error: str | None = None
    repairs = 0
    malformed = 0
    try:
        for _ in range(settings.max_model_turns):
            msg = client.chat(messages)
            tool_calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            entry = {"role": "assistant", "content": content}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            transcript.append(entry)
            if not tool_calls and msg.get("finish_reason") == "tool_calls" and malformed < 2:
                malformed += 1
                note = {
                    "role": "user",
                    "content": "Server check: your last tool call was not valid JSON and was "
                    "not run. Call one tool at a time with flat arguments. For search_decisions, "
                    "send only the query text, without incident_facts.",
                }
                messages.append(note)
                transcript.append({**note, "server_feedback": True})
                continue
            if not tool_calls:
                raw_final = content
                feedback = repair_feedback(content, runner) if repairs < MAX_REPAIRS else None
                if feedback is None:
                    break
                repairs += 1
                note = {"role": "user", "content": feedback}
                messages.append(note)
                transcript.append({**note, "server_feedback": True})
                continue
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                output = runner.execute(fn.get("name", ""), fn.get("arguments") or "{}")
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or f"call_{i}",
                    "name": fn.get("name", ""),
                    "content": json.dumps(output),
                }
                messages.append(tool_msg)
                transcript.append(
                    {**tool_msg, "content": f"<{len(tool_msg['content'])} chars; see tool_calls>"}
                )
        else:
            error = f"model did not finish within {settings.max_model_turns} turns"
    except httpx.HTTPError as exc:
        error = f"model request failed: {exc}"
        logger.error(error)
    return finalize(
        question, "live", settings, runner, index, transcript, raw_final, started, error
    )
