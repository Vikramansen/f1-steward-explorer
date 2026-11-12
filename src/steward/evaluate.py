"""Compare plain keyword retrieval with scope filtering plus incident-fact ranking."""

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from steward.tools import Toolbox

METHODS = ("keyword_bm25", "filters_only", "filters_plus_facts")


@dataclass
class Retrieval:
    """Ranked document IDs from one method for one query."""

    top: list[str]
    insufficient: bool


def keyword_bm25(toolbox: Toolbox, query: dict[str, Any]) -> Retrieval:
    """Plain FTS5 BM25 over every passage of every source. No scope checks."""
    hits = toolbox.index.search_docs(query["query"])
    top = [h.doc_id for h in hits[:3]]
    return Retrieval(top, insufficient=not top)


def _scoped(toolbox: Toolbox, query: dict[str, Any], use_facts: bool) -> Retrieval:
    if query["kind"] == "authority":
        out = toolbox.resolve_authority(event=None, date=None, issue=query["query"])
        wanted = out["requested_types"] or None
        apps = [a for a in out["applicable"] if wanted is None or a["source_type"] in wanted]
        statuses = [out["status_by_type"][t] for t in (wanted or out["status_by_type"])]
        return Retrieval(
            [a["id"] for a in apps[:3]],
            insufficient=all(s == "insufficient_evidence" for s in statuses),
        )
    out = toolbox.search_decisions(query["query"], use_facts=use_facts)
    return Retrieval(
        [r["id"] for r in out["results"][:3]], insufficient=out["status"] == "insufficient_evidence"
    )


def filters_only(toolbox: Toolbox, query: dict[str, Any]) -> Retrieval:
    """Authority, date, and supersession filters with hard constraints; text ranking only."""
    return _scoped(toolbox, query, use_facts=False)


def filters_plus_facts(toolbox: Toolbox, query: dict[str, Any]) -> Retrieval:
    """The full system: filters, then text plus incident-fact ranking."""
    return _scoped(toolbox, query, use_facts=True)


RUNNERS: dict[str, Callable[[Toolbox, dict[str, Any]], Retrieval]] = {
    "keyword_bm25": keyword_bm25,
    "filters_only": filters_only,
    "filters_plus_facts": filters_plus_facts,
}


def score(query: dict[str, Any], r: Retrieval) -> dict[str, Any]:
    """Score one retrieval against its label.

    hit@3: all expected IDs are in the top 3, or, for an expected
    "insufficient evidence" query, the method returned insufficient evidence.
    wrong_source: a labeled wrong source appears in the top 3.
    """
    if query["expect_insufficient"]:
        hit = r.insufficient
    else:
        hit = bool(query["expected"]) and set(query["expected"]) <= set(r.top)
    wrong_ids = query["wrong_sources"]
    wrong = [d for d in r.top if "*" in wrong_ids or d in wrong_ids]
    return {"top3": r.top, "insufficient": r.insufficient, "hit": hit, "wrong": wrong}


def run_eval(
    toolbox: Toolbox, queries_path: Path, repeats: int = 30, split: str | None = None
) -> dict[str, Any]:
    """Run every method on every query (optionally one split) and collect metrics."""
    queries = json.loads(queries_path.read_text(encoding="utf-8"))["queries"]
    queries = [q for q in queries if split is None or q["split"] == split]
    per_query: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {m: [] for m in METHODS}
    for q in queries:
        row: dict[str, Any] = {
            "id": q["id"],
            "split": q["split"],
            "kind": q["kind"],
            "tags": q["tags"],
            "query": q["query"],
            "expected": q["expected"] or "insufficient evidence",
        }
        for m in METHODS:
            result = RUNNERS[m](toolbox, q)
            for _ in range(repeats):
                t0 = time.perf_counter()
                RUNNERS[m](toolbox, q)
                latencies[m].append((time.perf_counter() - t0) * 1000)
            row[m] = score(q, result)
        per_query.append(row)

    summary: dict[str, Any] = {}
    for m in METHODS:
        summary[m] = {}
        for split in ("dev", "heldout", "all"):
            rows = [r for r in per_query if split == "all" or r["split"] == split]
            n = max(len(rows), 1)
            summary[m][split] = {
                "n": len(rows),
                "hit_at_3": round(sum(r[m]["hit"] for r in rows) / n, 3),
                "wrong_source_rate": round(sum(bool(r[m]["wrong"]) for r in rows) / n, 3),
            }
        lat = sorted(latencies[m])
        summary[m]["latency_ms"] = {
            "median": round(statistics.median(lat), 2),
            "p95": round(lat[int(0.95 * (len(lat) - 1))], 2),
            "samples": len(lat),
        }
    return {
        "summary": summary,
        "per_query": per_query,
        "repeats": repeats,
        "index_meta": toolbox.index.meta(),
    }


def render_markdown(results: dict[str, Any]) -> str:
    """Render eval results as a Markdown report."""
    s = results["summary"]
    lines = [
        "# Retrieval eval results",
        "",
        f"Generated by `steward eval`. Latency: {results['repeats']} timed repeats per "
        "query and method, local SQLite, warm cache.",
        "",
        "| Method | hit@3 dev | hit@3 held-out | hit@3 all | wrong-source rate all "
        "| median ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        lines.append(
            f"| {m} | {s[m]['dev']['hit_at_3']:.2f} ({s[m]['dev']['n']}) "
            f"| {s[m]['heldout']['hit_at_3']:.2f} ({s[m]['heldout']['n']}) "
            f"| {s[m]['all']['hit_at_3']:.2f} | {s[m]['all']['wrong_source_rate']:.2f} "
            f"| {s[m]['latency_ms']['median']} | {s[m]['latency_ms']['p95']} |"
        )
    lines += [
        "",
        "## Per query",
        "",
        "| ID | Split | Tags | Expected | keyword_bm25 top 3 | filters_plus_facts top 3 |",
        "|---|---|---|---|---|---|",
    ]

    def cell(r: dict[str, Any]) -> str:
        top = ", ".join(r["top3"]) or "(none)"
        flag = "hit" if r["hit"] else "MISS"
        wrong = f"; wrong: {', '.join(r['wrong'])}" if r["wrong"] else ""
        insuff = "; insufficient" if r["insufficient"] else ""
        return f"{flag}: {top}{insuff}{wrong}"

    for r in results["per_query"]:
        exp = r["expected"] if isinstance(r["expected"], str) else ", ".join(r["expected"])
        lines.append(
            f"| {r['id']} | {r['split']} | {', '.join(r['tags'])} | {exp} "
            f"| {cell(r['keyword_bm25'])} | {cell(r['filters_plus_facts'])} |"
        )
    failures = [
        r
        for r in results["per_query"]
        if not r["filters_plus_facts"]["hit"] or r["filters_plus_facts"]["wrong"]
    ]
    lines += ["", "## Failures of the full system", ""]
    if not failures:
        lines.append("None on this query set.")
    for r in failures:
        lines.append(f"- {r['id']} ({r['split']}): {r['query']} -> {cell(r['filters_plus_facts'])}")
    return "\n".join(lines) + "\n"
