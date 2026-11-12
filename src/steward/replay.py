"""Record live runs and replay them without a model or API key."""

import json
from pathlib import Path
from typing import Any

from steward.agent import (
    RunRecord,
    hold_ambiguous,
    incidents_from_calls,
    run_status,
    validate_claims,
)
from steward.tools import Toolbox, ToolRunner

REPLAY_LABEL = (
    "Replay of a recorded live run. No model call is made. Tool calls are re-run "
    "against the current local index and compared with the recording."
)


def save_record(record: RunRecord, path: Path) -> Path:
    """Write a run record as a replay file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_record(path: Path) -> RunRecord:
    """Read a replay file."""
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def replay(record: RunRecord, toolbox: Toolbox) -> dict[str, Any]:
    """Re-run recorded tool calls on the current index and re-check the recorded claims.

    Returns:
        The record (as a dict) with a "replay" section: label, per-call match flags,
        the claim check against fresh tool outputs, and the resulting status.
    """
    runner = ToolRunner(
        toolbox, max_calls=record.budget["max_tool_calls"], question=record.question
    )
    checks: list[dict[str, Any]] = []
    for call in record.tool_calls:
        if call["status"] == "refused_budget":
            continue
        fresh = runner.execute(call["name"], call["arguments"])
        checks.append(
            {
                "n": call["n"],
                "name": call["name"],
                "identical": _canonical(fresh) == _canonical(call["output"]),
            }
        )
    raw = [{"text": c.text, "evidence": c.evidence} for c in record.claims]
    claims = validate_claims(raw, runner.ctx, toolbox.manifest)
    fresh_calls = [
        {"name": c.name, "arguments": c.arguments, "output": c.output} for c in runner.calls
    ]
    incidents = incidents_from_calls(fresh_calls)
    hold_ambiguous(claims, incidents, runner.ctx)
    status = run_status(claims, incidents)
    data = record.model_dump()
    data["mode"] = "replay"
    data["replay"] = {
        "label": REPLAY_LABEL,
        "recorded_model": record.model,
        "recorded_at": record.started_at,
        "all_tool_outputs_identical": all(c["identical"] for c in checks),
        "call_checks": checks,
        "index_meta": toolbox.index.meta(),
        "status_now": status,
        "claims_now": [c.model_dump() for c in claims],
        "incidents_now": [i.model_dump() for i in incidents],
        "tool_outputs_now": [c.output for c in runner.calls],
    }
    return data
