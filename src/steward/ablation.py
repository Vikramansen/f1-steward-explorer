"""Remove one document from the index and show how the headline result changes."""

from typing import Any

from steward.agent import RunRecord
from steward.config import Settings
from steward.index import Index, build_index
from steward.manifest import Manifest
from steward.replay import replay
from steward.tools import Toolbox


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    """Reduce a replay result to status, incidents, and claim counts."""
    rep = data["replay"]
    return {
        "status": rep["status_now"],
        "tool_outputs_identical": rep["all_tool_outputs_identical"],
        "incidents": [(i["event"], i["status"], i["top_doc"]) for i in rep["incidents_now"]],
        "rejected_instead": {
            i["event"]: [f"{r['id']}: {r['reason']}" for r in i["rejected"]]
            for i in rep["incidents_now"]
            if i["rejected"]
        },
        "supported_claims": sum(c["supported"] for c in rep["claims_now"]),
        "unsupported_claims": sum(not c["supported"] for c in rep["claims_now"]),
    }


def run_ablation(
    record: RunRecord, manifest: Manifest, settings: Settings, doc_ids: list[str]
) -> dict[str, Any]:
    """Replay a recorded run on the full index and on indexes missing each document."""
    out: dict[str, Any] = {}
    full = Toolbox(Index(settings.index_path), manifest, settings)
    out["full_index"] = summarize(replay(record, full))
    for doc_id in doc_ids:
        path = settings.data_dir / "ablation" / f"without-{doc_id}.sqlite"
        build_index(manifest, settings.cache_dir, path, exclude=[doc_id])
        tb = Toolbox(Index(path), manifest, settings)
        out[f"without_{doc_id}"] = summarize(replay(record, tb))
    return out
