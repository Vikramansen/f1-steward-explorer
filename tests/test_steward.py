import dataclasses
from pathlib import Path

import pytest

from steward.ablation import run_ablation
from steward.agent import validate_claims
from steward.checks import verify_corpus
from steward.config import Settings
from steward.index import Index, build_index
from steward.manifest import Manifest
from steward.replay import load_record, replay
from steward.tools import Toolbox, ToolRunner

from .conftest import HEADLINE

REPLAYS = Path(__file__).resolve().parents[1] / "replays"


def top_ids(out: dict) -> list[str]:
    return [r["id"] for r in out["results"]]


def test_corpus_verifies(manifest: Manifest, settings: Settings, index_path: Path) -> None:
    assert verify_corpus(manifest, settings.cache_dir) == []


def test_headline_retrieves_doc69_and_doc47(toolbox: Toolbox) -> None:
    out = toolbox.search_decisions(HEADLINE)
    assert {"usa24-d69", "mex24-d47"} <= set(top_ids(out)[:3])
    assert "mex24-d43" not in top_ids(out)


def test_turn4_and_turn8_are_separated(toolbox: Toolbox) -> None:
    t4 = toolbox.search_decisions("Verstappen penalty at Turn 4 in Mexico 2024")
    t8 = toolbox.search_decisions("Verstappen penalty at Turn 8 in Mexico 2024")
    assert top_ids(t4)[0] == "mex24-d47"
    assert top_ids(t8)[0] == "mex24-d44"
    both = toolbox.search_decisions("Verstappen penalty for his battle with Norris in Mexico 2024")
    assert both["ambiguity"] and "mex24-d44" in both["ambiguity"]


def test_recalled_doc_is_excluded_and_not_quotable(toolbox: Toolbox) -> None:
    out = toolbox.search_decisions("Mexico 2024 Doc 43 Verstappen Turn 4")
    assert "mex24-d43" in {e["id"] for e in out["excluded_before_ranking"]}
    assert top_ids(out)[0] == "mex24-d47"
    errors = toolbox.get_passages(["mex24-d43"])["errors"]
    assert errors and "recalled" in errors[0]["error"]


def test_later_guidelines_are_not_authority_for_2024(toolbox: Toolbox) -> None:
    out = toolbox.resolve_authority(
        "2024 Mexico City Grand Prix", issue="driving standards guidelines"
    )
    rejected = {r["id"]: r["failure_mode"] for r in out["rejected"]}
    assert rejected["dsg-2026-v01"] in {"not_in_force_on_date", "wrong_season"}
    assert "dsg-2025-v41" in rejected
    assert not any(a["source_type"] == "driving_guideline" for a in out["applicable"])
    assert out["status_by_type"]["driving_guideline"] == "insufficient_evidence"
    assert out["unavailable"][0]["id"] == "dsg-2024"
    assert "sr-2024-i7" in {a["id"] for a in out["applicable"]}
    austin = toolbox.resolve_authority("2024 United States Grand Prix", issue="guidelines")
    cited = austin["unavailable"][0]["what_rulings_establish"]
    assert "usa24-d69:p1:reason-5" in {c["evidence_id"] for c in cited}


def test_wrong_date_falls_back_to_event(toolbox: Toolbox) -> None:
    out = toolbox.resolve_authority("2024 United States Grand Prix", date="2026-03-06")
    assert out["operative_date"] == "2024-10-20" and out["warnings"]


def test_unsupported_question_is_insufficient(toolbox: Toolbox) -> None:
    out = toolbox.search_decisions("Why did Hamilton get a penalty at Turn 1 in Austin 2024?")
    assert out["status"] == "insufficient_evidence" and out["results"] == []


def test_excerpt_urls_match_manifest(toolbox: Toolbox, manifest: Manifest) -> None:
    out = toolbox.get_passages(["usa24-d69", "mex24-d47"])
    by_id = {s.id: s for s in manifest.sources}
    assert out["passages"]
    for p in out["passages"]:
        src = by_id[p["doc_id"]]
        assert p["source_url"] == f"{src.canonical_url}#page={p['page']}"
        assert len(p["text"]) <= 400
    assert any(p.get("apex_spans") for p in out["passages"])


def test_budget_is_capped(toolbox: Toolbox) -> None:
    runner = ToolRunner(toolbox, max_calls=6)
    for _ in range(8):
        runner.execute("search_decisions", {"query": "Norris Austin 2024"})
    assert runner.executed == 6
    assert [c.status for c in runner.calls][-2:] == ["refused_budget", "refused_budget"]


def test_claim_validation(toolbox: Toolbox, manifest: Manifest) -> None:
    runner = ToolRunner(toolbox, max_calls=6, question=HEADLINE)
    runner.execute("search_decisions", {"query": HEADLINE})
    runner.execute("get_passages", {"document_ids": ["usa24-d69", "mex24-d47"]})
    raw = [
        {"text": "Norris got a 5 second penalty in Austin.", "evidence": ["usa24-d69:p1:decision"]},
        {
            "text": "Norris got a 10 second penalty in Austin.",
            "evidence": ["usa24-d69:p1:decision"],
        },
        {"text": "Verstappen was penalized in Austin.", "evidence": ["mex24-d47:p1:decision"]},
        {"text": "No evidence cited.", "evidence": []},
        {"text": "Made-up ID.", "evidence": ["usa24-d69:p9:reason-9"]},
        {
            "text": "The Driving Standards Guidelines set 10 seconds.",
            "evidence": ["usa24-d69:p1:reason-5"],
        },
    ]
    supported = [c.supported for c in validate_claims(raw, runner.ctx, manifest)]
    assert supported == [True, False, False, False, False, False]


def test_headline_replay_is_identical(toolbox: Toolbox) -> None:
    out = replay(load_record(REPLAYS / "headline.json"), toolbox)["replay"]
    assert out["all_tool_outputs_identical"]
    assert out["status_now"] == "answered"
    assert {i["top_doc"] for i in out["incidents_now"]} >= {"usa24-d69", "mex24-d47"}


def test_ambiguous_question_waits_for_choice(toolbox: Toolbox) -> None:
    out = replay(load_record(REPLAYS / "ambiguous-mexico-turn.json"), toolbox)["replay"]
    assert out["all_tool_outputs_identical"]
    assert out["status_now"] == "needs_choice"
    (inc,) = out["incidents_now"]
    assert {c["id"] for c in inc["choices"]} == {"mex24-d47", "mex24-d44"}
    assert not any(c["supported"] for c in out["claims_now"])


def test_appendix_l_is_unverified_and_not_citable(toolbox: Toolbox, manifest: Manifest) -> None:
    runner = ToolRunner(toolbox, max_calls=6)
    out = runner.execute("resolve_authority", {"event": "2024 Mexico City Grand Prix"})
    assert [a["id"] for a in out["unverified"]] == ["isc-appl-2024"]
    assert "isc-appl-2024" not in {a["id"] for a in out["applicable"]}
    assert out["status_by_type"]["isc_appendix"] == "unverified"
    runner.execute("get_passages", {"document_ids": ["isc-appl-2024"]})
    raw = [
        {
            "text": "Appendix L bans forcing a driver off the track.",
            "evidence": ["isc-appl-2024:p56:ch4-art2b-1"],
        }
    ]
    (claim,) = validate_claims(raw, runner.ctx, manifest)
    assert not claim.supported and "not verified" in claim.problems[0]


def test_rulings_api(index_path: Path, settings: Settings) -> None:
    from fastapi.testclient import TestClient

    from steward.app import create_app

    client = TestClient(create_app(dataclasses.replace(settings, index_path=index_path)))
    views = client.get("/api/rulings/mex24-d44").json()
    assert views and all(v["doc_id"] == "mex24-d44" for v in views)
    assert client.get("/api/rulings/mex24-d43").status_code == 404
    assert client.get("/api/rulings/sr-2024-i7").status_code == 404


def test_missing_side_names_the_rejected_ruling(
    manifest: Manifest, settings: Settings, index_path: Path, tmp_path: Path
) -> None:
    s = dataclasses.replace(settings, index_path=index_path, data_dir=tmp_path)
    build_index(manifest, s.cache_dir, tmp_path / "cut.sqlite", exclude=["mex24-d47"])
    tb = Toolbox(Index(tmp_path / "cut.sqlite"), manifest, s)
    out = replay(load_record(REPLAYS / "headline.json"), tb)["replay"]
    assert out["status_now"] == "partial"
    sides = {i["event"]: i for i in out["incidents_now"]}
    assert sides["2024 United States Grand Prix"]["top_doc"] == "usa24-d69"
    mexico = sides["2024 Mexico City Grand Prix"]
    assert mexico["top_doc"] is None
    assert any(r["id"] == "mex24-d44" and "Turn 8" in r["reason"] for r in mexico["rejected"])


@pytest.mark.parametrize("doc_id", ["usa24-d69", "mex24-d47"])
def test_ablation_changes_result(
    doc_id: str, manifest: Manifest, settings: Settings, index_path: Path, tmp_path: Path
) -> None:
    s = dataclasses.replace(settings, index_path=index_path, data_dir=tmp_path)
    out = run_ablation(load_record(REPLAYS / "headline.json"), manifest, s, [doc_id])
    full, cut = out["full_index"], out[f"without_{doc_id}"]
    assert full["unsupported_claims"] == 0
    assert not cut["tool_outputs_identical"]
    assert doc_id not in {top for _, _, top in cut["incidents"]}
    assert cut["supported_claims"] < full["supported_claims"]
