"""Command-line entry point: `steward <command>`."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from steward.config import Settings
from steward.index import Index
from steward.manifest import load_manifest
from steward.tools import Toolbox

console = Console()
HEADLINE_DOCS = ["usa24-d69", "mex24-d47"]


def _toolbox(settings: Settings) -> Toolbox:
    return Toolbox(Index(settings.index_path), load_manifest(settings.manifest_path), settings)


def cmd_fetch(args: argparse.Namespace, s: Settings) -> int:
    """Download the FIA PDFs and check their SHA-256."""
    from steward.fetch import fetch_sources

    for r in fetch_sources(load_manifest(s.manifest_path), s.cache_dir, force=args.force):
        console.print(f"{r.status:>10}  {r.source_id}")
    return 0


def cmd_build(args: argparse.Namespace, s: Settings) -> int:
    """Build the FTS5 index, optionally without some documents."""
    from steward.index import build_index

    out = Path(args.out) if args.out else s.index_path
    build_index(load_manifest(s.manifest_path), s.cache_dir, out, exclude=args.exclude)
    return 0


def cmd_verify(args: argparse.Namespace, s: Settings) -> int:
    """Check the corpus, and optionally the FIA listing pages."""
    from steward.checks import verify_corpus
    from steward.fetch import verify_listings

    manifest = load_manifest(s.manifest_path)
    problems = verify_corpus(manifest, s.cache_dir)
    if args.online:
        problems += verify_listings(manifest)
    for p in problems:
        console.print(f"[red]FAIL[/red] {p}")
    console.print(
        f"{'OK' if not problems else 'FAILED'}: {len(manifest.sources)} sources checked"
        + (" (with online listing checks)" if args.online else "")
    )
    return 1 if problems else 0


def cmd_search(args: argparse.Namespace, s: Settings) -> int:
    """Run search_decisions and print the JSON output."""
    console.print_json(json.dumps(_toolbox(s).search_decisions(args.query)))
    return 0


def cmd_authority(args: argparse.Namespace, s: Settings) -> int:
    """Run resolve_authority and print the JSON output."""
    console.print_json(json.dumps(_toolbox(s).resolve_authority(args.event, args.date, args.issue)))
    return 0


def cmd_ask(args: argparse.Namespace, s: Settings) -> int:
    """Answer a question with the live model, and optionally record the run."""
    from steward.agent import run_agent
    from steward.replay import save_record
    from steward.tools import ToolRunner

    tb = _toolbox(s)
    runner = ToolRunner(tb, s.max_tool_calls, question=args.question)
    record = run_agent(args.question, tb.index, runner, s)
    for c in record.tool_calls:
        console.print(
            f"[cyan]tool {c['n']}[/cyan] {c['name']}({json.dumps(c['arguments'])}) "
            f"-> {c['status']} in {c['latency_ms']} ms"
        )
    for c in record.claims:
        mark = "[green]supported[/green]" if c.supported else f"[red]removed[/red] {c.problems}"
        console.print(f"- {c.text} {c.evidence} {mark}")
    console.print(f"status: {record.status}  error: {record.error}")
    if args.record:
        console.print(f"recorded to {save_record(record, Path(args.record))}")
    return 0 if record.error is None else 1


def cmd_replay(args: argparse.Namespace, s: Settings) -> int:
    """Replay a recorded run on the current index."""
    from steward.replay import load_record, replay

    data = replay(load_record(Path(args.path)), _toolbox(s))
    rep = data["replay"]
    console.print(f"[bold]{rep['label']}[/bold]")
    console.print(f"recorded model: {rep['recorded_model']} at {rep['recorded_at']}")
    for c in rep["call_checks"]:
        console.print(
            f"  call {c['n']} {c['name']}: {'identical' if c['identical'] else 'CHANGED'}"
        )
    for c in rep["claims_now"]:
        mark = "supported" if c["supported"] else f"removed {c['problems']}"
        console.print(f"  - {c['text']} {c['evidence']} [{mark}]")
    console.print(f"status now: {rep['status_now']}")
    return 0 if rep["all_tool_outputs_identical"] else 1


def cmd_ablation(args: argparse.Namespace, s: Settings) -> int:
    """Replay a run on indexes that each lack one document."""
    from steward.ablation import run_ablation
    from steward.replay import load_record

    result = run_ablation(
        load_record(Path(args.path)), load_manifest(s.manifest_path), s, args.docs or HEADLINE_DOCS
    )
    console.print_json(json.dumps(result))
    return 0


def cmd_eval(args: argparse.Namespace, s: Settings) -> int:
    """Run the retrieval eval and write the results files."""
    from steward.evaluate import METHODS, render_markdown, run_eval

    results = run_eval(
        _toolbox(s), s.eval_dir / "queries.json", repeats=args.repeats, split=args.split
    )
    table = Table(title="Retrieval eval (hit@3 / wrong-source rate / latency)")
    for col in (
        "method",
        "hit@3 dev",
        "hit@3 held-out",
        "hit@3 all",
        "wrong-src all",
        "median ms",
        "p95 ms",
    ):
        table.add_column(col)
    for m in METHODS:
        r = results["summary"][m]
        table.add_row(
            m,
            f"{r['dev']['hit_at_3']:.2f}",
            f"{r['heldout']['hit_at_3']:.2f}",
            f"{r['all']['hit_at_3']:.2f}",
            f"{r['all']['wrong_source_rate']:.2f}",
            str(r["latency_ms"]["median"]),
            str(r["latency_ms"]["p95"]),
        )
    console.print(table)
    for row in results["per_query"]:
        full, base = row["filters_plus_facts"], row["keyword_bm25"]
        console.print(
            f"{row['id']} [{row['split']}] full={'hit' if full['hit'] else 'MISS'} "
            f"{full['top3']} | bm25={'hit' if base['hit'] else 'MISS'} {base['top3']}"
        )
    if not args.no_write and args.split is None:
        (s.eval_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        (s.eval_dir / "RESULTS.md").write_text(render_markdown(results))
        console.print(f"wrote {s.eval_dir / 'results.json'} and RESULTS.md")
    return 0


def cmd_serve(args: argparse.Namespace, s: Settings) -> int:
    """Start the web UI."""
    import uvicorn

    from steward.app import create_app

    uvicorn.run(create_app(s), host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a command."""
    parser = argparse.ArgumentParser(prog="steward", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="download FIA PDFs into data/cache and check SHA-256")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("build", help="build the SQLite FTS5 index from the cache")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--out")
    p = sub.add_parser("verify", help="check cache, fact quotes, links (and FIA listings online)")
    p.add_argument("--online", action="store_true")
    p = sub.add_parser("search", help="run search_decisions")
    p.add_argument("query")
    p = sub.add_parser("authority", help="run resolve_authority")
    p.add_argument("issue")
    p.add_argument("--event")
    p.add_argument("--date")
    p = sub.add_parser("ask", help="answer with the live model")
    p.add_argument("question")
    p.add_argument("--record", help="save the run as a replay JSON file")
    p = sub.add_parser("replay", help="replay a recorded run without a model")
    p.add_argument("path")
    p = sub.add_parser("ablation", help="replay with each headline doc removed from the index")
    p.add_argument("path")
    p.add_argument("--docs", nargs="*")
    p = sub.add_parser("eval", help="run the retrieval eval")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--split", choices=["dev", "heldout"], help="run one split only (no write)")
    p.add_argument("--no-write", action="store_true")
    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    handlers = {
        "fetch": cmd_fetch,
        "build": cmd_build,
        "verify": cmd_verify,
        "search": cmd_search,
        "authority": cmd_authority,
        "ask": cmd_ask,
        "replay": cmd_replay,
        "ablation": cmd_ablation,
        "eval": cmd_eval,
        "serve": cmd_serve,
    }
    return handlers[args.cmd](args, settings)


if __name__ == "__main__":
    sys.exit(main())
