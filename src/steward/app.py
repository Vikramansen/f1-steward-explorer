"""FastAPI server: web UI, live runs, keyless replays, and passage rendering."""

import uuid
from collections.abc import Awaitable, Callable
from importlib import resources
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from steward.agent import run_agent
from steward.config import Settings
from steward.index import Index
from steward.manifest import Manifest, load_manifest
from steward.replay import load_record, replay
from steward.tools import Toolbox, ToolRunner, apex_spans, page_url

HEADLINE = (
    "Why did Norris get five seconds in Austin in 2024, while Verstappen got ten "
    "seconds at Turn 4 in Mexico the next week?"
)


class AskRequest(BaseModel):
    """Body of POST /api/ask."""

    question: str = Field(min_length=3, max_length=500)


class PassageView(BaseModel):
    """One cached excerpt, with official and local PDF links."""

    evidence_id: str
    doc_id: str
    title: str
    source_type: str
    event: str | None
    page: int | None
    label: str
    text: str
    source_url: str | None
    cached_url: str | None
    apex_spans: list[list[int]]
    status: str


class State:
    """Objects shared by all requests: settings, manifest, and index."""

    def __init__(self, settings: Settings) -> None:
        """Load the manifest and open the index."""
        self.settings = settings
        self.manifest: Manifest = load_manifest(settings.manifest_path)
        self.index = Index(settings.index_path)
        self.toolbox = Toolbox(self.index, self.manifest, settings)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or Settings.from_env()
    state = State(settings)
    app = FastAPI(title="F1 Steward Decision Explorer")
    web_dir = Path(str(resources.files("steward").joinpath("web")))
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    def get_state() -> State:
        return state

    StateDep = Annotated[State, Depends(get_state)]

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        with logger.contextualize(request_id=rid):
            logger.info("{} {}", request.method, request.url.path)
            response = await call_next(request)
        response.headers["x-request-id"] = rid
        if request.url.path.startswith("/static/"):
            response.headers["cache-control"] = "no-cache"
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index_page() -> HTMLResponse:
        return HTMLResponse((web_dir / "index.html").read_text(encoding="utf-8"))

    @app.get("/health")
    async def health(st: StateDep) -> dict[str, Any]:
        return {"status": "ok", "documents": len(st.index.doc_ids()), "index": st.index.meta()}

    @app.get("/api/config")
    async def config(st: StateDep) -> dict[str, Any]:
        replays = sorted(p.stem for p in st.settings.replay_dir.glob("*.json"))
        return {
            "model": st.settings.model_name,
            "model_base_url": st.settings.model_base_url,
            "max_tool_calls": st.settings.max_tool_calls,
            "replays": replays,
            "headline": HEADLINE,
        }

    @app.post("/api/ask")
    async def ask(req: AskRequest, st: StateDep) -> dict[str, Any]:
        runner = ToolRunner(st.toolbox, st.settings.max_tool_calls, question=req.question)
        record = await run_in_threadpool(run_agent, req.question, st.index, runner, st.settings)
        return record.model_dump()

    @app.get("/api/replays/{name}")
    async def get_replay(name: str, st: StateDep) -> dict[str, Any]:
        path = st.settings.replay_dir / f"{name}.json"
        if not path.is_file() or path.parent != st.settings.replay_dir:
            raise HTTPException(404, "replay not found")
        return await run_in_threadpool(replay, load_record(path), st.toolbox)

    def passage_views(st: State, ids: list[str]) -> list[PassageView]:
        views: list[PassageView] = []
        for eid in ids[:40]:
            row = st.index.passage(eid)
            if row is None or not row["quotable"]:
                continue
            src = st.index.source(row["doc_id"])
            if src is None:
                continue
            ev = st.manifest.events.get(src.event_key or "")
            views.append(
                PassageView(
                    evidence_id=eid,
                    doc_id=src.id,
                    title=src.title,
                    source_type=src.source_type,
                    event=ev.name if ev else None,
                    page=row["page"],
                    label=row["label"],
                    text=row["text"],
                    source_url=page_url(src, row["page"]),
                    cached_url=f"/cache/{src.id}.pdf#page={row['page']}"
                    if src.cache_file
                    else None,
                    apex_spans=apex_spans(row["text"]),
                    status=src.status,
                )
            )
        return views

    @app.get("/api/passages")
    async def passages(ids: str, st: StateDep) -> list[PassageView]:
        return passage_views(st, [i for i in ids.split(",") if i])

    @app.get("/api/rulings/{doc_id}")
    async def ruling(doc_id: str, st: StateDep) -> list[PassageView]:
        """Excerpts of one current ruling, picked by get_passages. No model call."""
        src = st.index.source(doc_id)
        if src is None or not src.is_decision or src.status != "current":
            raise HTTPException(404, "no current ruling with this ID")
        out = st.toolbox.get_passages([doc_id])
        return passage_views(st, [p["evidence_id"] for p in out["passages"]])

    @app.get("/api/sources")
    async def sources(st: StateDep) -> list[dict[str, Any]]:
        return [
            s.model_dump(mode="json", exclude={"passages", "fact_quotes"})
            for s in st.manifest.sources
        ]

    @app.get("/cache/{doc_id}.pdf")
    async def cached_pdf(doc_id: str, st: StateDep) -> FileResponse:
        src = st.index.source(doc_id)
        if src is None or not src.cache_file:
            raise HTTPException(404, "not cached")
        return FileResponse(st.settings.cache_dir / src.cache_file, media_type="application/pdf")

    return app
