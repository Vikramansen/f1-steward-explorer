# F1 Steward Decision Explorer

A small retrieval app over official FIA steward documents. Ask a question
such as "Why did Norris get five seconds in Austin while Verstappen got ten
in Mexico?" and it finds the matching rulings, shows the quoted passages
with links to the source PDFs, and explains why similar documents were
left out.

## How it works

- **Sources.** A manifest of 18 FIA documents from the 2024 United States
  and Mexico City Grands Prix: steward decisions, sporting regulations, and
  driving guidelines. Each entry pins a URL and a SHA-256 hash.
- **Index.** The PDFs are downloaded, checked against their hashes, split
  into passages, and stored in SQLite with full-text search.
- **Retrieval.** Version, date, and event checks run before ranking. A
  recalled document or a guideline from a later season is never returned
  as a match.
- **Answers.** A small local model can call three tools (find rulings,
  check which rules applied, fetch passages). Every claim must cite a
  passage returned in that run. If a question matches more than one
  ruling, the app asks you to choose.
- **Replays.** Recorded runs can be replayed locally without a model.

## Try it

Requires Python 3.11+ and `uv`.

```bash
git clone https://github.com/Vikramansen/f1-steward-explorer.git
cd f1-steward-explorer
uv sync
uv run steward fetch
uv run steward build
uv run steward serve
```

The app runs on your machine at `http://127.0.0.1:8000`.

[Engineering notes and evaluation](docs/engineering.md)
