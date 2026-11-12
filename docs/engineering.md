# Engineering notes

Independent project built on public FIA documents. Not affiliated with or endorsed by the FIA or
Formula 1. The repository stores links, SHA-256 hashes, and short excerpts. `steward fetch`
downloads the PDFs from fia.com.

## Sources

`data/manifest.json` lists 18 sources, each with canonical URL, SHA-256, event, date, session,
turn, document number, type, and version status. Document numbers repeat across events (Austin
and Mexico both have a Doc 43 and a Doc 44), so IDs are event-scoped: `usa24-d69`, `mex24-d47`.

| Source | Label | Basis |
|---|---|---|
| Austin 2024 Doc 69 (Norris, Turn 12, 5 s) | current | FIA event listing |
| Mexico 2024 Doc 47 (Verstappen, Turn 4, 10 s) | current, corrected | Listed at 23:40 as the "(corrected)" version of Doc 43 |
| Mexico 2024 Doc 43 | recalled, no text | Listed as "Recalled - Doc 43 ..." at 22:30 with no PDF |
| Mexico 2024 Doc 44 (Verstappen, Turn 8, 10 s) | current | Different incident with the same drivers |
| 9 other 2024 Austin and Mexico decisions | current | Distractors and comparison rulings |
| 2024 Sporting Regulations Issue 7 | applicable | Last 2024 issue (31 July 2024). 2025 Issue 2 calls its black text unchanged from Issue 7 |
| Appendix L, ISC 2024 | **unverified, not citable** | Consolidated file published 11 Dec 2024, after the events. Art. 2 b) matches the 6 Dec 2023 edition word for word, but amendments carry no dates, so the text in force on the race dates is not established |
| Driving Standards Guidelines 2025 v4.1, 2026 v01 | rejected for 2024 | Not in force on the race dates. First made public 25 June 2025 |
| Driving Standards Guidelines 2024 | not public | The UI shows what the 2024 rulings say about them instead |

## Design

Two ideas drive the design:

- Check each source's version and date scope before ranking by relevance.
- Rank rulings by incident facts, not only by text: overtake side, position at the apex, room
  at the exit, advantage retained, and the role of the penalized driver.

The model (default `mlx-community/Qwen3-4B-Instruct-2507-4bit` via `mlx_lm.server`) can call
three tools. The server runs at most 6 calls per run.

- `resolve_authority(event, date, issue)` checks season, in-force window, publication date, and
  recall status. Sources whose edition on the date is not established are labeled `unverified`
  and cannot be cited.
- `search_decisions(query, incident_facts)` removes superseded versions, then applies hard
  filters (season, event, drivers, turn, session), then ranks by BM25 plus matched facts minus
  contradicted facts. Facts the user did not state stay unknown. If several rulings match the
  same drivers at different turns, the UI asks the user to choose. If one side of a comparison
  has no eligible ruling, that side says so and names the closest rejected rulings.
- `get_passages(document_ids)` returns excerpts of at most 400 characters, with page links and
  evidence IDs such as `usa24-d69:p1:reason-2`.

**What claim validation checks.** It checks citations and selected facts, not full entailment.

- Each cited ID was returned by `get_passages` in this run.
- The cited source is eligible: a target or comparison ruling, or an applicable authority.
- Events, drivers, turns, and penalty lengths in the claim match the cited rulings.
- Named rules (Sporting Regulations, Appendix L, Driving Standards Guidelines, article and issue
  numbers) appear in the cited text.

A claim can pass every check and still overstate what a passage says. The UI renders the
excerpts from the local cache, not from model text, so a reader can compare them.

## Evaluation

There are 12 labeled queries in [eval/queries.json](../eval/queries.json). The 8 dev queries
were used for tuning. The 4 held-out queries were written before tuning and run once. Full
results: [eval/RESULTS.md](../eval/RESULTS.md).

| Method | hit@3 dev | hit@3 held-out | Queries with a wrong source in top 3 |
|---|---|---|---|
| Keyword BM25 (FTS5 only) | 4/8 | 2/4 | 8/12 |
| Authority/date filters | 8/8 | 4/4 | 1/12 |
| Filters + incident facts | 8/8 | 4/4 | 0/12 |

- **BM25 failures.** BM25 misses the COTA paraphrase, ranks the recalled Doc 43 in the top 3,
  and returns the 2025/2026 guidelines for a 2024 question.
- **What incident facts add.** Filters alone put Doc 49 in the top 3 for the Mexico query
  without driver names (q03). Incident facts remove it.
- **Latency.** All three methods take well under 1 ms per query (local SQLite, 30 repeats).

12 queries over 18 sources shows that the checks work. It does not show that the ranking
generalizes.

## Known model failures

These are recorded runs in `replays/`. The UI and `steward replay` re-run them without a model.

| Replay | What happened |
|---|---|
| `headline` | Works: 4 tool calls, Doc 69 and Doc 47, 2 claims pass the checks. |
| `ambiguous-mexico-turn` | The model picked Turn 8 without asking. The server holds all 4 claims and asks the user to choose between Turn 4 and Turn 8. |
| `unsupported-hamilton` | No ruling matches. The model fetched Doc 69 (a near miss) anyway. The server marks it as not citable, and the run ends with insufficient evidence. |
| `failure-qwen3-1.7b-headline` | Qwen3-1.7B repeated a search, never fetched passages, and cited a placeholder ID. |

Other failures seen with the 4B model, each now handled in code:

- **Invented facts** (Turn 4 for Austin). Facts not in the question are dropped.
- **Citing a near-miss ruling.** Only eligible sources can be cited.
- **Wrong dates.** The race date is used instead.
- **Overstated rules.** One claim said Doc 47's "standard penalty" was defined in Sporting
  Regulations Issue 7. It was removed, because the cited text does not name it.
- **Malformed tool calls.** The model often drops a closing brace in nested arguments, so
  `mlx_lm.server` discards the call. The server asks for flat arguments.

## Verification

All of these were run from a clean clone, with no model server and no API key:

```bash
uv sync
uv run steward fetch       # 17 PDFs, SHA-256 checked
uv run steward build
uv run steward verify      # hashes, fact quotes, page ranges, links
uv run pytest -q
uv run steward eval
uv run steward ablation replays/headline.json --docs mex24-d47
uv run steward serve       # local UI at http://127.0.0.1:8000
```

The ablation run removes Doc 47 and replays the headline run. The Mexico side then reports "no
eligible ruling" and names Doc 44 as rejected (Turn 8, question asks Turn 4). Doc 44 does not
take Doc 47's place.

Optional live mode:

```bash
uv sync --extra mlx
uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 --prompt-cache-size 2
```

Any OpenAI-compatible endpoint that supports tool calls works. Configure it with
`STEWARD_MODEL_BASE_URL`, `STEWARD_MODEL`, `STEWARD_API_KEY`, and `STEWARD_MODEL_TIMEOUT`. Live
runs take about 40–80 s on a laptop.

## Limitations

- The incident facts are hand-annotated. `steward verify` checks that each one has a quote in
  the PDF.
- The question parser uses rules and covers the phrasings in this corpus.
- Claim validation does not check entailment.
- The 2024 Driving Standards Guidelines are not public, and the Appendix L edition in force on
  the race dates is not established.
