"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let CONFIG = null;
let SOURCES = {};

const MODE_TEXT = {
  superseded_version: "Superseded version",
  not_in_force_on_date: "Not in force on the event date",
  wrong_season: "Wrong season",
  different_incident: "Different incident",
};
const LOOKS_RIGHT = {
  superseded_version: "Same event, car, turn, and offence title as the target ruling.",
  not_in_force_on_date: "Official FIA text on the same topic, with matching wording.",
  wrong_season: "Official FIA text on the same topic.",
  different_incident: "Same event and same drivers, so keyword search ranks it high.",
};

async function getJSON(url, opts) {
  const resp = await fetch(url, opts);
  if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
  return resp.json();
}

async function init() {
  CONFIG = await getJSON("/api/config");
  for (const s of await getJSON("/api/sources")) SOURCES[s.id] = s;
  $("question").value = CONFIG.headline;
  $("model-line").textContent = `Live model: ${CONFIG.model} via ${CONFIG.model_base_url}. ` +
    `Hard limit: ${CONFIG.max_tool_calls} tool calls per run.`;
  const sel = $("replay-select");
  const names = [...CONFIG.replays].sort((a, b) => (a === "headline" ? -1 : b === "headline" ? 1 : a.localeCompare(b)));
  sel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  $("run-live").onclick = () => run(() => getJSON("/api/ask", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ question: $("question").value }),
  }));
  $("run-replay").onclick = () => run(() => getJSON(`/api/replays/${encodeURIComponent(sel.value)}`));
  renderSources();
  const fromHash = () => {
    const m = location.hash.match(/^#replay=?(.*)$/);
    if (m) { if (m[1]) sel.value = decodeURIComponent(m[1]); $("run-replay").click(); }
  };
  window.addEventListener("hashchange", fromHash);
  fromHash();
}

async function run(fetcher) {
  $("loading").classList.remove("hidden");
  $("results").classList.add("hidden");
  $("banner").classList.add("hidden");
  try {
    const data = await fetcher();
    await render(toView(data));
  } catch (err) {
    showBanner("warn", `Run failed: ${esc(err.message)}. To run without a model, use a recorded replay.`);
  } finally {
    $("loading").classList.add("hidden");
  }
}

function toView(data) {
  if (data.mode === "replay") {
    const r = data.replay;
    let i = 0;
    const calls = data.tool_calls.map((c) => {
      if (c.status === "refused_budget") return c;
      const check = r.call_checks[i];
      const out = { ...c, recorded_output: c.output, output: r.tool_outputs_now[i], identical: check.identical };
      i += 1;
      return out;
    });
    return { mode: "replay", data, question: data.question, claims: r.claims_now,
      incidents: r.incidents_now, status: r.status_now, calls, budget: data.budget,
      identical: r.all_tool_outputs_identical, label: r.label, index: r.index_meta };
  }
  return { mode: "live", data, question: data.question, claims: data.claims,
    incidents: data.incidents, status: data.status, calls: data.tool_calls, budget: data.budget,
    error: data.error };
}

function showBanner(kind, html) {
  const b = $("banner");
  b.className = `banner ${kind}`;
  b.innerHTML = html;
}

async function render(v) {
  $("question").value = v.question;
  if (v.mode === "replay") {
    const excluded = JSON.parse(v.index.excluded || "[]");
    showBanner(v.identical ? "replay" : "warn",
      `<span class="mode">REPLAY.</span> ${esc(v.label)}<br>Recorded with <code>${esc(v.data.model)}</code> ` +
      `at ${esc(v.data.started_at)}. Re-run on the current index: ` +
      (v.identical ? "<b>all tool outputs identical</b>." : "<b>tool outputs CHANGED</b> (see trace).") +
      (excluded.length ? ` Index built without: <code>${esc(excluded.join(", "))}</code>.` : ""));
  } else {
    showBanner(v.error ? "warn" : "live", `<span class="mode">LIVE.</span> Model <code>${esc(v.data.model)}</code>, ` +
      `${v.data.duration_s}s.` + (v.error ? ` Problem: ${esc(v.error)}` : ""));
  }
  const ids = new Set();
  for (const c of v.calls) {
    if (c.name === "get_passages") (c.output.passages || []).forEach((p) => ids.add(p.evidence_id));
    if (c.name === "resolve_authority") (c.output.unavailable || []).forEach((u) =>
      (u.what_rulings_establish || []).forEach((e) => ids.add(e.evidence_id)));
  }
  const passages = ids.size ? await getJSON(`/api/passages?ids=${encodeURIComponent([...ids].join(","))}`) : [];
  const byId = Object.fromEntries(passages.map((p) => [p.evidence_id, p]));
  const fetched = new Set();
  for (const c of v.calls) if (c.name === "get_passages") (c.output.passages || []).forEach((p) => fetched.add(p.evidence_id));

  renderRulings(v, byId, fetched);
  renderClaims(v, byId);
  renderWhy(v, byId);
  renderTrace(v);
  $("results").classList.remove("hidden");
}

function highlighted(text, spans) {
  let out = "", pos = 0;
  for (const [s, e] of spans || []) {
    out += esc(text.slice(pos, s)) + "<mark>" + esc(text.slice(s, e)) + "</mark>";
    pos = e;
  }
  return out + esc(text.slice(pos));
}

function excerptHTML(p) {
  const apex = p.apex_spans && p.apex_spans.length;
  return `<div class="excerpt${apex ? " apex" : ""}" id="ev-${esc(p.evidence_id)}">
    ${apex ? "<b>Position at the apex:</b> " : ""}${highlighted(p.text, p.apex_spans)}
    <div class="src"><span class="eid">${esc(p.evidence_id)}</span>
      <a href="${esc(p.source_url)}" target="_blank" rel="noopener">Official FIA PDF, page ${esc(p.page)}</a>
      ${p.cached_url ? ` · <a href="${esc(p.cached_url)}" target="_blank">cached copy</a>` : ""}</div></div>`;
}

function searchRow(v, docId) {
  for (const c of v.calls) {
    if (c.name !== "search_decisions") continue;
    const row = (c.output.results || []).find((r) => r.id === docId);
    if (row) return row;
  }
  return null;
}

const turnLabel = (turns) => `Turn ${(turns || []).join("/")}`;

function rulingBody(docId, mine, row) {
  const src = SOURCES[docId] || {};
  const field = (k) => mine.find((p) => p.evidence_id.endsWith(`:${k}`));
  const drv = field("driver"), dec = field("decision"), fact = field("fact"), inf = field("infringement");
  const reasons = mine.filter((p) => /:reason-\d+$/.test(p.evidence_id))
    .sort((a, b) => (b.apex_spans.length - a.apex_spans.length));
  const kv = (label, p) => p ? `<dt>${label}</dt><dd>${esc(p.text)} <a class="eid" href="${esc(p.source_url)}" target="_blank" rel="noopener">${esc(p.evidence_id)}</a></dd>` : "";
  return `<h3>${esc((row && row.event) || src.event_key)} · Doc ${esc(src.doc_number)}</h3>
    <div class="meta">${esc(src.title)}<br>${esc(src.document_date)} · ${esc(src.session)} · ${esc(turnLabel(src.turns))}
      · status: ${esc(src.status)}</div>
    ${mine.length ? `<dl class="kv">${kv("Penalized", drv)}${kv("Decision", dec)}${kv("Fact", fact)}${kv("Infringement", inf)}</dl>
      ${reasons.map(excerptHTML).join("")}` :
      `<p class="muted">search_decisions returned this ruling, but get_passages was not called for it in this run, so no excerpt is shown.</p>`}
    ${row && row.matched_facts ? `<p class="muted">Matched: ${esc(row.matched_facts.join(", ") || "text only")}</p>` : ""}`;
}

function choiceCard(inc, question) {
  const buttons = inc.choices.map((c) => `<button class="choice" data-doc="${esc(c.id)}"
      data-turn="${esc(c.turns.join("/"))}">${esc(turnLabel(c.turns))} · Doc ${esc(c.doc_number)} · ${esc(c.penalized)} penalized</button>`).join("");
  return `<div class="card choose" data-question="${esc(question)}"><h3>${esc(inc.event || "")}: which incident?</h3>
    <p>The question matches more than one ruling with the same drivers. Choose one.
      No claim about this incident is shown until you do.</p>
    <div class="choices">${buttons}</div><div class="chosen"></div></div>`;
}

function emptyCard(inc) {
  const rows = (inc.rejected || []).slice(0, 3).map((r) => `<li><code>${esc(r.id)}</code>: ${esc(r.reason)}</li>`).join("");
  return `<div class="card insufficient"><h3>${esc(inc.event || "No event matched")}: no eligible ruling</h3>
    <p>No ruling passes the scope checks for:<br><i>${esc(inc.query)}</i></p>
    ${rows ? `<p>Closest rulings, rejected:</p><ul>${rows}</ul>` : ""}</div>`;
}

async function choose(btn) {
  const card = btn.closest(".card");
  card.querySelectorAll(".choice").forEach((b) => b.classList.toggle("picked", b === btn));
  const docId = btn.dataset.doc, turn = btn.dataset.turn;
  const views = await getJSON(`/api/rulings/${encodeURIComponent(docId)}`);
  const q = `${card.dataset.question.replace(/\?\s*$/, "")} at Turn ${turn}?`;
  card.querySelector(".chosen").innerHTML = rulingBody(docId, views, searchRowById(docId)) +
    `<p class="muted">You chose Turn ${esc(turn)}. These excerpts come from the cached ruling. No model answer was made for this choice.</p>
     <button class="ask-turn">Ask the live model about Turn ${esc(turn)}</button>`;
  card.querySelector(".ask-turn").onclick = () => { $("question").value = q; $("run-live").click(); };
  const pending = $("claims").querySelector(".pending");
  if (pending) pending.innerHTML = `<b>You chose Turn ${esc(turn)}.</b> The ruling excerpts are shown above. ` +
    "The held model claims stay hidden, because the model chose without asking. Ask the live model for an answer about this turn.";
}

let LAST_VIEW = null;
const searchRowById = (docId) => (LAST_VIEW ? searchRow(LAST_VIEW, docId) : null);

function renderRulings(v, byId, fetched) {
  LAST_VIEW = v;
  const seen = new Set();
  const cards = [];
  for (const inc of v.incidents) {
    const key = inc.choices.length ? inc.choices.map((c) => c.id).join(",") : inc.top_doc || `none:${inc.event}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (inc.choices.length) { cards.push(choiceCard(inc, v.question)); continue; }
    if (!inc.top_doc) { cards.push(emptyCard(inc)); continue; }
    const mine = Object.values(byId).filter((p) => p.doc_id === inc.top_doc && fetched.has(p.evidence_id));
    cards.push(`<div class="card">${rulingBody(inc.top_doc, mine, searchRow(v, inc.top_doc))}</div>`);
  }
  $("rulings").innerHTML = cards.join("") || `<p class="muted">No search_decisions call in this run.</p>`;
  $("rulings").querySelectorAll(".choice").forEach((b) => { b.onclick = () => choose(b); });
}

function renderClaims(v, byId) {
  const badge = $("status-badge");
  badge.className = `badge ${v.status}`;
  badge.textContent = { answered: "answered", needs_choice: "choose an incident",
    partial: "partial: one side has no eligible ruling" }[v.status]
    || "insufficient evidence";
  const chip = (id) => {
    const p = byId[id];
    return p ? `<a class="eid" href="#ev-${esc(id)}" title="${esc(p.text)}">${esc(id)}</a> <a href="${esc(p.source_url)}" target="_blank" rel="noopener">p.${esc(p.page)}</a>`
      : `<span class="eid">${esc(id)}</span>`;
  };
  const isHeld = (c) => c.problems.some((p) => p.startsWith("the question matches more than one ruling"));
  const ok = v.claims.filter((c) => c.supported);
  const held = v.claims.filter(isHeld);
  const bad = v.claims.filter((c) => !c.supported && !isHeld(c));
  $("claims").innerHTML = ok.map((c) => `<li>${esc(c.text)} ${c.evidence.map(chip).join(" ")}</li>`).join("")
    || (v.status === "needs_choice"
      ? `<li class="pending"><b>Choose an incident above.</b> ${held.length} model claim(s) about it are held, because the model picked one ruling without asking.</li>`
      : `<li><b>Insufficient evidence.</b> No claim passed the citation check.</li>`);
  $("removed-wrap").classList.toggle("hidden", bad.length === 0);
  $("removed-count").textContent = bad.length;
  $("removed").innerHTML = bad.map((c) => `<li>${esc(c.text)} <span class="why">Removed: ${esc(c.problems.join("; "))}</span></li>`).join("");
}

function renderWhy(v, byId) {
  const excluded = {}, near = {}, rejected = {};
  let authority = null;
  for (const c of v.calls) {
    const o = c.output || {};
    if (c.name === "search_decisions") {
      (o.excluded_before_ranking || []).forEach((e) => { excluded[e.id] = e; });
      (o.near_misses || []).forEach((e) => { near[e.id] = e; });
    }
    if (c.name === "resolve_authority" && !o.error) {
      authority = o;
      (o.rejected || []).forEach((e) => { rejected[e.id] = e; });
    }
  }
  const oneMiss = Object.values(near).filter((n) => n.hard_failures === 1);
  const sameDrivers = oneMiss.filter((n) => !n.reason.startsWith("does not involve"));
  const nearList = sameDrivers.length ? sameDrivers : oneMiss.slice(0, 1);
  const row = (e, title) => ({ id: e.id, title, mode: e.failure_mode, reason: e.reason,
    looks: e.failure_mode === "different_incident" && e.reason.startsWith("does not involve")
      ? "Same event, and the text matches the question." : LOOKS_RIGHT[e.failure_mode] });
  const superseded = Object.values(excluded).map((e) => row(e, e.title));
  const guidelines = Object.values(rejected).filter((e) => e.source_type === "driving_guideline")
    .map((e) => row(e, e.title));
  const misses = nearList.map((e) => row(e, (SOURCES[e.id] || {}).title));
  const candidates = [...superseded, ...guidelines, ...misses];
  const pick = superseded[0] || misses[0] || guidelines[0];
  let html = "";
  if (pick) {
    html += `<div class="pick"><b>Most plausible rejected result:</b> <code>${esc(pick.id)}</code>, ${esc(pick.title)}<br>
      <span class="mode ${esc(pick.mode)}">${esc(MODE_TEXT[pick.mode] || pick.mode)}.</span> ${esc(pick.reason)}</div>`;
  }
  html += `<p>Three different ways a plausible source can be wrong for this incident. The checks run in code before ranking.</p>
    <table><tr><th>Candidate</th><th>Failure mode</th><th>Why it looks right</th><th>Why it was rejected</th></tr>
    ${candidates.map((c) => `<tr><td><code>${esc(c.id)}</code><br>${esc(c.title)}</td>
      <td class="mode ${esc(c.mode)}">${esc(MODE_TEXT[c.mode] || c.mode)}</td>
      <td>${esc(c.looks || "")}</td><td>${esc(c.reason)}</td></tr>`).join("")}</table>`;
  if (authority) {
    html += `<h3>Authority for ${esc(authority.event || "")} on ${esc(authority.operative_date)}</h3><table>
      <tr><th>Source</th><th>Verdict</th><th>Scope note</th></tr>
      ${authority.applicable.map((a) => `<tr><td><a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.title)}</a></td>
        <td>${esc(a.verdict.replaceAll("_", " "))}</td><td>${esc(a.scope_note || "")}</td></tr>`).join("")}
      ${(authority.unverified || []).map((a) => `<tr><td><a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.title)}</a></td>
        <td class="mode unverified">unverified edition: not citable</td><td>${esc(a.scope_note || "")}</td></tr>`).join("")}
      ${authority.rejected.map((a) => `<tr><td><a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.title)}</a></td>
        <td class="mode ${esc(a.failure_mode)}">rejected: ${esc(MODE_TEXT[a.failure_mode] || a.failure_mode)}</td><td>${esc(a.reason)}</td></tr>`).join("")}
      </table>`;
    for (const u of authority.unavailable || []) {
      html += `<h3>Not available: ${esc(u.title)}</h3><p>${esc(u.reason)} <a href="${esc(u.evidence_url)}" target="_blank" rel="noopener">FIA announcement</a></p>
        <p><b>What the ${esc(authority.event || "")} rulings themselves establish about the guidelines:</b></p>
        ${(u.what_rulings_establish || []).map((e) => byId[e.evidence_id] ? excerptHTML(byId[e.evidence_id]) : "").join("")
          || `<p class="muted">None of these rulings mention the guidelines.</p>`}`;
    }
  } else {
    html += `<p class="muted">resolve_authority was not called in this run, so no regulation or guideline check is shown.</p>`;
  }
  $("why-body").innerHTML = html;
}

function renderTrace(v) {
  $("budget").textContent = `${v.budget.executed} of ${v.budget.max_tool_calls} tool calls used` +
    (v.budget.refused ? `, ${v.budget.refused} refused` : "");
  $("trace-body").innerHTML = v.calls.map((c) => `<div class="call">
    <b>#${c.n} ${esc(c.name)}</b> · ${esc(c.status)} · ${esc(c.latency_ms)} ms
    ${c.identical === undefined ? "" : c.identical ? " · re-run identical to recording" : " · <b>re-run differs from recording</b>"}
    <pre>${esc(JSON.stringify(c.arguments, null, 2))}</pre>
    <details><summary>Output${v.mode === "replay" ? " (re-run now)" : ""}</summary><pre>${esc(JSON.stringify(c.output, null, 2))}</pre></details>
    ${c.identical === false ? `<details><summary>Recorded output</summary><pre>${esc(JSON.stringify(c.recorded_output, null, 2))}</pre></details>` : ""}
  </div>`).join("") || "<p>No tool calls.</p>";
  const feedback = (v.data.transcript || []).filter((t) => t.server_feedback);
  if (feedback.length) {
    $("trace-body").innerHTML += `<h3>Server checks sent to the model (${feedback.length})</h3>` +
      feedback.map((t) => `<p class="feedback">${esc(t.content)}</p>`).join("");
  }
}

function renderSources() {
  const rows = Object.values(SOURCES).map((s) => {
    const link = s.canonical_url || s.listing_url;
    const scope = s.event_key ? `${s.event_key}, ${s.document_date || ""}` : `season ${s.season}, valid ${s.valid_from} to ${s.valid_to || "open"}`;
    const status = s.status + (s.superseded_by ? ` → ${s.superseded_by}` : "");
    return `<tr><td><code>${esc(s.id)}</code></td><td>${esc(s.source_type.replaceAll("_", " "))}</td>
      <td><a href="${esc(link)}" target="_blank" rel="noopener">${esc(s.title)}</a></td><td>${esc(scope)}</td>
      <td>${esc(s.doc_number ?? "")}</td><td>${esc((s.turns || []).join("/"))}</td><td>${esc(status)}</td><td>${esc(s.page_count)}</td></tr>`;
  }).join("");
  $("sources-body").innerHTML = `<table><tr><th>ID</th><th>Type</th><th>Title (official link)</th><th>Scope</th>
    <th>Doc</th><th>Turn</th><th>Status</th><th>Pages</th></tr>${rows}</table>`;
}

init();
