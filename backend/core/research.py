"""
core/research.py
Deterministic web-research pipeline. There is no ReAct loop. The LLM is asked
for (1) a JSON plan and (2) concise notes; searching, fetching, extraction
and ranking are plain code, which small local models handle far more reliably.
"""
from __future__ import annotations

import json
import logging
import math
import time
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from itertools import zip_longest
from urllib.parse import urlparse

from config import (
    RESEARCH_CHUNK_CHARS, RESEARCH_CHUNKS_KEPT, RESEARCH_MAX_CHUNKS_PER_URL,
    RESEARCH_MAX_PAGES, RESEARCH_MAX_QUERIES, RESEARCH_NOTES_MAX_TOKENS,
    RESEARCH_PAGE_MAX_CHARS,
    RESEARCH_ENOUGH_URLS,
    RESEARCH_QUERY_DELAY_S,
    USER_REGION_NAME
)
from core import context_budget, web_search

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_STOP = set("""a an the of to in on for and or is are was were be been it this that with as at by from
what which who how why when where do does did can could should would will me my i you your about
best top get find give tell please""".split())


@dataclass
class ResearchResult:
    notes: str = ""
    sources: list[dict] = field(default_factory=list)   # {"n", "title", "url"}
    queries: list[str] = field(default_factory=list)


# ── 1. Plan ───────────────────────────────────────────────────────────────────

_PLAN_SYSTEM = """You decide whether a question needs the live internet and, if so, write search queries.
Today's date is {today}. The user is in {region}.
Reply with ONLY a JSON object and nothing else:
{{"needs_search": true, "queries": ["...", "..."]}}

Rules:
- needs_search is true for anything current or changing: prices, exchange rates, news, sports, weather, releases, sales, availability, "latest", "today", "current", or facts you are unsure about.
- needs_search is false for casual chat, opinions, math, coding help, explanations of stable concepts, or questions about text the user pasted.
- Write 1 to {max_q} short queries (3 to 8 words each). Each query must cover a DIFFERENT angle, for example one for the event and its dates, one for specific products and prices.
- For shopping, sales and prices, name {region} or its currency in the query so results are local.
- If the question is a short follow-up like "convert that to rupees", use the previous answer to write complete queries."""

_NOTES_SYSTEM = """You are a research assistant. You get numbered passages from web pages and a question.
Write concise research notes that answer the question using ONLY the passages.
- Short bullet points with the key facts, numbers, dates and names.
- After each fact, add its source number like [2].
- Copy numbers exactly. Always give the currency and, for events or sales, the event name and its dates.
- Compare event dates with today's date ({today}) and say whether each event is upcoming, ongoing or already over.
- If passages disagree, or cover different regions or currencies, report each separately with its source.
- If the passages do not contain the answer, say exactly what is missing. Never guess or fill in from memory.
No introduction, no conclusion, no advice."""


def plan_queries(model, lock, prompt: str, previous_answer: str = "") -> tuple[bool, list[str]]:
    system = _PLAN_SYSTEM.format(today=date.today().strftime("%A, %d %B %Y"),max_q=RESEARCH_MAX_QUERIES, region=USER_REGION_NAME)
    user = f"Question: {_clip(prompt, 1200)}"
    prev = _THINK_RE.sub("", previous_answer).strip()
    if prev:
        user = f"Previous answer (context only): {_clip(prev, 600)}\n\n{user}"
    try:
        raw = _complete(model, lock,
                        [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_tokens=160, temperature=0.1)
    except Exception:
        logger.warning("Planning call failed", exc_info=True)
        return True, [_fallback_query(prompt)]

    plan = _parse_json(raw)
    if plan is None:
        logger.info("Plan was not valid JSON (%r); searching with the question itself", raw[:100])
        return True, [_fallback_query(prompt)]

    needs = bool(plan.get("needs_search", True))
    qs = plan.get("queries", [])
    queries = [q.strip() for q in qs if isinstance(q, str) and q.strip()][:RESEARCH_MAX_QUERIES] if isinstance(qs, list) else []
    if needs and not queries:
        queries = [_fallback_query(prompt)]
    logger.info("Plan: needs_search=%s queries=%s", needs, queries)
    return needs, queries


# ── 2-4. Search, fetch, rank, notes ───────────────────────────────────────────

_NOTES_SYSTEM = """You are a research assistant. You get numbered passages from web pages and a question.
Write concise research notes that answer the question using ONLY the passages.
- Short bullet points with the key facts, numbers, dates and names.
- After each fact, add its source number like [2].
- Copy numbers exactly. State the currency or unit and the date if the passage gives one.
- If passages disagree, report both with their sources.
- If the passages do not contain the answer, say exactly what is missing. Never guess or fill in from memory.
No introduction, no conclusion, no advice. Today's date is {today}."""


def run_research(handle, queries: list[str], question: str, method: str,
                 brave_key: str | None, tavily_key: str | None):
    """Yields status events, then {"type": "_research_done", "result": ResearchResult}."""
    per_query: list[list[dict]] = []
    errors: list[str] = []

    for i, q in enumerate(queries):
        if i and sum(len(g) for g in per_query) >= RESEARCH_ENOUGH_URLS:
            break                                   # enough URLs already; fewer hits means fewer rate limits
        if i:
            time.sleep(RESEARCH_QUERY_DELAY_S)
        yield {"type": "status", "message": f"Searching: {q}"}
    # ... rest of the loop unchanged

    for q in queries:
        yield {"type": "status", "message": f"Searching: {q}"}
        try:
            results = web_search.search(q, method=method, brave_key=brave_key, tavily_key=tavily_key)
        except Exception as e:
            errors.append(str(e))
            continue
        good = []
        for r in results:
            if (r.get("url") or "").startswith("http"):
                good.append(r)
            elif r.get("title") == "Search failed":
                errors.append(r.get("snippet", ""))
        per_query.append(good)

    ordered, seen = [], set()
    for group in zip_longest(*per_query):           # round-robin so every query contributes
        for r in group:
            if r is None:
                continue
            key = r["url"].split("#")[0]
            if key not in seen:
                seen.add(key)
                ordered.append(r)

    if not ordered:
        why = errors[0] if errors else "the search returned no results"
        yield _done(ResearchResult(
            notes=f"The web search returned nothing usable ({why}). Say you could not find "
                  "current information; do not invent any.", queries=queries))
        return

    top = ordered[:RESEARCH_MAX_PAGES]
    yield {"type": "status", "message": f"Reading {len(top)} pages"}
    with ThreadPoolExecutor(max_workers=len(top)) as pool:
        texts = list(pool.map(lambda r: web_search.fetch_text(r["url"], RESEARCH_PAGE_MAX_CHARS), top))
    logger.info("Fetched %d/%d pages", sum(1 for t in texts if t), len(top))

    passages: list[tuple[dict, str]] = []
    for r, text in zip(top, texts):
        for c in _chunk(text or ""):
            passages.append((r, c))
    for r in ordered:                                # snippets cover pages that failed to load
        if r.get("snippet"):
            passages.append((r, r["snippet"]))

    picked: list[tuple[dict, str]] = []
    per_url: Counter = Counter()
    for score, r, text in _rank(passages, " ".join(queries) + " " + _clip(question, 300)):
        if score <= 0 or per_url[r["url"]] >= RESEARCH_MAX_CHUNKS_PER_URL:
            continue
        picked.append((r, text))
        per_url[r["url"]] += 1
        if len(picked) >= RESEARCH_CHUNKS_KEPT:
            break
    if not picked:
        picked = [(r, r["snippet"]) for r in ordered[:4] if r.get("snippet")]
    if not picked:
        yield _done(ResearchResult(
            notes="Pages were found but none had readable text. Say you could not read current "
                  "information; do not invent any.", queries=queries))
        return

    index: dict[str, int] = {}
    sources: list[dict] = []
    for r, _ in picked:
        if r["url"] not in index:
            index[r["url"]] = len(index) + 1
            sources.append({"n": index[r["url"]], "title": r.get("title") or _domain(r["url"]), "url": r["url"]})
    passages_text = "\n\n".join(f"[{index[r['url']]}] ({_domain(r['url'])}) {t}" for r, t in picked)

    yield {"type": "status", "message": "Summarizing findings"}
    try:
        with handle.lock:
            budget = max(300, handle.model.n_ctx() - RESEARCH_NOTES_MAX_TOKENS - 500)
            passages_text = context_budget.clip_to_tokens(handle.model, passages_text, budget, keep_tail_fraction=0.0)
            notes = _complete(
                handle.model, handle.lock,
                [{"role": "system", "content": _NOTES_SYSTEM.format(today=date.today().strftime("%A, %d %B %Y"))},
                 {"role": "user", "content": f"Question: {_clip(question, 600)}\n\nPassages:\n{passages_text}"}],
                max_tokens=RESEARCH_NOTES_MAX_TOKENS, temperature=0.2)
    except Exception:
        logger.warning("Notes generation failed; using raw passages", exc_info=True)
        notes = ""
    if not notes:
        notes = passages_text                        # still useful, and still numbered

    yield _done(ResearchResult(notes=notes, sources=sources, queries=queries))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _done(result: ResearchResult) -> dict:
    return {"type": "_research_done", "result": result}


def _complete(model, lock, messages, max_tokens: int, temperature: float) -> str:
    with lock:
        out = model.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature, stream=False)
    return _THINK_RE.sub("", out["choices"][0]["message"]["content"] or "").strip()


def _parse_json(text: str) -> dict | None:
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


def _fallback_query(prompt: str) -> str:
    return _clip(prompt, 160)


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def _chunk(text: str, size: int = RESEARCH_CHUNK_CHARS) -> list[str]:
    chunks, cur = [], ""
    for p in (p.strip() for p in text.split("\n")):
        if not p:
            continue
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}".strip()
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size:]
        cur = p
    if cur:
        chunks.append(cur)
    return chunks


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _rank(passages: list[tuple[dict, str]], query: str) -> list[tuple[float, dict, str]]:
    """BM25. Pure Python, no extra dependency. An embedding rerank can replace this later."""
    q_terms = {t for t in _tokens(query) if t not in _STOP}
    if not passages or not q_terms:
        return [(0.0, r, t) for r, t in passages]
    docs = [Counter(_tokens(t)) for _, t in passages]
    lens = [sum(d.values()) or 1 for d in docs]
    avg, n = sum(lens) / len(lens), len(docs)
    df = {t: sum(1 for d in docs if t in d) for t in q_terms}
    k1, b = 1.5, 0.75
    scored = []
    for (r, text), d, length in zip(passages, docs, lens):
        s = 0.0
        for t in q_terms:
            tf = d.get(t, 0)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
        scored.append((s, r, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored