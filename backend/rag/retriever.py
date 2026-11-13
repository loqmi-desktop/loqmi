"""
rag/retriever.py
Scoped retrieval with a relevance cutoff. Returns structured hits so callers
can build both the prompt context and the citation list.

Distance note: embeddings are normalised and Chroma's default space is
squared-L2, so cosine similarity = 1 - distance / 2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config import RETRIEVAL_MIN_SIMILARITY, RETRIEVAL_TOP_K
from rag.ingest import chat_store, kb_store

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    file: str
    page: Optional[int]      # None when pages are not meaningful (txt/md/image)
    text: str
    score: float             # cosine similarity, 0..1


def _search(store, query: str, k: int, where: Optional[dict],
            min_sim: Optional[float] = RETRIEVAL_MIN_SIMILARITY) -> list[Hit]:
    if not query.strip() or store._collection.count() == 0:
        return []
    pairs = store.similarity_search_with_score(query, k=k, filter=where)
    hits: list[Hit] = []
    for doc, dist in pairs:
        sim = 1.0 - float(dist) / 2.0
        if min_sim is not None and sim < min_sim:
            continue
        meta = doc.metadata
        hits.append(Hit(
            file=meta.get("source_file", "unknown"),
            page=meta.get("page") if meta.get("paged") else None,
            text=doc.page_content.strip(),
            score=sim,
        ))
    return hits


def retrieve_chat(query: str, chat_id: str, top_k: int = RETRIEVAL_TOP_K) -> list[Hit]:
    """
    Chunks from files attached to this chat only. NO similarity cutoff: the user attached
    these files on purpose, and generic requests ("summarize this") match no chunk strongly.
    Callers use Hit.score to decide whether the match is any good.
    """
    return _search(chat_store(), query, top_k, {"chat_id": chat_id}, min_sim=None)


def _spread(items: list, k: int) -> list:
    """k items spread evenly across the list, always including the first and last."""
    if k <= 0:
        return []
    if k >= len(items):
        return items
    if k == 1:
        return [items[0]]
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def sample_chat(chat_id: str, n_chunks: int) -> list[Hit]:
    """
    A reading-order sample from across every file attached to this chat. Used for broad
    requests (summaries, overviews) where no single passage is "the relevant one".
    """
    col = chat_store()._collection
    data = col.get(where={"chat_id": chat_id}, include=["documents", "metadatas"])
    by_doc: dict[str, list[tuple[int, Hit]]] = {}
    for cid, text, meta in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", [])):
        try:
            order = int(cid.rsplit("-", 1)[1])        # ids end with the chunk's index in its document
        except (IndexError, ValueError):
            order = 0
        hit = Hit(
            file=meta.get("source_file", "unknown"),
            page=meta.get("page") if meta.get("paged") else None,
            text=(text or "").strip(),
            score=0.0,
        )
        by_doc.setdefault(meta.get("doc_id", ""), []).append((order, hit))

    if not by_doc:
        return []
    quota = max(2, n_chunks // len(by_doc))
    out: list[Hit] = []
    for chunks in by_doc.values():
        ordered = [h for _, h in sorted(chunks, key=lambda t: t[0])]
        out.extend(_spread(ordered, quota))
    return out


def retrieve_kb(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Hit]:
    """Chunks from the global knowledge base."""
    return _search(kb_store(), query, top_k, None)


def format_hits(hits: list[Hit]) -> str:
    """Prompt text. Each excerpt is labelled so the model can cite file + page."""
    parts = []
    for h in hits:
        label = f"{h.file}, page {h.page}" if h.page else h.file
        parts.append(f"[{label}]\n{h.text}")
    return "\n\n---\n\n".join(parts)


def dedupe_sources(hits: list[Hit], scope: str) -> list[dict]:
    seen, out = set(), []
    for h in hits:
        key = (h.file, h.page)
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": h.file, "page": h.page, "scope": scope})
    return out


# Backwards-compatible helper for main.py / old callers.
def retrieve_context(query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
    return format_hits(retrieve_kb(query, top_k))