"""
core/doc_context.py
Decides, per turn, how attached files (and optionally the knowledge base)
reach the model:

  whole      every ready attachment fits in DOC_WHOLE_FRACTION of the prompt
             budget -> inject full text with page markers (most accurate)
  retrieval  otherwise -> top-k chunks from this chat's attachments
  (KB chunks are added on top when RAG is enabled for the chat)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from llama_cpp import Llama

from config import CHUNK_SIZE, DOC_WHOLE_FRACTION
from core import context_budget, memory
from rag.ingest import load_pages
from rag.retriever import dedupe_sources, format_hits, retrieve_chat, retrieve_kb, sample_chat

logger = logging.getLogger(__name__)


@dataclass
class DocContext:
    text: str = ""
    sources: list[dict] = field(default_factory=list)
    mode: str = "none"        # none | whole | sampled | retrieval


# Requests about the document as a whole: no single passage is "the relevant one".
_BROAD_RE = re.compile(
    r"\b(summar\w*|overview|tl;?dr|teach me|walk me through|main (points?|ideas?)|key (points?|takeaways?)"
    r"|in short|briefly"
    r"|explain (this|the) (paper|document|file|pdf|article))\b",
    re.IGNORECASE,
)
# "what is this about?" is broad; "what does the paper say about X" is a specific question.
_BROAD_END_RE = re.compile(
    r"what('s| is| does) (this|it|the (paper|document|file|pdf)) (about|say|cover)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_WEAK_MATCH = 0.35      # best cosine similarity below this = the question doesn't point at any passage
_CHARS_PER_TOKEN = 3.5  # pessimistic, so sampled text stays inside the budget


def _is_broad(query: str) -> bool:
    return bool(_BROAD_RE.search(query) or _BROAD_END_RE.search(query.strip()))


def _whole_text(docs: list[dict]) -> str:
    parts = []
    for d in docs:
        pdoc = load_pages(d["doc_id"])
        if pdoc is None:
            return ""          # cache missing -> force retrieval
        body = []
        for i, page in enumerate(pdoc.pages, start=1):
            if page.strip():
                body.append(f"[Page {i}]\n{page}" if pdoc.paged else page)
        parts.append(f"=== File: {d['filename']} ===\n" + "\n\n".join(body))
    return "\n\n".join(parts)


def build(
    model: Llama,
    chat_id: str,
    query: str,
    available_tokens: int,
    include_kb: bool,
) -> DocContext:
    sections: list[str] = []
    sources: list[dict] = []
    mode = "none"

    docs = [d for d in memory.list_chat_documents(chat_id) if d["status"] == "ready"]
    if docs:
        budget = int(available_tokens * DOC_WHOLE_FRACTION)
        total_chars = sum(d["chars"] for d in docs)
        whole = ""
        # ~3 chars/token is a pessimistic pre-check that avoids tokenizing huge texts.
        if total_chars <= budget * 3:
            candidate = _whole_text(docs)
            if candidate and context_budget.count_tokens(model, candidate) <= budget:
                whole = candidate

        if whole:
            sections.append("ATTACHED FILES\n" + whole)
            sources += [{"file": d["filename"], "page": None, "scope": "chat"} for d in docs]
            mode = "whole"
        else:
            hits = retrieve_chat(query, chat_id)
            best = max((h.score for h in hits), default=0.0)
            logger.info("Attached-file retrieval: %d hits, best similarity %.2f", len(hits), best)

            if _is_broad(query) or best < _WEAK_MATCH:
                # Overview request (or a question that matches nothing strongly): the file did not
                # fit whole, so show the model an even sample from start to end, in reading order.
                n = max(4, int(budget * _CHARS_PER_TOKEN / CHUNK_SIZE))
                sample = sample_chat(chat_id, n)
                if sample:
                    sections.append(
                        "ATTACHED FILES (the full text did not fit, so these are passages sampled "
                        "from across the file, in order. Base your answer on them and say if "
                        "something may be missing.)\n" + format_hits(sample)
                    )
                    sources += dedupe_sources(sample, "chat")
                    mode = "sampled"
            elif hits:
                sections.append("ATTACHED FILES (relevant excerpts)\n" + format_hits(hits))
                sources += dedupe_sources(hits, "chat")
                mode = "retrieval"

    if include_kb:
        kb_hits = retrieve_kb(query)
        if kb_hits:
            sections.append("KNOWLEDGE BASE (relevant excerpts)\n" + format_hits(kb_hits))
            sources += dedupe_sources(kb_hits, "kb")
            mode = mode if mode != "none" else "retrieval"

    logger.info("Doc context: mode=%s sources=%d", mode, len(sources))
    return DocContext(text="\n\n".join(sections), sources=sources, mode=mode)