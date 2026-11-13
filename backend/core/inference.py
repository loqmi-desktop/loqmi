"""
core/inference.py
Runs generation for a single turn.

With Internet on, a turn has two phases:
  1. research: plan queries, search, fetch, rank and write notes. Done by a
     small helper model (or the chat model as fallback); see core/research.py
     and model_manager.get_research_handle().
  2. answer: the routed model answers from the notes. It never sees tool
     syntax or raw web pages.
"""
from __future__ import annotations

import logging
from datetime import date

from llama_cpp import Llama

from config import (
    INFERENCE_MAX_TOKENS,
    INFERENCE_TEMPERATURE,
    OUTPUT_TOKEN_CAPS,
    RESEARCH_APPEND_SOURCES,
    RESEARCH_NOTES_IN_CONTEXT_MAX,
)
from core import context_budget, memory, model_manager, research, doc_context
from rag import retrieve_context

logger = logging.getLogger(__name__)

_SYSTEM_PROMPTS = {
    "chat":        "You are a helpful assistant.",
    "code":        "You are an expert programmer. Provide clean, well-commented code.",
    "reasoning":   "You are an expert analyst. Think step-by-step and reason carefully before answering.",
    "document":    "You are a document analysis assistant. Extract, summarise, and explain information from the provided context accurately.",
}

_NOTES_INSTRUCTIONS = """Today's date is {today}. The system gathered the research notes below from the internet for the user's question; the user did not write them. Base your answer on them and cite sources as [n] where useful. If the notes say information was missing or unclear, say so plainly instead of guessing. Do not mention tools, searching or these instructions.
--- RESEARCH NOTES ---
{notes}
--- END NOTES ---"""


def _build_system(base: str, rag_context: str, notes: str) -> str:
    s = base
    if rag_context:
        s += (
            "\n\nThe user's own files are provided below. Answer using them. "
            "Excerpts are labelled with the file name and page; when you rely on one, cite it "
            "like (file name, p. N). If the files do not contain the answer, say so instead of guessing.\n\n"
            f"--- START FILES ---\n{rag_context}\n--- END FILES ---"
        )
    if notes:
        s += "\n\n" + _NOTES_INSTRUCTIONS.format(today=date.today().strftime("%A, %d %B %Y"), notes=notes)
    return s


def run(prompt: str, persist_content: str, model: Llama, category: str, chat_id: str):
    base_system = _SYSTEM_PROMPTS.get(category, "You are a helpful assistant.")

    rag_enabled = memory.get_rag_enabled(chat_id)
    search_enabled = memory.get_search_settings(chat_id)["enabled"]

    # Phase 1: research (before anything is persisted)
    notes, sources = "", []
    if search_enabled:
        for ev in _research_phase(model, category, prompt, chat_id):
            if ev["type"] == "_done":
                notes, sources = ev["notes"], ev["sources"]
                if ev["restore"]:
                    # The main model was swapped out for research; `model` is now closed.
                    yield {"type": "status", "message": "Switching back to the main model…"}
                    model = model_manager.restore_main(category)
            else:
                yield ev

    # Phase 2: fit and answer
    max_out = OUTPUT_TOKEN_CAPS.get(category, INFERENCE_MAX_TOKENS)
    with model_manager.acquire_for_generation():
        if notes:
            avail, _ = context_budget.compute_budget(model.n_ctx(), max_out)
            cap = max(150, min(RESEARCH_NOTES_IN_CONTEXT_MAX, int(avail * 0.25)))
            notes = context_budget.clip_to_tokens(model, notes, cap, keep_tail_fraction=0.0)

        avail, _ = context_budget.compute_budget(model.n_ctx(), max_out)
        ctx = doc_context.build(
            model, chat_id, _retrieval_query(chat_id, persist_content, prompt), avail, include_kb=rag_enabled
        )
        rag_context = ctx.text
        fit = context_budget.fit_turn(
            model,
            build_system=lambda rag: _build_system(base_system, rag, notes),
            rag_context=rag_context,
            history=memory.get_history_for_inference(chat_id),
            prompt=prompt,
            typed_text=_typed_text(persist_content),
            max_output_tokens=max_out,
        )
    messages, max_tokens = fit.messages, fit.max_tokens

    first_turn = memory.count_messages(chat_id) == 0
    memory.append(chat_id, "user", persist_content)
    if first_turn:
        memory.update_title(chat_id, _title_text(persist_content))

    full_reply = ""
    try:
        ctx_event = fit.event()
        if ctx.sources:
            yield {"type": "sources", "items": ctx.sources, "mode": ctx.mode}
        if ctx_event:
            yield ctx_event

        for token_text in _stream_plain(model, messages, max_tokens):
            full_reply += token_text
            yield {"type": "token", "text": token_text}

        if sources and RESEARCH_APPEND_SOURCES and full_reply.strip():
            block = "\n\n---\n**Sources**\n" + "\n".join(
                f"{s['n']}. [{_md_safe(s['title'])}]({s['url']})" for s in sources
            )
            full_reply += block
            yield {"type": "token", "text": block}
    except Exception:
        if not full_reply:
            memory.delete_last_message(chat_id, "user")
        raise
    finally:
        if full_reply:
            memory.append(chat_id, "assistant", full_reply)


def _research_phase(model: Llama, category: str, prompt: str, chat_id: str):
    """
    Yields status events, then {"type": "_done", "notes", "sources", "restore"}.
    Never raises: a failure becomes a note telling the answering model the search failed.

    Order matters: a non-reasoning main model plans BEFORE any helper is
    loaded, because the "swapped" fallback closes the main model.
    """
    search = memory.get_search_settings(chat_id)
    method = search["method"] or "duckduckgo"
    brave_key = memory.get_search_key("brave")
    tavily_key = memory.get_search_key("tavily")
    last_answer = _last_assistant(chat_id)

    notes, sources, handle = "", [], None
    try:
        if category != "reasoning":
            yield {"type": "status", "message": "Checking whether a search is needed…"}
            needs, queries = research.plan_queries(
                model, model_manager.acquire_for_generation(), prompt, last_answer)
            if needs:
                handle = model_manager.get_research_handle(category)
        else:
            # Reasoning models think at length even for a tiny JSON plan; let the helper plan.
            yield {"type": "status", "message": "Checking whether a search is needed…"}
            handle = model_manager.get_research_handle(category)
            needs, queries = research.plan_queries(handle.model, handle.lock, prompt, last_answer)

        if needs and queries and handle is not None:
            logger.info("Research via '%s' model", handle.kind)
            for ev in research.run_research(handle, queries, prompt, method, brave_key, tavily_key):
                if ev["type"] == "_research_done":
                    notes, sources = ev["result"].notes, ev["result"].sources
                else:
                    yield ev
    except Exception as e:
        logger.warning("Research failed", exc_info=True)
        notes = (f"The web search failed ({e}). Tell the user you could not reach the internet, "
                 "then answer from general knowledge and mark it as possibly out of date.")
        sources = []

    restore = (handle is not None and handle.kind == "swapped") or model_manager.current_model_id() is None
    yield {"type": "_done", "notes": notes, "sources": sources, "restore": restore}


def _last_assistant(chat_id: str) -> str:
    for m in reversed(memory.get_history_for_inference(chat_id)):
        if m["role"] == "assistant":
            return m["content"]
    return ""


def _md_safe(text: str) -> str:
    return text.replace("[", "(").replace("]", ")").replace("\n", " ")[:80]


def _title_text(persist_content: str) -> str:
    blocks = memory.parse_content_blocks(persist_content)
    for block in blocks:
        if block.get("label") == "text" and block.get("content", "").strip():
            return block["content"].strip()
    return "Screenshot"


def _typed_text(persist_content: str) -> str:
    """The text the user actually typed (text blocks only, no OCR / file text)."""
    blocks = memory.parse_content_blocks(persist_content)
    return "\n\n".join(
        b["content"].strip() for b in blocks
        if b.get("label") == "text" and b.get("content", "").strip()
    )


def _stream_plain(model: Llama, messages: list[dict], max_tokens: int):
    with model_manager.acquire_for_generation():
        stream = model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=INFERENCE_TEMPERATURE,
            stream=True,
        )
        for chunk in stream:
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                yield token

def _retrieval_query(chat_id: str, persist_content: str, prompt: str) -> str:
    """Typed text (not OCR). Very short follow-ups borrow the previous user message."""
    q = _typed_text(persist_content) or prompt[:500]
    if len(q.split()) <= 6:
        for m in reversed(memory.get_history_for_inference(chat_id)):
            if m["role"] == "user":
                q = f"{m['content'][:300]} {q}"
                break
    return q

def _title_text(persist_content: str) -> str:
    blocks = memory.parse_content_blocks(persist_content)
    for block in blocks:
        if block.get("label") == "text" and block.get("content", "").strip():
            return block["content"].strip()
    for block in blocks:
        if block.get("label") == "file" and block.get("filename"):
            return block["filename"]
    return "Screenshot"