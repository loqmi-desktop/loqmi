"""
api/routes/chats.py
Chat and message management routes.

POST   /chats                        — create a new chat
GET    /chats                        — list all chats (sidebar)
DELETE /chats/{chat_id}              — delete a chat, its messages and its attached-file vectors
GET    /chats/{chat_id}/messages     — load full message history
POST   /chats/{chat_id}/messages     — send a message, stream response via SSE
PATCH  /chats/{chat_id}/rag          — toggle RAG on/off for this chat
PATCH  /chats/{chat_id}/search       — toggle web search and choose the provider

Chat-scoped file attachments live in api/routes/chat_documents.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import CLASSIFIER_CATEGORIES
from core import memory
from core import router as core_router
from rag import purge_chat

logger = logging.getLogger(__name__)

router_obj = APIRouter(prefix="/chats", tags=["chats"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateChatResponse(BaseModel):
    chat_id: str


class MessageRequest(BaseModel):
    # A list of content blocks, e.g.
    #   [{"label": "text",  "content": "what's in this?"},
    #    {"label": "image", "msg": "...", "svg": "...", "attachment_id": "..."},
    #    {"label": "file",  "doc_id": "...", "filename": "report.pdf", "pages": 12}]
    # Stored as-is (JSON-encoded) via memory.append. NEVER used directly
    # for inference — see inference_override_text below. Loosely typed as
    # List[Dict[str, Any]] rather than a strict discriminated union since
    # the shape is owned by the frontend's ContentBlock type; tighten this
    # with a proper Pydantic union later if it drifts.
    content: List[Dict[str, Any]]

    # Ephemeral text used ONLY to build THIS turn's prompt — e.g. the
    # user's typed text plus a screenshot's raw OCR text. Never persisted.
    # If omitted, falls back to the flattened text blocks of `content` (so
    # plain text-only messages still work without the frontend needing to
    # pass this explicitly).
    inference_override_text: Optional[str] = None

    # One of CLASSIFIER_CATEGORIES, or None to let the classifier decide.
    override_category: Optional[str] = None


class RagToggleRequest(BaseModel):
    enabled: bool

class SearchToggleRequest(BaseModel):
    enabled: bool
    method: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router_obj.post("", response_model=CreateChatResponse)
def create_chat():
    chat_id = memory.create_chat()
    return {"chat_id": chat_id}


@router_obj.get("")
def list_chats():
    return memory.list_chats()


@router_obj.delete("/{chat_id}", status_code=204)
def delete_chat(chat_id: str):
    _assert_chat_exists(chat_id)
    memory.delete_chat(chat_id)   # messages and chat_documents rows cascade
    try:
        purge_chat(chat_id)       # remove this chat's attached-file vectors
    except Exception:
        # The chat is already gone; orphaned vectors are harmless, so don't fail the request.
        logger.warning("Could not purge vectors for chat %s", chat_id, exc_info=True)


@router_obj.get("/{chat_id}/messages")
def get_messages(chat_id: str):
    _assert_chat_exists(chat_id)
    history = memory.get_history(chat_id)
    # Parse each row's stored content string into its block array so the
    # frontend receives real ContentBlock[] JSON, not a JSON-encoded
    # string it would have to parse itself. Old plain-text rows come back
    # as a single implicit text block — see memory.parse_content_blocks.
    for msg in history:
        msg["content"] = memory.parse_content_blocks(msg["content"])
    return history


@router_obj.post("/{chat_id}/messages")
async def send_message(chat_id: str, body: MessageRequest):
    """
    Send a user message and stream the assistant's response via SSE.
    Each SSE event is a JSON object:
      {"type": "token",   "text": "..."}       — one token
      {"type": "status",  "message": "..."}    — progress indicator
      {"type": "sources", "items": [...]}      — files / pages the answer used
      {"type": "context", ...}                 — history/prompt was trimmed to fit
      {"type": "stats",   "category": ..., "classify_ms": ..., "inference_s": ...}
      {"type": "error",   "message": "..."}
    """
    _assert_chat_exists(chat_id)

    if body.override_category and body.override_category not in CLASSIFIER_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown override_category '{body.override_category}'. "
                f"Valid values: {', '.join(CLASSIFIER_CATEGORIES)}."
            ),
        )

    # What actually gets stored in chats.db for this turn — compact,
    # forever. json.dumps here matches what memory.parse_content_blocks
    # expects to read back on GET /chats/{id}/messages.
    persist_content = json.dumps(body.content)

    # What the model sees for THIS turn only — never persisted. Falls
    # back to just the typed text blocks if the frontend didn't send an
    # explicit override.
    prompt = body.inference_override_text
    if not prompt:
        prompt = "\n\n".join(
            b.get("content", "") for b in body.content if b.get("label") == "text"
        ).strip()

    def event_stream():
        try:
            # persist_content flows through to inference.run(), which is
            # what actually calls memory.append() — see core/inference.py.
            for chunk in core_router.handle(
                prompt=prompt,
                persist_content=persist_content,
                chat_id=chat_id,
                override_category=body.override_category,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # prevents nginx from buffering SSE
        },
    )


@router_obj.patch("/{chat_id}/rag", status_code=204)
def toggle_rag(chat_id: str, body: RagToggleRequest):
    _assert_chat_exists(chat_id)
    memory.set_rag_enabled(chat_id, body.enabled)

@router_obj.patch("/{chat_id}/search", status_code=204)
def toggle_search(chat_id: str, body: SearchToggleRequest):
    _assert_chat_exists(chat_id)
    memory.set_search_settings(chat_id, body.enabled, body.method)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_chat_exists(chat_id: str):
    if not memory.chat_exists(chat_id):
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")


# FastAPI router alias (server.py imports `router`)
router = router_obj