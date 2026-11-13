"""
api/routes/attachments.py — Screenshot/image attachment handling.

Distinct from /documents/ingest: screenshots do NOT go into the RAG
knowledge base. Instead, each screenshot gets OCR'd and summarized by the
LLM, and the client receives a compact block:

    { attachment_id, msg, svg, raw_text }

- `msg`       — LLM-generated summary of the OCR text (1-2 sentences).
                This is what the client should persist in chat history —
                it's what gets replayed into every future turn.
- `svg`       — base64 WEBP thumbnail (~400px wide) for inline display.
                Named `svg` for schema consistency with the frontend's
                content blocks; it's a raster thumbnail, not vector art.
- `raw_text`  — full OCR text. Returned ONLY so the caller can fold it
                into the CURRENT turn's prompt for a detailed first
                answer. This endpoint never writes raw_text to disk or
                any database — callers must not persist it either.

The full-resolution image is saved to disk (APP_DATA_DIR/attachments/)
so it can be viewed again later; only the thumbnail lives inline in chat
history via `svg`.
"""

from __future__ import annotations

import asyncio
import base64
import io
import uuid

import pytesseract
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from config import APP_DATA_DIR
from core import model_manager
from rag.ingest import _ocr_image

router = APIRouter(prefix="/attachments", tags=["attachments"])

ATTACHMENTS_DIR = APP_DATA_DIR / "attachments"
ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)

THUMBNAIL_MAX_WIDTH = 400
# Cap OCR text handed to the summarizer so a dense screenshot (a long code
# file, a wall of terminal output) can't overflow the summarizer's own
# context window before the main response even starts.
MAX_OCR_CHARS_FOR_SUMMARY = 4000
SUMMARY_MAX_TOKENS = 120

SUMMARY_SYSTEM_PROMPT = (
    "You summarize screenshot text for a chat assistant's long-term memory. "
    "Given raw OCR'd text from a screenshot, write a 1-2 sentence summary "
    "capturing only the key facts a future conversation might need to "
    "recall. No preamble, no commentary, no markdown — just the summary."
)


class ScreenshotResponse(BaseModel):
    attachment_id: str
    msg: str
    svg: str           # base64 data URI thumbnail
    raw_text: str       # ephemeral — caller must not persist this field


@router.post("/screenshot", response_model=ScreenshotResponse)
async def upload_screenshot(file: UploadFile = File(...)) -> ScreenshotResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    raw_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except Exception:
        raise HTTPException(400, "Could not read image file")

    attachment_id = uuid.uuid4().hex

    # ── 1. Save the full-resolution original to disk ──────────────────────
    full_path = ATTACHMENTS_DIR / f"{attachment_id}.png"
    image.convert("RGB").save(full_path, format="PNG")

    # ── 2. Generate a small inline thumbnail ───────────────────────────────
    thumb = image.copy()
    thumb.thumbnail((THUMBNAIL_MAX_WIDTH, THUMBNAIL_MAX_WIDTH * 4))  # cap width, let height flow
    thumb_buffer = io.BytesIO()
    thumb.convert("RGB").save(thumb_buffer, format="WEBP", quality=80)
    thumb_b64 = base64.b64encode(thumb_buffer.getvalue()).decode("ascii")
    svg = f"data:image/webp;base64,{thumb_b64}"

    # ── 3. OCR — CPU-bound and blocking; off the event loop ────────────────
    raw_text = (await asyncio.to_thread(_ocr_image, image)).strip()

    # ── 4. Summarize — sync, before responding (per product decision) ──────
    summary_source = raw_text[:MAX_OCR_CHARS_FOR_SUMMARY]
    msg = (
        await _summarize(summary_source)
        if summary_source
        else "Screenshot (no readable text detected)"
    )

    return ScreenshotResponse(
        attachment_id=attachment_id,
        msg=msg,
        svg=svg,
        raw_text=raw_text,
    )


@router.get("/{attachment_id}")
async def get_attachment(attachment_id: str) -> FileResponse:
    """Serve the full-resolution original for a previously uploaded screenshot."""
    path = ATTACHMENTS_DIR / f"{attachment_id}.png"
    if not path.exists():
        raise HTTPException(404, "Attachment not found")
    return FileResponse(path, media_type="image/png")


async def _summarize(text: str) -> str:
    """
    Runs a single, non-streaming completion to summarize OCR'd screenshot
    text. Reuses whatever model category is already loaded when possible
    (model_manager keeps exactly one model in RAM — forcing a fixed
    category here would evict the active chat model, then evict the
    summarizer's model right back out again for the actual reply that
    follows). Falls back to "document" — whose system prompt is already
    tuned for extraction/summarization — if nothing is loaded yet.

    Wrapped in model_manager.acquire_for_generation() so this can't race
    a concurrently streaming chat response on the same Llama instance.
    Runs in a thread since llama-cpp-python's call is blocking/CPU-bound.
    """

    def _run() -> str:
        category = model_manager.current_category() or "document"
        model = model_manager.load(category)
        with model_manager.acquire_for_generation():
            result = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=SUMMARY_MAX_TOKENS,
                temperature=0.3,
                stream=False,
            )
        return result["choices"][0]["message"]["content"].strip()

    return await asyncio.to_thread(_run)