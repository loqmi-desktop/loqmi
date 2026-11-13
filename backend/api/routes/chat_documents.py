"""
api/routes/chat_documents.py
Chat-scoped file attachments (PDF, TXT, MD).

POST   /chats/{chat_id}/documents                    upload; returns at once, processing runs in the background
GET    /chats/{chat_id}/documents                    list attachments of this chat
GET    /chats/{chat_id}/documents/{doc_id}           status (poll this; stage text lives in `detail`)
DELETE /chats/{chat_id}/documents/{doc_id}           detach (removes chunks for this chat)
POST   /chats/{chat_id}/documents/{doc_id}/promote   copy into the global knowledge base

status: processing -> ready | failed
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from config import MAX_UPLOAD_BYTES
from core import memory
from rag import ingest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chats", tags=["chat-documents"])

ALLOWED = {".pdf", ".txt", ".md"}


def _require_chat(chat_id: str) -> None:
    if not memory.chat_exists(chat_id):
        raise HTTPException(404, "Chat not found")


def _process(chat_id: str, doc_id: str, filename: str, src: Path) -> None:
    """Background job: extract -> cache pages -> embed -> mark ready."""
    def progress(msg: str) -> None:
        memory.update_chat_document(chat_id, doc_id, detail=msg)

    try:
        doc = ingest.extract(src, progress)
        ingest.save_pages(doc_id, doc)
        memory.update_chat_document(
            chat_id, doc_id, pages=len(doc.pages), chars=doc.chars, detail="Indexing…"
        )
        ingest.delete_chat_doc_chunks(chat_id, doc_id)   # idempotent re-attach
        ingest.index_pages(
            doc,
            filename=filename,
            doc_id=doc_id,
            store=ingest.chat_store(),
            id_prefix=f"{chat_id}-{doc_id}",
            extra_meta={"scope": "chat", "chat_id": chat_id},
            progress=progress,
        )
        memory.update_chat_document(chat_id, doc_id, status="ready", detail="")
    except ingest.IngestError as e:
        memory.update_chat_document(chat_id, doc_id, status="failed", detail=str(e))
    except Exception:
        logger.exception("Attachment processing failed for %s", filename)
        memory.update_chat_document(
            chat_id, doc_id, status="failed", detail="Something went wrong while reading this file."
        )
    finally:
        shutil.rmtree(src.parent, ignore_errors=True)   # the per-upload temp dir


@router.post("/{chat_id}/documents", status_code=202)
def attach_document(chat_id: str, background: BackgroundTasks, file: UploadFile = File(...)):
    # Plain `def`: FastAPI runs it in the threadpool, so reading the upload never blocks the loop.
    _require_chat(chat_id)

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename:
        raise HTTPException(400, "Missing file name")
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Attach a PDF, TXT or MD file.")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(400, "File is empty")

    doc_id = hashlib.sha256(data).hexdigest()[:16]
    existing = memory.get_chat_document(chat_id, doc_id)
    if existing and existing["status"] != "failed":
        return existing                                   # same file attached twice -> reuse

    tmp_dir = Path(tempfile.mkdtemp(prefix="lm_attach_"))
    src = tmp_dir / filename
    src.write_bytes(data)

    memory.add_chat_document(chat_id, doc_id, filename, len(data))
    background.add_task(_process, chat_id, doc_id, filename, src)
    return memory.get_chat_document(chat_id, doc_id)


@router.get("/{chat_id}/documents")
def list_documents(chat_id: str):
    _require_chat(chat_id)
    return {"documents": memory.list_chat_documents(chat_id)}


@router.get("/{chat_id}/documents/{doc_id}")
def get_document(chat_id: str, doc_id: str):
    doc = memory.get_chat_document(chat_id, doc_id)
    if doc is None:
        raise HTTPException(404, "Attachment not found")
    return doc


@router.delete("/{chat_id}/documents/{doc_id}", status_code=204)
def detach_document(chat_id: str, doc_id: str):
    if memory.get_chat_document(chat_id, doc_id) is None:
        return
    ingest.delete_chat_doc_chunks(chat_id, doc_id)
    memory.remove_chat_document(chat_id, doc_id)
    if memory.count_doc_refs(doc_id) == 0:
        ingest.delete_pages(doc_id)


@router.post("/{chat_id}/documents/{doc_id}/promote")
def promote_document(chat_id: str, doc_id: str):
    doc = memory.get_chat_document(chat_id, doc_id)
    if doc is None:
        raise HTTPException(404, "Attachment not found")
    if doc["status"] != "ready":
        raise HTTPException(409, "This file is still being processed.")
    try:
        chunks = ingest.promote_to_kb(doc_id, doc["filename"])
    except ingest.IngestError as e:
        raise HTTPException(422, str(e))
    memory.add_document(doc["filename"], doc["size_bytes"], chunks)
    return {"filename": doc["filename"], "chunks_stored": chunks}