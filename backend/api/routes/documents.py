"""
api/routes/documents.py
RAG document management routes.

POST   /documents/ingest     — upload and ingest a file into ChromaDB
GET    /documents            — list all indexed source documents with metadata
DELETE /documents/{filename} — remove all chunks for a source file and clear metadata
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from core import memory
from rag import ingest_file, delete_source, IngestError

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}


@router.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    """
    Accept a file upload, extract text, ingest into ChromaDB, and track metadata.

    Deliberately a sync handler: FastAPI runs it in its threadpool, so the
    blocking embedding + ChromaDB work doesn't stall the event loop (other
    requests, including SSE streams, stay responsive).
    """
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: .pdf, .txt, .md, .png, .jpg, .jpeg",
        )

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / filename

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(tmp_path)
        chunks, _text = ingest_file(str(tmp_path))

    except IngestError as e:
            raise HTTPException(status_code=422, detail=str(e))
    
    memory.add_document(filename, file_size, chunks)



    return {"filename": filename, "chunks_stored": chunks}


@router.get("")
def get_documents():
    """Return list of all indexed documents with full metadata."""
    # This now returns a list of dictionaries with sizes and dates!
    return {"documents": memory.list_documents_metadata()}


@router.delete("/{filename}", status_code=204)
def remove_document(filename: str):
    """Delete chunks from Chroma and metadata from SQLite."""
    deleted = delete_source(filename)
    memory.remove_document_record(filename) # Clean up the DB
    
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed chunks found for '{filename}'.",
        )