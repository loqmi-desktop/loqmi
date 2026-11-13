"""
rag/ingest.py
Extraction + indexing for PDFs, text files and images.

Design:
  extract()      file  -> ExtractedDoc (text per page, OCR fallback per page)
  index_pages()  pages -> chunks in a Chroma collection, tagged with
                          source_file / doc_id / page / paged
  ingest_file()  legacy knowledge-base entry point (used by /documents/ingest)

Chat-scoped attachments live in their own collection (CHROMA_CHAT_COLLECTION)
so the existing knowledge-base collection needs no migration.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHAT_DOCS_DIR,
    CHROMA_CHAT_COLLECTION,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_OCR_PAGES,
    MAX_PDF_PAGES,
    MIN_PAGE_CHARS,
)
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


class IngestError(ValueError):
    """A user-facing problem with the file (message is safe to show)."""


@dataclass
class ExtractedDoc:
    pages: list[str]
    paged: bool            # True for PDFs: page numbers are meaningful
    ocr_pages: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.pages if p.strip())

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.pages)


# ── Stores ────────────────────────────────────────────────────────────────────

_stores: dict[str, Chroma] = {}


def get_store(collection: str) -> Chroma:
    if collection not in _stores:
        _stores[collection] = Chroma(
            collection_name=collection,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return _stores[collection]


def kb_store() -> Chroma:
    return get_store(CHROMA_COLLECTION_NAME)


def chat_store() -> Chroma:
    return get_store(CHROMA_CHAT_COLLECTION)


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── OCR (optional engines, no hard dependency) ────────────────────────────────

_ocr_engine = None


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_image(img) -> str:
    """OCR a PIL image. Prefers RapidOCR (no external binary), falls back to Tesseract."""
    global _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        if _ocr_engine is None:
            _ocr_engine = RapidOCR()
        result, _ = _ocr_engine(np.array(img.convert("RGB")))
        return "\n".join(line[1] for line in (result or []))
    except ImportError:
        pass
    try:
        import pytesseract
        return pytesseract.image_to_string(img)
    except Exception as e:
        raise IngestError(
            "This file needs OCR, but no OCR engine is available in this build."
        ) from e


# ── Extraction ────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf(path: Path, progress: Optional[ProgressFn]) -> ExtractedDoc:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        if "password" in str(e).lower():
            raise IngestError("This PDF is password-protected. Remove the password and try again.")
        raise IngestError("This PDF looks corrupt or unreadable.")

    n = len(pdf)
    if n == 0:
        pdf.close()
        raise IngestError("This PDF has no pages.")
    if n > MAX_PDF_PAGES:
        pdf.close()
        raise IngestError(f"This PDF has {n} pages. The limit is {MAX_PDF_PAGES}.")

    pages: list[str] = []
    ocr_used = 0
    can_ocr = None  # resolved lazily, only if a page needs it

    try:
        for i in range(n):
            if progress:
                progress(f"Reading page {i + 1} of {n}")
            page = pdf[i]
            try:
                tp = page.get_textpage()
                text = (tp.get_text_range() or "").strip()
                tp.close()

                if len(text) < MIN_PAGE_CHARS and ocr_used < MAX_OCR_PAGES:
                    if can_ocr is None:
                        can_ocr = ocr_available()
                    if can_ocr:
                        if progress:
                            progress(f"Running OCR on page {i + 1} of {n}")
                        text = _ocr_image(page.render(scale=2.0).to_pil()).strip()
                        ocr_used += 1
                pages.append(_clean(text))
            finally:
                page.close()
    finally:
        pdf.close()

    doc = ExtractedDoc(pages=pages, paged=True, ocr_pages=ocr_used)
    if doc.chars < MIN_PAGE_CHARS:
        if can_ocr is False:
            raise IngestError(
                "This looks like a scanned PDF and no OCR engine is installed "
                "(pip install rapidocr-onnxruntime)."
            )
        raise IngestError("No text could be extracted from this PDF.")
    return doc


def _extract_text(path: Path) -> ExtractedDoc:
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        raise IngestError("This file is empty.")
    # Split long plain text into ~3000-char segments so whole-doc injection and
    # retrieval treat it like a paged document (page numbers are NOT shown).
    segs = [text[i:i + 3000] for i in range(0, len(text), 3000)]
    return ExtractedDoc(pages=segs, paged=False)


def _extract_image(path: Path) -> ExtractedDoc:
    from PIL import Image
    text = _clean(_ocr_image(Image.open(path)))
    if not text:
        raise IngestError("No text could be read from this image.")
    return ExtractedDoc(pages=[text], paged=False, ocr_pages=1)


def extract(path: Path, progress: Optional[ProgressFn] = None) -> ExtractedDoc:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, progress)
    if suffix in (".txt", ".md"):
        return _extract_text(path)
    if suffix in (".png", ".jpg", ".jpeg"):
        return _extract_image(path)
    raise IngestError(f"Unsupported file type '{suffix}'. Supported: .pdf, .txt, .md, .png, .jpg, .jpeg")


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_pages(
    doc: ExtractedDoc,
    *,
    filename: str,
    doc_id: str,
    store: Chroma,
    id_prefix: str,
    extra_meta: Optional[dict] = None,
    progress: Optional[ProgressFn] = None,
) -> int:
    """Chunk page by page (so every chunk keeps its page number) and upsert."""
    splitter = _splitter()
    chunks: list[Document] = []
    for page_no, text in enumerate(doc.pages, start=1):
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            chunks.append(Document(
                page_content=piece,
                metadata={
                    "source_file": filename,
                    "doc_id": doc_id,
                    "page": page_no,
                    "paged": doc.paged,
                    **(extra_meta or {}),
                },
            ))
    if not chunks:
        raise IngestError("No text could be extracted from this file.")

    BATCH = 64
    for start in range(0, len(chunks), BATCH):
        if progress:
            progress(f"Indexing {min(start + BATCH, len(chunks))} of {len(chunks)} passages")
        batch = chunks[start:start + BATCH]
        store.add_documents(batch, ids=[f"{id_prefix}-{start + k}" for k in range(len(batch))])

    logger.info("Indexed '%s': %d chunks", filename, len(chunks))
    return len(chunks)


# ── Page cache (so a chat can inject a whole small document later) ────────────

def save_pages(doc_id: str, doc: ExtractedDoc) -> None:
    CHAT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (CHAT_DOCS_DIR / f"{doc_id}.json").write_text(
        json.dumps({"paged": doc.paged, "pages": doc.pages}), encoding="utf-8"
    )


def load_pages(doc_id: str) -> Optional[ExtractedDoc]:
    p = CHAT_DOCS_DIR / f"{doc_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return ExtractedDoc(pages=data["pages"], paged=data.get("paged", False))


def delete_pages(doc_id: str) -> None:
    (CHAT_DOCS_DIR / f"{doc_id}.json").unlink(missing_ok=True)


# ── Chat-scoped chunk management ──────────────────────────────────────────────

def delete_chat_doc_chunks(chat_id: str, doc_id: str) -> int:
    col = chat_store()._collection
    ids = col.get(where={"$and": [{"chat_id": chat_id}, {"doc_id": doc_id}]}).get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)


def purge_chat(chat_id: str) -> int:
    """Remove every chat-scoped chunk for a chat. Call when the chat is deleted."""
    col = chat_store()._collection
    ids = col.get(where={"chat_id": chat_id}).get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)


# ── Knowledge base (legacy entry points) ──────────────────────────────────────

def delete_source(source_file_name: str) -> int:
    col = kb_store()._collection
    ids = col.get(where={"source_file": source_file_name}).get("ids", [])
    if not ids:
        return 0
    col.delete(ids=ids)
    logger.info("Deleted %d chunks from '%s'", len(ids), source_file_name)
    return len(ids)


def ingest_file(file_path: str, progress: Optional[ProgressFn] = None) -> tuple[int, str]:
    """
    Knowledge-base ingest. Re-uploading a file with the same name REPLACES its
    chunks instead of duplicating them. Returns (chunk_count, full_text).
    Raises IngestError for problems the user can fix.
    """
    path = Path(file_path)
    doc = extract(path, progress)
    delete_source(path.name)
    n = index_pages(
        doc,
        filename=path.name,
        doc_id=path.name,
        store=kb_store(),
        id_prefix=f"kb-{path.name}",
        extra_meta={"scope": "kb"},
        progress=progress,
    )
    return n, doc.text


def promote_to_kb(doc_id: str, filename: str) -> int:
    """Copy a chat attachment into the knowledge base (re-embeds from cached pages)."""
    doc = load_pages(doc_id)
    if doc is None:
        raise IngestError("The original text for this file is no longer available.")
    delete_source(filename)
    return index_pages(
        doc,
        filename=filename,
        doc_id=doc_id,
        store=kb_store(),
        id_prefix=f"kb-{filename}",
        extra_meta={"scope": "kb"},
    )