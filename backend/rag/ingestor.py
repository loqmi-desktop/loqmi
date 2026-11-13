from __future__ import annotations

import logging
from pathlib import Path
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(file_path: str) -> tuple[int, str]:
    """
    Ingest a single PDF, text file, or Image into ChromaDB.
    Returns the number of chunks stored.
    Raises FileNotFoundError or ValueError for bad input.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    
    # 1. Handle PDFs
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        raw_docs = loader.load()
        
    # 2. Handle Text/Markdown
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
        raw_docs = loader.load()
        
    # 3. Handle Images (Screenshots via OCR)
    elif suffix in (".png", ".jpg", ".jpeg"):
        try:
            import pytesseract
            from PIL import Image as PILImage
            extracted_text = pytesseract.image_to_string(PILImage.open(str(path)))
            raw_docs = [Document(
                page_content=extracted_text,
                metadata={"source": str(path)}
            )]
        except ImportError:
            raise ValueError("Image ingestion requires pytesseract. Not available in this build.")
        except Exception as e:
            raise ValueError(f"Failed to extract text from image: {str(e)}")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: .pdf, .txt, .md, .png, .jpg, .jpeg"
        )
    full_text = "\n\n".join([doc.page_content for doc in raw_docs])

    chunks = _get_splitter().split_documents(raw_docs)

    # Tag every chunk with its source filename for later deletion/listing
    for chunk in chunks:
        chunk.metadata["source_file"] = path.name

    _get_store().add_documents(chunks)

    logger.info("Ingested '%s' — %d chunks stored", path.name, len(chunks))
    return len(chunks), full_text


def delete_source(source_file_name: str) -> int:
    """
    Remove all chunks belonging to a specific source file from ChromaDB.
    Returns the number of chunks deleted.
    """
    store = _get_store()

    result = store._collection.get(
        where={"source_file": source_file_name},
        include=["metadatas"],
    )
    ids = result.get("ids", [])

    if not ids:
        logger.debug("No chunks found for '%s'", source_file_name)
        return 0

    store._collection.delete(ids=ids)
    logger.info("Deleted %d chunks from '%s'", len(ids), source_file_name)
    return len(ids)