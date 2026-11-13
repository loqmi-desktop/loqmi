from __future__ import annotations
import logging
from typing import Optional

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the singleton embedding model, loading it on first call.
    Thread-safe for read access — FastAPI runs in a single process.
    """
    global _embeddings

    if _embeddings is None:
        logger.info("Loading embedding model '%s' (first call only)", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},          # embeddings always on CPU
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model ready")

    return _embeddings