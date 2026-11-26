"""
core/router.py
"""
from __future__ import annotations

import logging
import time

from config import (
    CLASSIFIER_CATEGORIES,
    CLASSIFIER_MIN_CONFIDENCE,
    ROUTING_FOLLOWUP_MAX_WORDS,
    get_model_config,
)
from core import classifier, inference, memory, model_manager

logger = logging.getLogger(__name__)


def _stabilize(category: str, confidence: float, prompt: str, chat_id: str) -> tuple[str, str | None]:
    """
    Keep routing sticky within a chat. Short follow-ups ("continue",
    "why?") carry no signal for the classifier, and low-confidence
    predictions are mostly guesses; both reuse the chat's previous
    category (or fall back to 'chat' if there isn't one).
    Returns (category, reason_if_overridden).
    """
    last = memory.get_last_category(chat_id)

    if last and category != last and len(prompt.split()) <= ROUTING_FOLLOWUP_MAX_WORDS:
        return last, "short follow-up"

    if confidence < CLASSIFIER_MIN_CONFIDENCE:
        fallback = last or "chat"
        if fallback != category:
            return fallback, "low confidence"

    return category, None

def _guard_category(category: str, prompt: str, persist_content: str,
                    chat_id: str, last: str | None) -> tuple[str, str | None]:
    """
    Rule-based sanity checks on top of the classifier:
    - 'document' needs a document in play (attached file, RAG on, or an earlier document turn).
    - a very short opening message ("hello", "thanks") has no signal; use chat.
    """
    if category == "document":
        has_file_block = any(
            b.get("label") == "file" for b in memory.parse_content_blocks(persist_content)
        )
        if not (has_file_block or memory.get_rag_enabled(chat_id) or last == "document"):
            return "chat", "no document in play"

    if last is None and len(prompt.split()) <= 2 and category != "chat":
        return "chat", "too short to classify"

    return category, None


def handle(prompt: str, persist_content: str, chat_id: str, override_category: str | None = None):
    """
    Full pipeline for a specific chat session.
    Bypasses the classification model if override_category is explicitly set.
    """
    t0 = time.time()

    if override_category:
        if override_category not in CLASSIFIER_CATEGORIES:
            raise ValueError(
                f"Unknown category '{override_category}'. "
                f"Valid: {', '.join(CLASSIFIER_CATEGORIES)}"
            )
        category = override_category
        classify_ms = 0.0
        logger.debug("Override forced category: %s", category)
    else:
        raw_category, confidence = classifier.classify_with_confidence(prompt)
        classify_ms = round((time.time() - t0) * 1000, 1)
        last = memory.get_last_category(chat_id)
        category, reason = _stabilize(raw_category, confidence, prompt, chat_id)
        category, guard_reason = _guard_category(category, prompt, persist_content, chat_id, last)
        reason = guard_reason or reason
        logger.info(
            "Classified '%s' (conf=%.2f, %sms) -> using '%s'%s",
            raw_category, confidence, classify_ms, category,
            f" [{reason}]" if reason else "",
        )

    memory.set_last_category(chat_id, category)
    model = model_manager.load(category)

    t1 = time.time()
    for chunk in inference.run(prompt, persist_content, model, category, chat_id):
        yield chunk

    inference_s = round(time.time() - t1, 2)
    logger.info("Turn done — classify: %sms | inference: %ss", classify_ms, inference_s)

    current_cat = model_manager.current_category()
    current_id = model_manager.current_model_id()
    cfg = get_model_config(current_cat, current_id) if current_cat and current_id else None
    real_model_name = cfg["name"] if cfg else "Local Model"

    yield {
        "type":        "stats",
        "category":    category,
        "model_name":  real_model_name,
        "classify_ms": classify_ms,
        "inference_s": inference_s,
    }