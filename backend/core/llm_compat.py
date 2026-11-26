"""core/llm_compat.py: chat-template compatibility helpers."""
from __future__ import annotations

import logging

from llama_cpp import Llama

logger = logging.getLogger(__name__)


def fold_system_into_last_user(messages: list[dict]) -> list[dict]:
    """Move the system prompt into the newest user message (what Mistral's own template does)."""
    if not messages or messages[0].get("role") != "system":
        return messages
    system, rest = messages[0]["content"], list(messages[1:])
    for i in range(len(rest) - 1, -1, -1):
        if rest[i]["role"] == "user":
            rest[i] = {"role": "user", "content": f"{system}\n\n---\n\n{rest[i]['content']}"}
            return rest
    return [{"role": "user", "content": system}, *rest]


def create_chat_completion(model: Llama, messages: list[dict], **kwargs):
    """
    Like model.create_chat_completion, but if the GGUF's chat template rejects a
    system role (older Mistral-style templates), retry with it folded into the user turn.
    """
    try:
        return model.create_chat_completion(messages=messages, **kwargs)
    except ValueError as e:  # llama.cpp surfaces template raise_exception() as ValueError
        if not messages or messages[0].get("role") != "system":
            raise
        logger.info("Chat template rejected the system role (%s); folding it into the user message", e)
        return model.create_chat_completion(
            messages=fold_system_into_last_user(messages), **kwargs
        )