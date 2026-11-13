"""
core/context_budget.py
Token budgeting for one chat turn. Pure logic, no I/O except tokenizing
with the loaded model, so every decision is made with the model's own
tokenizer.

Priorities, highest first:
  1. the system prompt (RAG context is trimmed first if it's too big)
  2. the current user message (never silently dropped; oversized *typed*
     text is rejected, oversized OCR / attached-file text is trimmed)
  3. history, as whole user+assistant turns from newest to oldest
Output room and a template safety margin are reserved before any of that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from llama_cpp import Llama

from config import (
    CONTEXT_HISTORY_MSG_MAX_FRACTION,
    CONTEXT_MIN_OUTPUT_TOKENS,
    CONTEXT_MIN_SAFETY_TOKENS,
    CONTEXT_MSG_OVERHEAD,
    CONTEXT_OUTPUT_FRACTION,
    CONTEXT_SAFETY_FRACTION,
    CONTEXT_SYSTEM_MAX_FRACTION,
    INFERENCE_MAX_TOKENS,
)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_MIN_AVAILABLE = 200
_ACTION_RE = re.compile(r"<action>.*?(?:</action>|$)", re.DOTALL | re.IGNORECASE)


class ContextOverflowError(Exception):
    """The turn cannot fit in the context window. Raised BEFORE anything is persisted."""


@dataclass
class FitResult:
    messages: list[dict]
    max_tokens: int
    n_ctx: int
    available_tokens: int
    used_tokens: int
    dropped_messages: int = 0
    clipped_messages: int = 0
    prompt_trimmed: bool = False
    rag_trimmed: bool = False
    tool_budget_tokens: int = 0

    @property
    def was_trimmed(self) -> bool:
        return bool(
            self.dropped_messages or self.clipped_messages
            or self.prompt_trimmed or self.rag_trimmed
        )

    def event(self) -> dict | None:
        """SSE event for the UI, or None if nothing was trimmed."""
        if not self.was_trimmed:
            return None
        return {
            "type": "context",
            "dropped_messages": self.dropped_messages,
            "clipped_messages": self.clipped_messages,
            "prompt_trimmed": self.prompt_trimmed,
            "rag_trimmed": self.rag_trimmed,
            "used_tokens": self.used_tokens,
            "available_tokens": self.available_tokens,
            "n_ctx": self.n_ctx,
        }


# ── Token helpers ─────────────────────────────────────────────────────────────

def count_tokens(model: Llama, text: str) -> int:
    if not text:
        return 0
    return len(model.tokenize(text.encode("utf-8"), add_bos=False))


def clip_to_tokens(model: Llama, text: str, max_tokens: int, keep_tail_fraction: float = 0.3) -> str:
    """
    Shrink `text` to roughly `max_tokens`, keeping the start and the end
    and replacing the middle with a marker. (Code and logs tend to carry
    the important parts at both ends.) Returns `text` untouched if it fits.
    """
    tokens = model.tokenize(text.encode("utf-8"), add_bos=False)
    if len(tokens) <= max_tokens:
        return text

    keep = max(0, max_tokens - 12)               # ~12 tokens for the marker
    tail_n = int(keep * keep_tail_fraction)
    head_n = keep - tail_n
    omitted = len(tokens) - head_n - tail_n

    head = model.detokenize(tokens[:head_n]).decode("utf-8", errors="ignore")
    tail = (
        model.detokenize(tokens[len(tokens) - tail_n:]).decode("utf-8", errors="ignore")
        if tail_n else ""
    )
    return f"{head}\n[… ~{omitted} tokens omitted …]\n{tail}"


def compute_budget(n_ctx: int, max_output: int = INFERENCE_MAX_TOKENS) -> tuple[int, int]:
    """Returns (available_prompt_tokens, reserved_output_tokens)."""
    margin = max(CONTEXT_MIN_SAFETY_TOKENS, int(n_ctx * CONTEXT_SAFETY_FRACTION))
    max_out = min(max_output, max(CONTEXT_MIN_OUTPUT_TOKENS, int(n_ctx * CONTEXT_OUTPUT_FRACTION)))
    return n_ctx - margin - max_out, max_out


# ── History normalization ─────────────────────────────────────────────────────

def _normalize_history(history: list[dict]) -> list[dict]:
    """
    Make history strictly user/assistant alternating, starting with a user
    message and ending with an assistant message. Strict chat templates
    reject anything else, and failed turns used to leave orphan user
    messages behind. Also strips <think> blocks from assistant messages:
    replaying a reasoning model's scratch work wastes context.
    """
    out: list[dict] = []
    for m in history:
        role, content = m["role"], (m["content"] or "").strip()
        if role == "assistant":
            content = _ACTION_RE.sub("", _THINK_RE.sub("", content)).strip()
        if not content:
            continue
        if out and out[-1]["role"] == role:
            out[-1] = {"role": role, "content": content}   # keep the later one
        else:
            out.append({"role": role, "content": content})

    while out and out[0]["role"] != "user":
        out.pop(0)
    if out and out[-1]["role"] == "user":
        out.pop()           # orphan; the new user message follows directly
    return out


# ── The fit ───────────────────────────────────────────────────────────────────

def fit_turn(
    model: Llama,
    *,
    build_system: Callable[[str], str],
    rag_context: str,
    history: list[dict],
    prompt: str,
    typed_text: str,
    max_output_tokens: int = INFERENCE_MAX_TOKENS,
    tool_reserve_fraction: float = 0.0,
) -> FitResult:
    ov = CONTEXT_MSG_OVERHEAD
    n_ctx = model.n_ctx()
    available, max_out = compute_budget(n_ctx, max_output_tokens)
    if available < _MIN_AVAILABLE:
        raise ContextOverflowError(
            f"The context window ({n_ctx} tokens) is too small to hold a conversation. "
            f"Raise it in Settings."
        )

    # 1. System prompt (+ RAG). RAG is the first thing to give way.
    rag_trimmed = False
    system = build_system(rag_context)
    system_tokens = count_tokens(model, system) + ov
    system_cap = int(available * CONTEXT_SYSTEM_MAX_FRACTION)

    if system_tokens > system_cap and rag_context:
        rag_tokens = count_tokens(model, rag_context)
        allowed = rag_tokens - (system_tokens - system_cap) - 16
        rag_context = (
            clip_to_tokens(model, rag_context, allowed, keep_tail_fraction=0.0)
            if allowed > 50 else ""
        )
        rag_trimmed = True
        system = build_system(rag_context)
        system_tokens = count_tokens(model, system) + ov

    if system_tokens > available - 100:
        raise ContextOverflowError(
            "The system prompt alone fills the context window. "
            "Raise the context window in Settings."
        )

    # 2. Current user message. Never dropped silently.
    prompt_budget = available - system_tokens - ov
    prompt_tokens = count_tokens(model, prompt)
    prompt_trimmed = False

    if prompt_tokens > prompt_budget:
        typed_tokens = count_tokens(model, typed_text)
        if typed_tokens > prompt_budget:
            raise ContextOverflowError(
                f"Your message is about {typed_tokens:,} tokens, but only "
                f"{max(prompt_budget, 0):,} fit in the current {n_ctx:,}-token "
                f"context window (after the system prompt and room for the reply). "
                f"Shorten it, or raise the context window in Settings."
            )
        if typed_text and prompt.startswith(typed_text):
            # Typed text is sacred; only the attached/OCR text gives way.
            extra = prompt[len(typed_text):]
            prompt = typed_text + clip_to_tokens(model, extra, prompt_budget - typed_tokens)
        else:
            prompt = clip_to_tokens(model, prompt, prompt_budget)
        prompt_trimmed = True
        prompt_tokens = count_tokens(model, prompt)

    remaining = available - system_tokens - (prompt_tokens + ov)

    # Hold back room for tool observations so history can't eat all of it.
    tool_reserve = min(int(available * tool_reserve_fraction), max(0, remaining // 2))
    remaining -= tool_reserve

    # 3. History: whole turns, newest first. A single huge old message is
    #    clipped (head + tail) instead of evicting everything older than it.
    norm = _normalize_history(history)
    pairs = list(zip(norm[0::2], norm[1::2]))
    per_msg_cap = max(100, int(available * CONTEXT_HISTORY_MSG_MAX_FRACTION))

    kept: list[tuple[dict, dict]] = []
    clipped = 0
    for u, a in reversed(pairs):
        u_text, a_text = u["content"], a["content"]
        u_tok, a_tok = count_tokens(model, u_text), count_tokens(model, a_text)

        if u_tok > per_msg_cap:
            u_text, u_tok = clip_to_tokens(model, u_text, per_msg_cap), per_msg_cap
            clipped += 1
        if a_tok > per_msg_cap:
            a_text, a_tok = clip_to_tokens(model, a_text, per_msg_cap), per_msg_cap
            clipped += 1

        cost = u_tok + a_tok + 2 * ov
        if cost > remaining:
            break
        clipped += int(u_tok == per_msg_cap) + int(a_tok == per_msg_cap)
        kept.append(({"role": "user", "content": u_text}, {"role": "assistant", "content": a_text}))
        remaining -= cost

    kept.reverse()
    history_msgs: list[dict] = []
    for u, a in kept:
        history_msgs.extend([u, a])

    tool_budget = remaining + tool_reserve   # reserve plus whatever history didn't use
    return FitResult(
        messages=[{"role": "system", "content": system}, *history_msgs, {"role": "user", "content": prompt}],
        max_tokens=max_out,
        n_ctx=n_ctx,
        available_tokens=available,
        used_tokens=available - tool_budget,
        tool_budget_tokens=tool_budget,
        dropped_messages=len(norm) - len(history_msgs),
        clipped_messages=clipped,
        prompt_trimmed=prompt_trimmed,
        rag_trimmed=rag_trimmed,
    )