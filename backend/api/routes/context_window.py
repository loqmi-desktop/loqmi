"""
api/routes/context_window.py — per-category context window configuration.

GET /context-window             — current settings + options for every category
PUT /context-window/{category}  — set the user's chosen window for one category

Each category's model keeps exactly one context window at a time (n_ctx is
fixed at Llama() construction time — see core/model_manager.py). Setting a
new value here doesn't force an immediate reload; model_manager picks it
up the next time that category is loaded, including swapping out an
already-loaded model if only the context window changed.
"""

from __future__ import annotations

import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import (
    CLASSIFIER_CATEGORIES,
    get_model_config,
    get_recommended_model,
)
from core import memory, model_manager

router = APIRouter(prefix="/context-window", tags=["context-window"])

# Presets offered in the UI; each category's actual option list is this
# list filtered down to values <= that model's real ceiling.
PRESETS = [2048, 4096, 8192, 16384, 32768]


class ContextWindowUpdate(BaseModel):
    value: int


class CategoryContextInfo(BaseModel):
    category: str
    model_id: str
    model_name: str
    current: int       # effective value in use right now
    ceiling: int        # model's real trained max, from GGUF metadata
    recommended: int    # soft, RAM-tier-based suggestion — not a hard cap
    options: list[int]  # presets <= ceiling


def _active_model_for_category(category: str) -> dict | None:
    model_id = memory.get_active_model(category)
    if model_id is None:
        return get_recommended_model(category)
    return get_model_config(category, model_id)


def _recommend(ceiling: int) -> int:
    """
    A rough, RAM-tier-based suggestion — not a precise KV-cache
    calculation (that varies too much across architectures to compute
    reliably from what we know here). This never blocks anything; it's
    just what gets marked "Recommended" in the UI. The user can pick any
    option regardless.
    """
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    if total_gb < 8:
        target = 2048
    elif total_gb < 16:
        target = 4096
    elif total_gb < 32:
        target = 8192
    else:
        target = 16384

    valid = [p for p in PRESETS if p <= ceiling]
    if not valid:
        return ceiling
    candidates = [p for p in valid if p <= target]
    return max(candidates) if candidates else max(valid)


@router.get("", response_model=list[CategoryContextInfo])
def get_all_context_windows():
    results = []
    for category in CLASSIFIER_CATEGORIES:
        cfg = _active_model_for_category(category)
        if cfg is None:
            continue
        model_id = cfg["id"]
        ceiling = model_manager.get_context_ceiling(category, model_id, cfg)

        current = memory.get_context_window(category) or cfg.get("n_ctx", 2048)
        current = min(current, ceiling)

        options = [p for p in PRESETS if p <= ceiling]
        if ceiling not in options:
            options.append(ceiling)
            options.sort()

        results.append(CategoryContextInfo(
            category=category,
            model_id=model_id,
            model_name=cfg["name"],
            current=current,
            ceiling=ceiling,
            recommended=_recommend(ceiling),
            options=options,
        ))
    return results


@router.put("/{category}", status_code=204)
def set_context_window(category: str, body: ContextWindowUpdate):
    if category not in CLASSIFIER_CATEGORIES:
        raise HTTPException(404, f"Unknown category '{category}'")

    cfg = _active_model_for_category(category)
    if cfg is None:
        raise HTTPException(400, f"No active model for category '{category}'")

    ceiling = model_manager.get_context_ceiling(category, cfg["id"], cfg)
    if body.value > ceiling:
        raise HTTPException(
            400,
            f"Requested context window {body.value} exceeds this model's "
            f"trained maximum of {ceiling}.",
        )
    if body.value < 512:
        raise HTTPException(400, "Context window must be at least 512 tokens.")

    memory.set_context_window(category, body.value)