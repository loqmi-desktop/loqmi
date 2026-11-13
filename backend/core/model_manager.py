"""
core/model_manager.py — GGUF model loader/unloader
Keeps exactly one model in RAM at a time.
Reuses the loaded model if the same category is requested again.
Supports multiple models per category with active model selection.
"""

from __future__ import annotations

import logging
import threading
from typing import Literal, Any, NamedTuple
import psutil

from llama_cpp import Llama

from config import ALL_MODEL_CATEGORIES, CLASSIFIER_CATEGORIES, HELPER_CATEGORY, HELPER_N_CTX, HELPER_RAM_FACTOR, HELPER_RAM_HEADROOM_GB, get_models_for_category, get_model_config, get_recommended_model, model_path, model_exists

from core import memory

logger = logging.getLogger(__name__)

# Download state machine (in-memory, resets on restart)
DownloadState = Literal["not_downloaded", "downloading", "ready", "deleting"]
_download_states: dict[str, dict[str, DownloadState]] = {}  # {category: {model_id: state}}

# Guards the loaded-model globals and download states. RLock because load()
# recurses into itself when falling back to the chat category.
#
# Also guards generation itself — see acquire_for_generation() below. The
# Llama object isn't safe for concurrent use from multiple threads, and
# previously nothing prevented two generation calls (e.g. a live chat
# reply and the screenshot-summary call in api/routes/attachments.py)
# from racing each other, or from racing a load() that swaps the model
# out mid-generation.
_lock = threading.RLock()

_current_model: Llama | None = None
_current_category: str | None = None
_current_model_id: str | None = None
_current_n_ctx: int | None = None


def _init_download_states() -> None:
    """Initialize download states based on disk."""
    global _download_states

    for category in ALL_MODEL_CATEGORIES:
        models = get_models_for_category(category)
        _download_states[category] = {}

        for model in models:
            model_id = model["id"]

            if model_exists(category, model_id):
                _download_states[category][model_id] = "ready"
            else:
                _download_states[category][model_id] = "not_downloaded"


def get_download_state(category: str, model_id: str) -> DownloadState:
    """Get the download state of a model (not_downloaded, downloading, ready, deleting)."""
    if not _download_states:
        _init_download_states()
    return _download_states.get(category, {}).get(model_id, "not_downloaded")


def set_download_state(category: str, model_id: str, state: DownloadState) -> None:
    """Set the download state of a model."""
    if not _download_states:
        _init_download_states()
    if category not in _download_states:
        _download_states[category] = {}
    _download_states[category][model_id] = state


# ── Context window ────────────────────────────────────────────────────────

def get_context_ceiling(category: str, model_id: str, cfg: dict | None = None) -> int:
    """
    The model's REAL trained max context length, read from its own GGUF
    metadata — not the (often conservative) default in MODELS[...]["n_ctx"].

    Read via a cheap `vocab_only=True` load, which parses only the
    tokenizer and metadata, not the full weight tensors — fast and
    low-RAM, safe to do without disturbing whatever model is currently
    loaded for generation. Result is cached in the settings table
    (memory.set_cached_n_ctx_train) so this only ever runs once per
    model file, not once per request.

    Falls back to the model's configured default n_ctx if the metadata
    can't be read for any reason (missing key, unexpected architecture,
    file not downloaded yet) rather than raising — this is advisory data
    for the UI, not something that should break model loading.
    """
    cached = memory.get_cached_n_ctx_train(category, model_id)
    if cached is not None:
        return cached

    if cfg is None:
        cfg = get_model_config(category, model_id)
    default_n_ctx = cfg.get("n_ctx", 2048) if cfg else 2048

    if not model_exists(category, model_id):
        return default_n_ctx

    ceiling = default_n_ctx
    try:
        peek = Llama(
            model_path=str(model_path(category, model_id)),
            vocab_only=True,
            verbose=False,
        )
        try:
            arch = peek.metadata.get("general.architecture")
            raw = peek.metadata.get(f"{arch}.context_length") if arch else None
            if raw is not None:
                ceiling = int(raw)
        finally:
            del peek
    except Exception:
        logger.warning(
            "Could not read GGUF metadata for '%s:%s' — falling back to configured n_ctx=%s",
            category, model_id, default_n_ctx, exc_info=True,
        )

    memory.set_cached_n_ctx_train(category, model_id, ceiling)
    return ceiling


def _resolve_n_ctx(category: str, model_id: str, cfg: dict) -> int:
    """
    Effective context window to load this model with: the user's saved
    per-category preference if set, clamped to the model's real ceiling
    (in case the preference predates a model swap or the ceiling turned
    out lower than assumed); otherwise the model's configured default.
    """
    default_n_ctx = cfg.get("n_ctx", 2048)
    override = memory.get_context_window(category)
    if override is None:
        return default_n_ctx
    ceiling = get_context_ceiling(category, model_id, cfg)
    return min(override, ceiling)


def load(category: str, model_id: str | None = None) -> Llama:
    """
    Load the model for the given category into RAM.
    If model_id is None, uses the active model from settings (or recommended if not set).
    Unloads the previous model first if a different one is loaded.
    Reuses the current model if the same category+model_id+context-window
    is already loaded.
    
    Raises FileNotFoundError if the model file hasn't been downloaded yet.
    Raises ValueError if model_id is invalid.
    """
    global _current_model, _current_category, _current_model_id

    with _lock:
        return _load_locked(category, model_id)


def _load_locked(category: str, model_id: str | None) -> Llama:
    global _current_model, _current_category, _current_model_id, _current_n_ctx

    # Resolve which model to load
    effective_category = category

    if model_id is None:
        # Use active model from settings, fall back to recommended
        model_id = memory.get_active_model(effective_category)
        if model_id is None:
            cfg = get_recommended_model(effective_category)
            if cfg is None:
                raise ValueError(f"No models available for category '{effective_category}'")
            model_id = cfg["id"]

    # Fast path: same category+model already loaded. Still need to check
    # whether the user's context-window setting changed since it loaded —
    # n_ctx is fixed at Llama() construction time, so a settings-only
    # change must still trigger a reload even though category/model_id
    # match.
    if (_current_category == effective_category and
        _current_model_id == model_id and
        _current_model is not None):
        cfg_check = get_model_config(effective_category, model_id)
        if cfg_check is not None and _resolve_n_ctx(effective_category, model_id, cfg_check) == _current_n_ctx:
            logger.debug("'%s:%s' already loaded, reusing", effective_category, model_id)
            return _current_model
        # else: context window changed since load — fall through to reload

    # Unload current model
    if _current_model is not None:
        logger.info("Unloading '%s:%s'", _current_category, _current_model_id)
        _current_model.close()
        _current_model = None
        _current_category = None
        _current_model_id = None
        _current_n_ctx = None

    # Validate and check if model is downloaded
    cfg = get_model_config(effective_category, model_id)
    if cfg is None:
        raise ValueError(f"Model '{model_id}' not found in category '{effective_category}'")

    if not model_exists(effective_category, model_id):
        if effective_category != "chat":
            logger.info("Model '%s' for '%s' not found, falling back to 'chat'", model_id, effective_category)
            return _load_locked("chat", None)
        else:
            raise FileNotFoundError(
                f"Base chat model '{model_id}' not downloaded. "
                f"Expected at: {model_path(effective_category, model_id)}"
            )

    effective_n_ctx = _resolve_n_ctx(effective_category, model_id, cfg)

    # Load the model
    logger.info("Loading '%s:%s' (n_ctx=%s)...", effective_category, model_id, effective_n_ctx)
    _current_model = Llama(
        model_path=str(model_path(effective_category, model_id)),
        n_ctx=effective_n_ctx,
        n_gpu_layers=cfg.get("n_gpu_layers", -1), # Safe fallback for Apple Metal
        verbose=False,
    )
    _current_category = effective_category
    _current_model_id = model_id
    _current_n_ctx = effective_n_ctx
    logger.info("Model ready")
    return _current_model


def unload() -> None:
    """Explicitly unload the current model and free RAM."""
    global _current_model, _current_category, _current_model_id, _current_n_ctx
    with _lock:
        if _current_model is not None:
            logger.info("Unloading '%s:%s'", _current_category, _current_model_id)
            _current_model.close()
            _current_model = None
            _current_category = None
            _current_model_id = None
            _current_n_ctx = None


def current_category() -> str | None:
    return _current_category


def current_model_id() -> str | None:
    return _current_model_id


def current_n_ctx() -> int | None:
    return _current_n_ctx


def acquire_for_generation():
    """
    Returns the same lock that guards load()/unload(), so generation
    callers elsewhere (core/inference.py, the screenshot-summary call in
    api/routes/attachments.py) can serialize model.create_chat_completion()
    calls against each other and against a concurrent model swap.

    Usage:
        with model_manager.acquire_for_generation():
            result = model.create_chat_completion(...)

    Safe to hold across a generator's yields (e.g. while streaming tokens)
    since RLock is reentrant per-thread and the generator runs on a single
    thread for its whole lifetime.
    """
    return _lock


# ── Search helper: an optional second resident model ──────────────────────────
# Lock order: _lock and _helper_lock are never held together, so they can't deadlock.

_helper_lock = threading.RLock()
_helper_model: Llama | None = None
_helper_key: tuple[str, str] | None = None     # (category, model_id)


class ResearchHandle(NamedTuple):
    model: Llama
    lock: Any
    kind: str     # "dedicated" | "main" | "chat_resident" | "swapped"


def handle_for_main(model: Llama) -> ResearchHandle:
    return ResearchHandle(model, _lock, "main")


def _fits_in_ram(category: str, model_id: str) -> bool:
    cfg = get_model_config(category, model_id) or {}
    need = float(cfg.get("size_gb") or 5.0) * HELPER_RAM_FACTOR + HELPER_RAM_HEADROOM_GB
    free = psutil.virtual_memory().available / 1024 ** 3
    if free < need:
        logger.info("Second model '%s:%s' needs ~%.1f GB free, only %.1f GB available",
                    category, model_id, need, free)
    return free >= need


def _same_file(a: tuple[str, str], b: tuple[str, str]) -> bool:
    try:
        return model_path(*a) == model_path(*b)
    except ValueError:
        return False


def _unload_helper_locked() -> None:
    global _helper_model, _helper_key
    if _helper_model is not None:
        logger.info("Unloading search helper '%s:%s'", *_helper_key)
        _helper_model.close()
        _helper_model, _helper_key = None, None


def unload_helper() -> None:
    with _helper_lock:
        _unload_helper_locked()


def unload_helper_if(category: str, model_id: str) -> None:
    """Called before deleting a model file."""
    with _helper_lock:
        if _helper_key == (category, model_id):
            _unload_helper_locked()


def _load_helper_slot(category: str, model_id: str) -> Llama:
    global _helper_model, _helper_key
    key = (category, model_id)
    with _helper_lock:
        if _helper_model is not None and _helper_key == key:
            return _helper_model
        _unload_helper_locked()
        cfg = get_model_config(category, model_id) or {}
        logger.info("Loading search helper '%s:%s' (n_ctx=%s)...", category, model_id, HELPER_N_CTX)
        _helper_model = Llama(
            model_path=str(model_path(category, model_id)),
            n_ctx=HELPER_N_CTX,
            n_gpu_layers=cfg.get("n_gpu_layers", -1),
            verbose=False,
        )
        _helper_key = key
        return _helper_model


def get_research_handle(main_category: str) -> ResearchHandle:
    """
    Order of preference:
      1. dedicated helper, resident alongside the main model (no swap)
      2. main model is already the chat model (free)
      3. dedicated helper swapped in for the main model (small, fast to load)
      4. no dedicated helper: chat model, resident if RAM allows, else swapped in
    kind == "swapped" means the main model was unloaded: the caller MUST call
    restore_main() afterwards.
    """
    main_key = (_current_category, _current_model_id) if _current_model is not None else None

    helper_id = memory.get_active_model(HELPER_CATEGORY)
    helper_key = (HELPER_CATEGORY, helper_id) if helper_id and model_exists(HELPER_CATEGORY, helper_id) else None
    chat_id = memory.get_active_model("chat")
    chat_key = ("chat", chat_id) if chat_id and model_exists("chat", chat_id) else None
    if not helper_key and not chat_key:
        raise FileNotFoundError("No search helper and no chat model are downloaded.")

    if helper_key and (_helper_key == helper_key or _fits_in_ram(*helper_key)):
        return ResearchHandle(_load_helper_slot(*helper_key), _helper_lock, "dedicated")

    if chat_key and main_key and (main_key == chat_key or _same_file(main_key, chat_key)):
        return ResearchHandle(_current_model, _lock, "main")

    if helper_key:
        unload()                                    # free the main model, then use the small helper
        return ResearchHandle(_load_helper_slot(*helper_key), _helper_lock, "swapped")

    if _helper_key == chat_key or _fits_in_ram(*chat_key):
        return ResearchHandle(_load_helper_slot(*chat_key), _helper_lock, "chat_resident")
    return ResearchHandle(load(*chat_key), _lock, "swapped")


def restore_main(category: str) -> Llama:
    """Undo a "swapped" handle: free the helper slot, then reload the routed model."""
    unload_helper()
    return load(category)