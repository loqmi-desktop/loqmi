"""
All paths, constants, and tunable parameters live here.
Nothing else in the codebase hardcodes paths or magic numbers.
"""

import os
import sys
from pathlib import Path

# ── App identity ──────────────────────────────────────────────────────────────

APP_NAME    = "loqmi"
APP_VERSION = "0.1.0"


# ── OS-aware base directory ───────────────────────────────────────────────────

def _resolve_app_data_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_NAME


APP_DATA_DIR: Path = _resolve_app_data_dir()

# ── Subdirectories ────────────────────────────────────────────────────────────

MODELS_DIR:   Path = APP_DATA_DIR / "models"     
CHROMA_DIR:   Path = APP_DATA_DIR / "chroma_db"  
LOGS_DIR:     Path = APP_DATA_DIR / "logs"

CHAT_DOCS_DIR: Path = APP_DATA_DIR / "chat_docs"

for _dir in (MODELS_DIR, CHROMA_DIR, LOGS_DIR, CHAT_DOCS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── Model catalog ─────────────────────────────────────────────────────────────
#
# The catalog lives in SQLite (table `suggested_models`), seeded from
# model/model.json or a GitHub-hosted JSON — see core/catalog_sync.py.
#
# This is intentionally NOT a frozen module-level dict. An earlier version
# did `MODELS: dict = _load_models_from_db()` at import time, which caused
# two real bugs:
#
#   1. Circular import: this module did `from core import memory` at its
#      own top level to query the catalog immediately. But core/memory.py
#      does `from config import APP_DATA_DIR` at ITS top level too. When
#      core.memory is imported first (as server.py does), Python hands
#      config.py a partially-initialized `memory` module that doesn't
#      have get_suggested_models() defined yet — that call raised
#      AttributeError, silently swallowed by a broad `except Exception`,
#      permanently freezing MODELS as empty for the life of the process.
#      Every downstream function (model_exists, get_recommended_model,
#      the context-window endpoints) then behaved as if zero models
#      existed, regardless of what was actually downloaded or in the DB.
#   2. Even without that bug, a frozen dict can't reflect a catalog
#      refreshed later via POST /setup/refresh-catalog without a full
#      server restart — defeating the point of a user-refreshable list.
#
# The functions below query SQLite live instead, and import `core.memory`
# LOCALLY inside each function body (not at module level) — this avoids
# the circular import entirely, regardless of which module happens to be
# imported first.

CLASSIFIER_CATEGORIES: list[str] = ["chat", "code", "reasoning", "document"]


def get_models_for_category(category: str) -> list[dict]:
    """
    Live query against the SQLite-backed catalog — always reflects the
    current state, so refreshing the catalog takes effect immediately,
    no restart required. Returns [] if the catalog hasn't been seeded
    yet, or if called before the DB is ready.
    """
    from core import memory  # local import — avoids the circular import described above
    try:
        raw = memory.get_suggested_models(category)
    except Exception:
        return []
    return [{**dict(m), "recommended": bool(m.get("recommended", 0))} for m in raw]


def get_model_config(category: str, model_id: str) -> dict | None:
    for model in get_models_for_category(category):
        if model["id"] == model_id:
            return model
    return None


def get_recommended_model(category: str) -> dict | None:
    models = get_models_for_category(category)
    recommended = [m for m in models if m.get("recommended")]
    return recommended[0] if recommended else (models[0] if models else None)


def model_path(category: str, model_id: str | None = None) -> Path:
    if model_id is None:
        cfg = get_recommended_model(category)
        if cfg is None:
            raise ValueError(f"No model found for category '{category}'")
        model_id = cfg["id"]
    cfg = get_model_config(category, model_id)
    if cfg is None:
        raise ValueError(f"Model '{model_id}' not found in category '{category}'")
    return MODELS_DIR / cfg["filename"]


def model_exists(category: str, model_id: str | None = None) -> bool:
    try:
        return model_path(category, model_id).exists()
    except ValueError:
        return False

# ── Classifier ────────────────────────────────────────────────────────────────
def _get_bundle_dir() -> Path:
    """Returns the base path for bundled assets — works both frozen and from source."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent

BUNDLE_DIR: Path = _get_bundle_dir()

CLASSIFIER_ONNX_DIR: Path = BUNDLE_DIR / "core" / "classifier_onnx"

# ── Embedding model (offline-first) ───────────────────────────────────────────
SENTENCE_TRANSFORMERS_DIR: Path = APP_DATA_DIR / "sentence_transformers"
_st_snapshots_dir = (
    SENTENCE_TRANSFORMERS_DIR
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
)

def _resolve_embedding_model() -> str:
    if _st_snapshots_dir.exists():
        snapshots = sorted(_st_snapshots_dir.iterdir())
        if snapshots:
            return str(snapshots[0])
    return "sentence-transformers/all-MiniLM-L6-v2"  # fallback, needs internet

EMBEDDING_MODEL: str = _resolve_embedding_model()

# ── RAG ───────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME   = "loqmi_docs"        # global knowledge base (unchanged)
CHROMA_CHAT_COLLECTION   = "loqmi_chat_docs"   # per-chat attachments
CHUNK_SIZE               = 1000    # chars; 512 split paragraphs too finely
CHUNK_OVERLAP            = 150
RETRIEVAL_TOP_K          = 6
RETRIEVAL_MIN_SIMILARITY = 0.25    # cosine; below this a chunk is not injected
MAX_CONTEXT_CHARS        = 12000   # legacy safety cap; real trimming is token-based

# ── Attachments ───────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES   = 50 * 1024 * 1024
MAX_PDF_PAGES      = 500
MAX_OCR_PAGES      = 40     # OCR is slow; cap per file
MIN_PAGE_CHARS     = 25     # fewer extracted chars than this -> treat page as scanned
DOC_WHOLE_FRACTION = 0.35   # inject a whole file if it fits in this share of the prompt budget

# ── Inference ─────────────────────────────────────────────────────────────────
INFERENCE_MAX_TOKENS = 1024
INFERENCE_TEMPERATURE = 0.7

# ── API server ────────────────────────────────────────────────────────────────
API_HOST = "127.0.0.1"   
API_PORT = 8765

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH: Path = APP_DATA_DIR / "chats.db"

# ── Web Search ────────────────────────────────────────────────────────────────
SEARCH_METHODS         = ["duckduckgo", "searxng", "brave", "tavily"]
DEFAULT_SEARCH_METHOD  = "duckduckgo"
SEARCH_MAX_ITERATIONS  = 5            # max tool calls per user turn
SEARCH_RESULT_COUNT    = 10            # results returned per web_search call
FETCH_MAX_CHARS        = 10000         # truncate fetched page content to this many chars
SEARXNG_INSTANCE_URL   = "https://searx.be"   # public fallback; swap for a self-hosted instance if you bundle one
SEARCH_KEY_PROVIDERS   = ["brave", "tavily"]

SEARCH_OBSERVATION_MAX_TOKENS   = 1500  # cap per tool result
SEARCH_CONTEXT_RESERVE_FRACTION = 0.35  # share of the prompt budget held back for tool results
SEARCH_MIN_ROOM_TOKENS          = 300   # below this, stop calling tools and answer

# Per-category reply caps (still limited by n_ctx; reasoning needs n_ctx >= 8192 to get 2048)
OUTPUT_TOKEN_CAPS = {"chat": 1024, "code": 2048, "reasoning": 3072, "document": 1024}

# ── Routing stability ─────────────────────────────────────────────────────────
CLASSIFIER_MIN_CONFIDENCE  = 0.60   # below this, don't trust the classifier
ROUTING_FOLLOWUP_MAX_WORDS = 4      # "continue", "why?", "fix it" -> reuse last category

# ── Context budgeting ─────────────────────────────────────────────────────────
CONTEXT_SAFETY_FRACTION          = 0.05   # headroom for chat-template tokens
CONTEXT_MIN_SAFETY_TOKENS        = 64
CONTEXT_MSG_OVERHEAD             = 12     # per-message role/template tokens
CONTEXT_MIN_OUTPUT_TOKENS        = 256
CONTEXT_OUTPUT_FRACTION          = 0.25   # reply reserve, capped at this share of n_ctx
CONTEXT_SYSTEM_MAX_FRACTION      = 0.50   # system prompt (+RAG) may use at most this share
CONTEXT_HISTORY_MSG_MAX_FRACTION = 0.35   # one old message may use at most this share  

# ── Search helper / research pipeline ─────────────────────────────────────────
HELPER_CATEGORY        = "helper"      # NOT a classifier label; never add it to CLASSIFIER_CATEGORIES
ALL_MODEL_CATEGORIES: list[str] = CLASSIFIER_CATEGORIES + [HELPER_CATEGORY]
HELPER_N_CTX           = 4096
HELPER_RAM_HEADROOM_GB = 0.4           # free RAM required beyond ~1.25x the model file size

RESEARCH_MAX_QUERIES        = 5
RESEARCH_MAX_PAGES          = 4
RESEARCH_PAGE_MAX_CHARS     = 8000
RESEARCH_CHUNK_CHARS        = 900
RESEARCH_CHUNKS_KEPT        = 6
RESEARCH_MAX_CHUNKS_PER_URL = 2
RESEARCH_NOTES_MAX_TOKENS   = 600      # helper's output
RESEARCH_NOTES_IN_CONTEXT_MAX = 700    # cap on notes inside the main model's prompt
RESEARCH_APPEND_SOURCES     = True     # append a Sources list to the saved reply

HELPER_RAM_FACTOR      = 1.15    # file size -> RAM estimate (weights + a 4k KV cache)
SEARCH_REGION          = "in-en" # ddgs region code; "wt-wt" for none
USER_REGION_NAME       = "India" # told to the planner so queries and currency fit
RESEARCH_ENOUGH_URLS   = 6       # stop issuing more queries once we have this many URLs
RESEARCH_QUERY_DELAY_S = 0.8