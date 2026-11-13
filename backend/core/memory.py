"""
core/memory.py — Persistent chat memory via SQLite
Each chat has its own isolated history and RAG toggle state.
Also tracks active model selection per category.
Survives app restarts. Deleted only when user explicitly deletes a chat.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import APP_DATA_DIR

DB_PATH = APP_DATA_DIR / "chats.db"


# ── DB setup ──────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    """Context manager for SQLite connections."""
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they don't exist. Called once on app startup."""
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'New Chat',
                rag_enabled INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
            ON messages(chat_id);

            CREATE TABLE IF NOT EXISTS documents (
                filename        TEXT PRIMARY KEY,
                file_size_bytes INTEGER NOT NULL,
                chunk_count     INTEGER NOT NULL,
                upload_date     TEXT NOT NULL
            );

            -- Catalog of recommended models
            CREATE TABLE IF NOT EXISTS suggested_models (
                id              TEXT PRIMARY KEY,
                category        TEXT NOT NULL,
                name            TEXT NOT NULL,
                filename        TEXT NOT NULL,
                url             TEXT NOT NULL,
                size_gb         REAL,
                n_ctx           INTEGER NOT NULL DEFAULT 8192,
                ram_required    TEXT,
                hardware_target TEXT,
                description     TEXT,
                recommended     INTEGER NOT NULL DEFAULT 0
            );

            -- Models installed on disk
            -- Can be suggested models or Hugging Face downloads
            CREATE TABLE IF NOT EXISTS downloaded_models (
                id              TEXT PRIMARY KEY,
                category        TEXT NOT NULL,
                name            TEXT NOT NULL,
                filename        TEXT NOT NULL,
                repo_id         TEXT,
                size_gb         REAL,
                n_ctx           INTEGER NOT NULL DEFAULT 8192,
                source          TEXT NOT NULL DEFAULT 'suggested',
                downloaded_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_documents (
                chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                doc_id      TEXT NOT NULL,
                filename    TEXT NOT NULL,
                size_bytes  INTEGER NOT NULL,
                pages       INTEGER NOT NULL DEFAULT 0,
                chars       INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'processing',
                detail      TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                PRIMARY KEY (chat_id, doc_id)
            );
        """)

        # Migrations for existing databases.
        # These are safe to run every time the application starts.
        _migrate_add_column(
            con,
            "chats",
            "search_enabled",
            "INTEGER NOT NULL DEFAULT 0"
        )

        _migrate_add_column(
            con,
            "chats",
            "search_method",
            "TEXT NOT NULL DEFAULT 'duckduckgo'"
        )

        _migrate_add_column(con, "chats", "last_category", "TEXT")

def _migrate_add_column(con, table: str, column: str, decl: str) -> None:
    existing = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# ── Chat management ───────────────────────────────────────────────────────────

def create_chat(chat_id: str | None = None) -> str:
    """
    Create a new chat session.
    Generates a UUID if chat_id is not provided.
    Returns the chat_id.
    """
    if chat_id is None:
        chat_id = str(uuid.uuid4())

    now = _now()
    with _conn() as con:
        con.execute(
            "INSERT INTO chats (id, title, rag_enabled, created_at) VALUES (?, ?, 0, ?)",
            (chat_id, "New Chat", now),
        )
    return chat_id


def delete_chat(chat_id: str) -> None:
    """Delete a chat and all its messages (CASCADE handles messages)."""
    with _conn() as con:
        con.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def list_chats() -> list[dict]:
    """
    Return all chats ordered by most recently created.
    Used to populate the frontend sidebar.
    """
    with _conn() as con:
        rows = con.execute(
            "SELECT id, title, rag_enabled, search_enabled, search_method, created_at "
            "FROM chats ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def update_title(chat_id: str, title: str) -> None:
    """Set the chat title. Called after the first user message."""
    # Truncate to 60 chars for sidebar display
    title = title[:60].strip()
    with _conn() as con:
        con.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))


# ── RAG toggle ────────────────────────────────────────────────────────────

def set_rag_enabled(chat_id: str, enabled: bool) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE chats SET rag_enabled = ? WHERE id = ?",
            (1 if enabled else 0, chat_id),
        )


def get_rag_enabled(chat_id: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT rag_enabled FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"Chat '{chat_id}' does not exist.")
    return bool(row["rag_enabled"])


# ── Active model selection ────────────────────────────────────────────────────

def set_active_model(category: str, model_id: str) -> None:
    """
    Set the active model for a category (persisted in SQLite).
    Key format: "active_model:{category}" e.g. "active_model:chat"
    """
    key = f"active_model:{category}"
    with _conn() as con:
        con.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, model_id),
        )


def get_active_model(category: str) -> str | None:
    """
    Get the active model_id for a category, or None if not set.
    Falls back to recommended model if not explicitly set.
    """
    key = f"active_model:{category}"
    with _conn() as con:
        row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    
    if row is not None:
        return row["value"]
    
    # Fallback: return recommended model id for the category
    from config import get_recommended_model
    cfg = get_recommended_model(category)
    return cfg["id"] if cfg else None


# ── Context window ────────────────────────────────────────────────────────
#
# Two distinct things are cached here, under different key prefixes:
#
#   context_window:{category}       — the user's chosen n_ctx for this
#                                      category (a preference, set via the
#                                      Settings UI). None if unset, in
#                                      which case the model's own
#                                      configured default (MODELS[...]["n_ctx"])
#                                      applies.
#
#   n_ctx_train:{category}:{model_id} — the model's REAL trained max
#                                      context length, read once from GGUF
#                                      metadata (see
#                                      model_manager.get_context_ceiling)
#                                      and cached here so it's never
#                                      re-read. This is a fact about the
#                                      model file, not a preference — it
#                                      doesn't change unless the model
#                                      file itself is replaced.

def set_context_window(category: str, value: int) -> None:
    key = f"context_window:{category}"
    with _conn() as con:
        con.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )


def get_context_window(category: str) -> int | None:
    key = f"context_window:{category}"
    with _conn() as con:
        row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return int(row["value"]) if row else None


def set_cached_n_ctx_train(category: str, model_id: str, n_ctx_train: int) -> None:
    key = f"n_ctx_train:{category}:{model_id}"
    with _conn() as con:
        con.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(n_ctx_train)),
        )


def get_cached_n_ctx_train(category: str, model_id: str) -> int | None:
    key = f"n_ctx_train:{category}:{model_id}"
    with _conn() as con:
        row = con.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return int(row["value"]) if row else None


# ── Message history ───────────────────────────────────────────────────────────

def append(chat_id: str, role: str, content: str) -> None:
    """
    Append a single message to a chat's history.

    `content` is expected to already be a JSON-encoded string produced by
    the API layer (a serialized array of content blocks — see
    parse_content_blocks below). This function itself is format-agnostic:
    it just stores whatever string it's given, so older plain-text rows
    written before the block format existed keep working unmodified.
    """
    with _conn() as con:
        con.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, _now()),
        )


def get_history(chat_id: str) -> list[dict[str, str]]:
    """
    Return the full message history for a chat as a list of
    {"id": ..., "role": ..., "content": ..., "created_at": ...} dicts.

    `content` here is the raw stored string (JSON-encoded blocks for
    new-format messages, plain text for old ones) — callers that need
    structured blocks should use parse_content_blocks() on it; callers
    building a model prompt should use get_history_for_inference() below
    instead of this function directly.
    """
    with _conn() as con:
        rows = con.execute(
            "SELECT id, role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
        
    return [
        {
            "id": str(row["id"]), # Cast to string to match frontend TypeScript
            "role": row["role"], 
            "content": row["content"],
            "created_at": row["created_at"]
        } 
        for row in rows
    ]


def clear_history(chat_id: str) -> None:
    """Delete all messages for a chat but keep the chat itself."""
    with _conn() as con:
        con.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))

def chat_exists(chat_id: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
    return row is not None

def get_last_category(chat_id: str) -> str | None:
    with _conn() as con:
        row = con.execute(
            "SELECT last_category FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
    return row["last_category"] if row else None


def set_last_category(chat_id: str, category: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE chats SET last_category = ? WHERE id = ?", (category, chat_id)
        )


def count_messages(chat_id: str) -> int:
    with _conn() as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ?", (chat_id,)
        ).fetchone()
    return int(row["n"])


def delete_last_message(chat_id: str, role: str) -> None:
    """
    Delete the chat's most recent message, but only if it has the given
    role. Used to roll back a user message whose generation failed before
    any reply was produced, so no orphan is left in history.
    """
    with _conn() as con:
        con.execute(
            "DELETE FROM messages WHERE role = ? AND id = "
            "(SELECT MAX(id) FROM messages WHERE chat_id = ?)",
            (role, chat_id),
        )


# ── Content blocks ──────────────────────────────────────────────────────────
#
# A message's `content` column stores a JSON-encoded array of blocks, e.g.:
#   [{"label": "text", "content": "what's in this screenshot?"},
#    {"label": "image", "msg": "Shows a Python traceback...", "svg": "data:...", "attachment_id": "a1b2c3"}]
#
# Image blocks never carry raw OCR text — only the LLM-generated summary
# (`msg`), which is what gets replayed into future turns. The raw text is
# used once (by the API layer, for the turn the screenshot was attached
# on) and is never persisted anywhere.
#
# Rows written before this format existed are just a plain string — those
# are treated as a single implicit text block so both formats read the
# same way through the functions below.

def parse_content_blocks(content: str) -> list[dict]:
    """
    Parse a message's stored `content` string into a list of content
    blocks. Falls back to wrapping the raw string as a single text block
    if it isn't valid JSON in the expected shape (covers pre-block-format
    rows transparently).
    """
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and all(
            isinstance(b, dict) and "label" in b for b in parsed
        ):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return [{"label": "text", "content": content}]


def _flatten_blocks_for_inference(blocks: list[dict]) -> str:
    """
    Turn a message's content blocks into the plain text fed to the model.
    Image blocks contribute their `msg` summary only — never raw OCR
    text, which keeps a chat with several screenshots from ballooning
    the context window on every subsequent turn. Text blocks pass
    through as-is. Blocks are joined with double newlines.
    """
    parts = []
    for block in blocks:
        if block.get("label") == "image":
            summary = (block.get("msg") or "").strip()
            if summary:
                parts.append(f"[Screenshot: {summary}]")
        elif block.get("label") == "file":
            name = (block.get("filename") or "").strip()
            if name:
                parts.append(f"[Attached file: {name}]")
        else:
            text = (block.get("content") or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def get_history_for_inference(chat_id: str) -> list[dict[str, str]]:
    """
    Like get_history(), but returns {"role", "content"} pairs with
    `content` flattened to plain text suitable for building a prompt.
    This is what the inference/prompt-building code should call instead
    of get_history() — it's the function that actually keeps replayed
    screenshot history compact.
    """
    history = get_history(chat_id)
    result = []
    for msg in history:
        blocks = parse_content_blocks(msg["content"])
        flat = _flatten_blocks_for_inference(blocks)
        if flat:  # skip messages that flatten to nothing (e.g. an image with no summary yet)
            result.append({"role": msg["role"], "content": flat})
    return result


# ── Internal ──────────────────────────────────────────────────────────────────

def add_document(filename: str, size_bytes: int, chunks: int) -> None:
    """Track a newly ingested document in SQLite."""
    with _conn() as con:
        con.execute(
            """
            INSERT INTO documents (filename, file_size_bytes, chunk_count, upload_date)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET 
                file_size_bytes = excluded.file_size_bytes,
                chunk_count = excluded.chunk_count,
                upload_date = excluded.upload_date
            """,
            (filename, size_bytes, chunks, _now()),
        )

def remove_document_record(filename: str) -> None:
    """Remove a document's metadata from SQLite."""
    with _conn() as con:
        con.execute("DELETE FROM documents WHERE filename = ?", (filename,))

def list_documents_metadata() -> list[dict]:
    """Return all document metadata for the UI table."""
    with _conn() as con:
        rows = con.execute(
            "SELECT filename, file_size_bytes, chunk_count, upload_date "
            "FROM documents ORDER BY upload_date DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def set_hf_token(token: str) -> None:
    """Save the Hugging Face API token."""
    with _conn() as con:
        con.execute(
            "INSERT INTO settings (key, value) VALUES ('hf_token', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (token,),
        )

def get_hf_token() -> str | None:
    """Retrieve the Hugging Face API token if one is set."""
    with _conn() as con:
        row = con.execute("SELECT value FROM settings WHERE key = 'hf_token'").fetchone()
    return row["value"] if row else None

# ── Search toggle ─────────────────────────────────────────────────────────

def set_search_settings(chat_id: str, enabled: bool, method: str | None = None) -> None:
    with _conn() as con:
        if method is not None:
            con.execute(
                "UPDATE chats SET search_enabled = ?, search_method = ? WHERE id = ?",
                (1 if enabled else 0, method, chat_id),
            )
        else:
            con.execute(
                "UPDATE chats SET search_enabled = ? WHERE id = ?",
                (1 if enabled else 0, chat_id),
            )


def get_search_settings(chat_id: str) -> dict:
    with _conn() as con:
        row = con.execute(
            "SELECT search_enabled, search_method FROM chats WHERE id = ?", (chat_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"Chat '{chat_id}' does not exist.")
    return {"enabled": bool(row["search_enabled"]), "method": row["search_method"]}

def set_search_key(provider: str, key: str) -> None:
    db_key = f"search_key:{provider}"
    with _conn() as con:
        con.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (db_key, key),
        )


def get_search_key(provider: str) -> str | None:
    db_key = f"search_key:{provider}"
    with _conn() as con:
        row = con.execute("SELECT value FROM settings WHERE key = ?", (db_key,)).fetchone()
    return row["value"] if row else None

# ── Suggested Models (Remote Catalog) ─────────────────────────────────────────

def sync_suggested_models(models_list: list[dict]) -> None:
    """
    Clears old catalog and inserts the freshly fetched catalog from GitHub.
    """
    with _conn() as con:
        con.execute("DELETE FROM suggested_models")
        for m in models_list:
            con.execute(
                """
                INSERT INTO suggested_models 
                (id, category, name, filename, url, size_gb, n_ctx, ram_required, hardware_target, description, recommended)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    m["id"],
                    m["category"],
                    m["name"],
                    m["filename"],
                    m["url"],
                    m.get("size_gb"),
                    m.get("n_ctx", 8192),
                    m.get("ram_required"),
                    m.get("hardware_target"),
                    m.get("description"),
                    1 if m.get("recommended") else 0,
                ),
            )


def get_suggested_models(category: str | None = None) -> list[dict]:
    """Retrieve the suggested catalog, optionally filtered by category."""
    with _conn() as con:
        if category:
            rows = con.execute(
                "SELECT * FROM suggested_models WHERE category = ?", (category,)
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM suggested_models").fetchall()
    return [dict(row) for row in rows]


# ── Downloaded Models (Local Storage) ─────────────────────────────────────────

def add_downloaded_model(model_data: dict) -> None:
    """Register a newly downloaded model (from GitHub list or Hugging Face)."""
    with _conn() as con:
        con.execute(
            """
            INSERT INTO downloaded_models 
            (id, category, name, filename, repo_id, size_gb, n_ctx, source, downloaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                category = excluded.category,
                name = excluded.name,
                downloaded_at = excluded.downloaded_at
            """,
            (
                model_data["id"],
                model_data["category"],
                model_data["name"],
                model_data["filename"],
                model_data.get("repo_id"),
                model_data.get("size_gb"),
                model_data.get("n_ctx", 8192),
                model_data.get("source", "suggested"),
                _now(),
            ),
        )


def remove_downloaded_model(model_id: str) -> None:
    """Remove a model from the downloaded table when deleted from disk."""
    with _conn() as con:
        con.execute("DELETE FROM downloaded_models WHERE id = ?", (model_id,))


def get_downloaded_models(category: str | None = None) -> list[dict]:
    """List all models currently installed and available for generation."""
    with _conn() as con:
        if category:
            rows = con.execute(
                "SELECT * FROM downloaded_models WHERE category = ?", (category,)
            ).fetchall()
        else:
            rows = con.execute("SELECT * FROM downloaded_models").fetchall()
    return [dict(row) for row in rows]


# ── Chat-scoped documents ─────────────────────────────────────────────────────

_CHAT_DOC_COLS = "doc_id, filename, size_bytes, pages, chars, status, detail, created_at"

def add_chat_document(chat_id: str, doc_id: str, filename: str, size_bytes: int) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO chat_documents (chat_id, doc_id, filename, size_bytes, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(chat_id, doc_id) DO UPDATE SET status='processing', detail='', "
            "filename=excluded.filename",
            (chat_id, doc_id, filename, size_bytes, _now()),
        )

def update_chat_document(chat_id: str, doc_id: str, **fields) -> None:
    allowed = {"status", "detail", "pages", "chars"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as con:
        con.execute(
            f"UPDATE chat_documents SET {sets} WHERE chat_id = ? AND doc_id = ?",
            (*fields.values(), chat_id, doc_id),
        )

def get_chat_document(chat_id: str, doc_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            f"SELECT {_CHAT_DOC_COLS} FROM chat_documents WHERE chat_id = ? AND doc_id = ?",
            (chat_id, doc_id),
        ).fetchone()
    return dict(row) if row else None

def list_chat_documents(chat_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            f"SELECT {_CHAT_DOC_COLS} FROM chat_documents WHERE chat_id = ? ORDER BY created_at",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def remove_chat_document(chat_id: str, doc_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM chat_documents WHERE chat_id = ? AND doc_id = ?", (chat_id, doc_id))

def count_doc_refs(doc_id: str) -> int:
    with _conn() as con:
        row = con.execute("SELECT COUNT(*) AS n FROM chat_documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return int(row["n"])