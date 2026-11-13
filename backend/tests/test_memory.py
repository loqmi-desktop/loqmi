"""Tests for the SQLite persistence layer (core/memory.py)."""

from core import memory


def test_create_and_list_chats():
    chat_id = memory.create_chat()
    chats = memory.list_chats()
    assert len(chats) == 1
    assert chats[0]["id"] == chat_id


def test_delete_chat_cascades_messages():
    chat_id = memory.create_chat()
    memory.append(chat_id, "user", "hello")
    memory.append(chat_id, "assistant", "hi")

    memory.delete_chat(chat_id)

    assert not memory.chat_exists(chat_id)
    assert memory.get_history(chat_id) == []


def test_history_roundtrip_and_order():
    chat_id = memory.create_chat()
    memory.append(chat_id, "user", "first")
    memory.append(chat_id, "assistant", "second")
    memory.append(chat_id, "user", "third")

    history = memory.get_history(chat_id)
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "first"),
        ("assistant", "second"),
        ("user", "third"),
    ]


def test_clear_history_keeps_chat():
    chat_id = memory.create_chat()
    memory.append(chat_id, "user", "hello")

    memory.clear_history(chat_id)

    assert memory.chat_exists(chat_id)
    assert memory.get_history(chat_id) == []


def test_update_title():
    chat_id = memory.create_chat()
    memory.update_title(chat_id, "My Chat")
    chats = memory.list_chats()
    assert chats[0]["title"] == "My Chat"


def test_rag_toggle_defaults_off():
    chat_id = memory.create_chat()
    assert memory.get_rag_enabled(chat_id) is False

    memory.set_rag_enabled(chat_id, True)
    assert memory.get_rag_enabled(chat_id) is True


def test_search_settings_default():
    chat_id = memory.create_chat()
    settings = memory.get_search_settings(chat_id)
    assert settings["enabled"] is False
    assert settings["method"] in (None, "duckduckgo")


def test_search_settings_roundtrip():
    chat_id = memory.create_chat()
    memory.set_search_settings(chat_id, True, "brave")
    settings = memory.get_search_settings(chat_id)
    assert settings["enabled"] is True
    assert settings["method"] == "brave"


def test_active_model_roundtrip():
    from config import get_recommended_model

    # Unset active model falls back to the category's recommended model
    recommended_id = get_recommended_model("chat")["id"]
    assert memory.get_active_model("chat") == recommended_id

    memory.set_active_model("chat", "qwen3-8b-q4")
    assert memory.get_active_model("chat") == "qwen3-8b-q4"


def test_document_metadata_roundtrip():
    memory.add_document("report.pdf", 12345, 42)
    docs = memory.list_documents_metadata()

    assert len(docs) == 1
    assert docs[0]["filename"] == "report.pdf"
    assert docs[0]["file_size_bytes"] == 12345
    assert docs[0]["chunk_count"] == 42
    assert docs[0]["upload_date"]

    memory.remove_document_record("report.pdf")
    assert memory.list_documents_metadata() == []


def test_token_storage_roundtrip():
    assert memory.get_hf_token() is None
    memory.set_hf_token("hf_test_token")
    assert memory.get_hf_token() == "hf_test_token"
