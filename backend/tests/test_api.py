"""API-level tests using FastAPI's TestClient with mocked inference."""

import pytest
from fastapi.testclient import TestClient

from api.server import app
from api.routes import documents as documents_routes


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_crud_flow(client):
    created = client.post("/chats")
    assert created.status_code == 200
    chat_id = created.json()["chat_id"]

    listed = client.get("/chats")
    assert listed.status_code == 200
    assert any(c["id"] == chat_id for c in listed.json())

    history = client.get(f"/chats/{chat_id}/messages")
    assert history.status_code == 200
    assert history.json() == []

    deleted = client.delete(f"/chats/{chat_id}")
    assert deleted.status_code == 204
    assert client.get(f"/chats/{chat_id}/messages").status_code == 404


def test_send_message_streams_sse(client, monkeypatch):
    from api.routes import chats as chats_routes

    chat_id = client.post("/chats").json()["chat_id"]

    def fake_handle(prompt, chat_id, override_category=None):
        yield {"type": "token", "text": "hello"}
        yield {"type": "stats", "category": "chat", "model_name": "Qwen 3 8B",
               "classify_ms": 1.0, "inference_s": 0.5}

    monkeypatch.setattr(chats_routes.core_router, "handle", fake_handle)

    resp = client.post(f"/chats/{chat_id}/messages", json={"content": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = resp.text
    assert '"type": "token"' in body or '"type":"token"' in body
    assert '"type": "stats"' in body or '"type":"stats"' in body
    assert "[DONE]" in body


def test_rag_toggle(client):
    chat_id = client.post("/chats").json()["chat_id"]

    resp = client.patch(f"/chats/{chat_id}/rag", json={"enabled": True})
    assert resp.status_code == 204

    from core import memory
    assert memory.get_rag_enabled(chat_id) is True


def test_search_toggle(client):
    chat_id = client.post("/chats").json()["chat_id"]

    resp = client.patch(
        f"/chats/{chat_id}/search", json={"enabled": True, "method": "brave"}
    )
    assert resp.status_code == 204

    from core import memory
    settings = memory.get_search_settings(chat_id)
    assert settings["enabled"] is True
    assert settings["method"] == "brave"


def test_unknown_chat_returns_404(client):
    assert client.get("/chats/nope/messages").status_code == 404
    assert client.delete("/chats/nope").status_code == 404


def test_list_documents(client):
    from core import memory

    memory.add_document("a.pdf", 100, 5)
    memory.add_document("b.md", 200, 10)

    resp = client.get("/documents")
    assert resp.status_code == 200
    filenames = {d["filename"] for d in resp.json()["documents"]}
    assert filenames == {"a.pdf", "b.md"}


def test_ingest_document(client, monkeypatch):
    def fake_ingest_file(path):
        return (7, "extracted text")

    monkeypatch.setattr(documents_routes, "ingest_file", fake_ingest_file)

    resp = client.post(
        "/documents/ingest",
        files={"file": ("notes.txt", b"some content", "text/plain")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["filename"] == "notes.txt"
    assert payload["chunks_stored"] == 7

    docs = client.get("/documents").json()["documents"]
    assert any(d["filename"] == "notes.txt" for d in docs)


def test_ingest_rejects_bad_extension(client):
    resp = client.post(
        "/documents/ingest",
        files={"file": ("malware.exe", b"binary", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_ingest_rejects_path_traversal(client, monkeypatch):
    def fail_ingest(path):
        raise AssertionError("ingest_file should not be called for a sanitized name")

    monkeypatch.setattr(documents_routes, "ingest_file", fail_ingest)

    resp = client.post(
        "/documents/ingest",
        files={"file": ("../../etc/passwd", b"x", "text/plain")},
    )
    assert resp.status_code == 415  # .passwd is not an allowed extension
