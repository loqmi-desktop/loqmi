"""
Shared pytest fixtures.

The full dependency stack (llama.cpp, onnxruntime, langchain, chromadb) is
heavy to install and only needed at runtime. The test suite stubs those
packages so the real application code — routes, memory, config, routing
logic — can be exercised with just pytest + fastapi + httpx + numpy.
"""

import sys
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module


class _Dummy:
    """Accepts any constructor args, records nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Dummy()

    def __call__(self, *args, **kwargs):
        return _Dummy()


# Stub at import time (before pytest collects test modules), because the
# backend modules import these packages at module scope.
_stub_module("llama_cpp", Llama=_Dummy)

_ort = types.ModuleType("onnxruntime")
_ort.InferenceSession = _Dummy
sys.modules["onnxruntime"] = _ort

_tokenizers = types.ModuleType("tokenizers")
_tokenizers.Tokenizer = _Dummy
sys.modules["tokenizers"] = _tokenizers

_pil = types.ModuleType("PIL")
_pil_image = types.ModuleType("PIL.Image")
_pil_image.open = _Dummy
_pil.Image = _pil_image
sys.modules["PIL"] = _pil
sys.modules["PIL.Image"] = _pil_image

_stub_module("langchain_chroma", Chroma=_Dummy)
_stub_module(
    "langchain_community.document_loaders",
    PyPDFLoader=_Dummy,
    TextLoader=_Dummy,
)
_stub_module("langchain_text_splitters", RecursiveCharacterTextSplitter=_Dummy)
_stub_module("langchain_core.documents", Document=_Dummy)
_stub_module("langchain_huggingface", HuggingFaceEmbeddings=_Dummy)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the SQLite DB at a temp file for every test."""
    from core import memory

    db_path = tmp_path / "chats.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    memory.init_db()
    yield db_path
