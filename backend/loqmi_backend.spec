# loqmi_backend.spec
# Run with: pyinstaller loqmi_backend.spec

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(SPECPATH)
# Resolve site-packages from the *current* interpreter instead of a hardcoded
# path, so the build works in any venv/conda env. Override by exporting
# loqmi_SITE_PACKAGES if collect_all misses something in exotic setups.
if os.environ.get("loqmi_SITE_PACKAGES"):
    SITE_PACKAGES = Path(os.environ["loqmi_SITE_PACKAGES"])
else:
    import sysconfig
    SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])
LLAMA_CPP_LIB = SITE_PACKAGES / "llama_cpp" / "lib"
ONNXRUNTIME_CAPI = SITE_PACKAGES / "onnxruntime" / "capi"

# ── Collect packages that need special handling ───────────────────────────────
datas = []
binaries = []
hiddenimports = []

# llama_cpp — collect everything including metal dylibs
llama_datas, llama_binaries, llama_hidden = collect_all("llama_cpp")
datas += llama_datas
binaries += llama_binaries
hiddenimports += llama_hidden

# chromadb — has dynamic imports for its backends
chroma_datas, chroma_binaries, chroma_hidden = collect_all("chromadb")
datas += chroma_datas
binaries += chroma_binaries
hiddenimports += chroma_hidden

# onnxruntime
onnx_datas, onnx_binaries, onnx_hidden = collect_all("onnxruntime")
datas += onnx_datas
binaries += onnx_binaries
hiddenimports += onnx_hidden

# sentence_transformers + transformers
for pkg in ["sentence_transformers", "tokenizers", "huggingface_hub"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# langchain stack
for pkg in ["langchain", "langchain_community", "langchain_chroma", "langchain_huggingface", "langchain_core", "langchain_text_splitters"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# fastapi + uvicorn
for pkg in ["fastapi", "uvicorn", "starlette", "anyio"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# web search stack
for pkg in ["ddgs", "primp", "lxml", "trafilatura", "courlan", "htmldate", "justext", "bs4"]:
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d; binaries += b; hiddenimports += h

# ── Explicit dylib binaries (llama_cpp Metal stack) ───────────────────────────
llama_dylibs = [
    "libllama.dylib",
    "libllama.0.dylib",
    "libggml.dylib",
    "libggml.0.dylib",
    "libggml-base.dylib",
    "libggml-base.0.dylib",
    "libggml-metal.dylib",
    "libggml-metal.0.dylib",
    "libggml-cpu.dylib",
    "libggml-cpu.0.dylib",
    "libggml-blas.dylib",
    "libggml-blas.0.dylib",
    "libmtmd.dylib",
    "libmtmd.0.dylib",
]
for dylib in llama_dylibs:
    full_path = LLAMA_CPP_LIB / dylib
    if full_path.exists():
        binaries.append((str(full_path), "llama_cpp/lib"))

# onnxruntime native libs
binaries += [
    (str(ONNXRUNTIME_CAPI / "libonnxruntime.1.26.0.dylib"), "onnxruntime/capi"),
    (str(ONNXRUNTIME_CAPI / "onnxruntime_pybind11_state.so"), "onnxruntime/capi"),
]

# ── Your backend source files ─────────────────────────────────────────────────
datas += [
    # classifier ONNX model files
    (str(BACKEND_DIR / "core" / "classifier_onnx"), "core/classifier_onnx"),
    # all Python source (api, core, rag modules)
    (str(BACKEND_DIR / "api"), "api"),
    (str(BACKEND_DIR / "core"), "core"),
    (str(BACKEND_DIR / "rag"), "rag"),
    (str(BACKEND_DIR / "config.py"), "."),
    (str(BACKEND_DIR / "model"), "model"),
]

# ── Hidden imports your code uses dynamically ─────────────────────────────────
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "chromadb.api.segment",
    "chromadb.segment.impl.manager.local",
    "chromadb.segment.impl.vector.local_persistent_hnsw",
    "chromadb.segment.impl.metadata.sqlite",
    "chromadb.telemetry.product.posthog",
    "langchain_community.document_loaders.pdf",
    "langchain_community.document_loaders.text",
    "PIL._tkinter_finder",
    "pytesseract",
    "sqlite3",
    "aiosqlite",
    "multipart",
    "python_multipart",
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [str(BACKEND_DIR / "server_entry.py")],
    pathex=[str(BACKEND_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "notebook",
        "jupyter",
        "IPython",
        "pytest",
        "test",
        "tests",
        "transformers",
        "torch",
        "optimum",  
        "sklearn",
        "scipy",
        "transformers.generation",
        "pandas",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="loqmi-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX breaks Metal dylibs — never use on macOS
    console=True,       # keep True so logs are visible during dev
    codesign_identity=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="loqmi-backend",
)