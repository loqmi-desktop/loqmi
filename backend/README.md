# loqmi Backend

FastAPI server that powers **loqmi**, a privacy-first desktop AI assistant. It runs local GGUF models through llama.cpp, routes each prompt to the right model by intent, does RAG over your documents with ChromaDB, optionally researches the web, and exposes everything over a local HTTP + SSE API on `127.0.0.1:8765`.

Nothing leaves your machine unless you explicitly turn on web search (or download a model).

> The desktop shell and UI live in the frontend folder: see the [frontend README](../frontend/README.md).

## Table of contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Running in development](#running-in-development)
- [API reference](#api-reference)
- [The intent classifier](#the-intent-classifier)
- [Web research](#web-research)
- [Model catalog and downloads](#model-catalog-and-downloads)
- [Chat template compatibility](#chat-template-compatibility)
- [Embeddings and OCR](#embeddings-and-ocr)
- [Process lifecycle](#process-lifecycle)
- [Packaging](#packaging)
- [Debugging the packaged backend](#debugging-the-packaged-backend)
- [Data and storage](#data-and-storage)
- [Platform support](#platform-support)
- [Upgrading](#upgrading)
- [Known issues](#known-issues)
- [Contributing](#contributing)
- [License](#license)

## How it works

```
request -> intent classifier (MobileBERT, ONNX)
        -> routing guards (sticky category, low-confidence fallback)
        -> router picks a model category: chat | code | reasoning | document
        -> optional web research (plan -> search -> fetch -> rank -> notes)
        -> optional file context (attached files and knowledge base,
                                  ChromaDB + sentence-transformers)
        -> context budgeting (history, files and notes fitted to the window)
        -> llama.cpp streams tokens
        -> SSE stream to the client
```

Four routable categories are supported: **chat**, **code**, **reasoning** and **document**. Each category has its own system prompt and an active model chosen by the user. A fifth, non-routable category, **helper**, holds small models that can plan searches and write research notes (see [Web research](#web-research)).

If a category's model has not been downloaded, the model manager falls back to the chat model. Requests that used to be a separate "translation" category (for example "translate this to French") are handled by the chat model.

Routing guards (`core/router.py`) sit on top of the classifier:

- Short follow-ups ("continue", "why?") and low-confidence predictions reuse the chat's previous category.
- The `document` category is only used when a document is actually in play (an attached file, RAG on, or an earlier document turn). Otherwise the prompt goes to `chat`.
- A very short first message ("hello") is sent to `chat`.

## Project structure

```
backend/
  api/
    server.py              FastAPI app, CORS, startup
    routes/
      attachements.py      Attachments route (note spelling in filename)
      chat_documents.py    Chat-scoped document routes
      chats.py             Chat session routes
      context_window.py    Context window inspection/management routes
      documents.py         Document management routes
      setup.py             Initial setup / onboarding routes
  core/
    classifier.py          ONNX intent classifier (MobileBERT)
    classifier_onnx/       Shipped ONNX model + tokenizer files
    context_budget.py      Token budgeting: system prompt, history, prompt, output room
    doc_context.py         Builds file context (whole file or retrieved chunks)
    inference.py           Prompts, research phase, file context, streaming
    lifecycle.py           Parent-process watchdog
    llm_compat.py          Chat-template fallback (system role folded into user turn)
    memory.py              SQLite: chats, messages, settings, catalog, documents
    model_manager.py       Loads GGUF models with lock guarding
    ocr_support.py         Locates Tesseract and reports OCR availability
    research.py            Deterministic web-research pipeline
    router.py              Classify -> guards -> load model -> stream inference -> stats
    utils.py               Model category guessing and shared helpers
    web_search.py          Search providers and page fetching
  rag/
    __init__.py            Exports ingestion helpers (from rag.ingest) and retrieve_context
    embeddings.py          Embedding backend
    ingest.py              Extraction + indexing for PDFs, text files, and images
    ingestor.py            Additional ingestion module (including PIL image handling)
    retriever.py           Scoped retrieval with relevance cutoff and citations
  scripts/
    check_catalog_urls.py  Catalog URL verification script
    cleaning.py            Dataset CSV cleaning script
    export_onnx.py         ONNX export script for intent classifier
    train_classifier.py    Intent classifier training script
  tests/
    conftest.py            Pytest fixtures
    test_api.py            API route tests
    test_config.py         Configuration tests
    test_memory.py         SQLite memory/storage tests
    test_router.py         Routing pipeline tests
    test_search_and_actions.py  Search and action tests
  model/
    model.json             Default model catalog (seed data)
  config.py                Paths, constants, catalog accessors
  loqmi_backend.spec       PyInstaller spec
  main.py                  Standalone REPL for testing the pipeline
  pytest.ini               Pytest configuration
  README.md                Backend documentation
  requirements.txt         Python dependencies
  server_entry.py          PyInstaller entry point (supports --selftest)
  test.py                  Top-level test script
```

## Requirements

- Python 3.11 (3.10+ should work)
- `llama-cpp-python` built for your platform. On Apple Silicon, build with Metal:
  ```bash
  CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python --no-cache-dir
  ```
- **Optional:** [Tesseract](https://github.com/tesseract-ocr/tesseract) for reading text from screenshots and scanned PDFs (`brew install tesseract`). The `pytesseract` Python wrapper is in `requirements.txt`, but the Tesseract program itself is not bundled. Without it, attaching a screenshot returns a clear 503 message with install instructions.
- **Optional:** a HuggingFace token (gated model downloads) and Brave/Tavily keys (web search) can be set in the app's Settings screen.
- **Internet once:** the first document upload downloads a small embedding model (`all-MiniLM-L6-v2`, about 90 MB) into the app data folder. After that, document search works offline.

## Running in development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.server:app --host 127.0.0.1 --port 8765
```

Run it without `--reload` and with a single worker if you also want to test the desktop app against it (see below).

Tip: if a backend is already listening on port 8765, the desktop app detects it and does **not** spawn its own. That lets you run `uvicorn` from source while developing the frontend with `pnpm tauri dev`.

A standalone REPL for testing the pipeline without the API:

```bash
python main.py          # commands: quit / reset / rag on / rag off
```

## API reference

All streaming endpoints use Server-Sent Events over plain `fetch` streams. Streams end with a literal `data: [DONE]` line.

### Health

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness probe, returns `{"status": "ok"}` |

### Chats

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/chats` | Create a chat session, returns `{ chat_id }` |
| GET | `/chats` | List sessions (with RAG and search settings) |
| DELETE | `/chats/{id}` | Delete a session, its messages and its attached-file vectors |
| GET | `/chats/{id}/messages` | Full history, each message's `content` is an array of content blocks |
| POST | `/chats/{id}/messages` | Send a message, SSE stream (see events below) |
| PATCH | `/chats/{id}/rag` | Toggle RAG for this chat |
| PATCH | `/chats/{id}/search` | Toggle web search and choose the provider |

`POST /chats/{id}/messages` body:

```json
{
  "content": [{ "label": "text", "content": "hello" }],
  "inference_override_text": null,
  "override_category": null
}
```

- `content` is persisted. It is an array of blocks: `text`, `image` (a summary, a thumbnail and an attachment id) or `file` (a pointer to a chat-attached document: `doc_id`, `filename`, `pages`).
- `inference_override_text` is what the model sees for this turn only (for example typed text plus a screenshot's OCR text). It is never stored.
- `override_category` skips the classifier. Valid values are `chat`, `code`, `reasoning`, `document`. Anything else returns a 400 before streaming starts.

SSE events:

| Event | Meaning |
|-------|---------|
| `token` | One piece of the reply (`text`) |
| `status` | Progress text while researching or switching models |
| `sources` | Files and pages the answer drew on |
| `context` | History or prompt was trimmed to fit the window |
| `stats` | Final routing info: category, model name, classify time, inference time |
| `error` | The turn failed (`message`) |

### Chat documents (attached files)

Files attached to a single chat. They are searchable in that chat only, until promoted.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/chats/{id}/documents` | Multipart upload. Returns 202 immediately and processes in the background |
| GET | `/chats/{id}/documents` | List the chat's attached documents |
| GET | `/chats/{id}/documents/{doc_id}` | Status: `processing`, `ready` or `failed`, with a detail string |
| DELETE | `/chats/{id}/documents/{doc_id}` | Remove an attached document |
| POST | `/chats/{id}/documents/{doc_id}/promote` | Move it into the global knowledge base |

### Documents (knowledge base / RAG)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/documents/ingest` | Multipart upload, chunk and embed into ChromaDB |
| GET | `/documents` | List indexed documents with size, chunk count and upload date |
| DELETE | `/documents/{filename}` | Remove a document and its chunks |

### Attachments

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/attachments/screenshot` | Upload an image, returns a summary, thumbnail and raw OCR text. Never touches the RAG knowledge base. Returns 503 with install instructions if Tesseract is missing |
| GET | `/attachments/{id}` | Full-resolution original |

### Setup and models

| Method | Path | Purpose |
|--------|------|---------|
| GET/POST | `/setup/token` | Get (masked) / save the HuggingFace token |
| GET/POST | `/setup/search-keys` | Get (masked) / save Brave and Tavily keys |
| GET | `/setup/status` | Per-model state for every category, including `downloading` and `progress_percent` |
| GET | `/setup/catalog` | The curated model catalog from SQLite |
| POST | `/setup/refresh-catalog` | Reload the catalog from `model/model.json` (or the hosted JSON) |
| GET | `/setup/downloaded` | Models installed locally |
| GET | `/setup/hf-search?q=` | Search GGUF repos on the HuggingFace Hub |
| GET | `/setup/hf-files?repo_id=` | List `.gguf` files in a repo |
| GET | `/setup/download` | SSE: download the recommended model of every category |
| POST | `/setup/models/{category}/{model_id}` | SSE: download one model (409 if already downloading) |
| DELETE | `/setup/models/{category}/{model_id}` | Delete weights and reset the active pointer |
| PUT | `/setup/models/{category}/select` | Set the active model for a category |

### Context window

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/context-window` | Per-category current, recommended and maximum context length |
| PUT | `/context-window/{category}` | Set the context length (clamped to the model's real maximum) |

## The intent classifier

A fine-tuned `google/mobilebert-uncased`, exported to ONNX and run with `onnxruntime` on CPU. It decides which category's model should answer.

**Important:** the order of labels must be identical in three places, or routing will be silently wrong (or raise an `IndexError`):

1. `CATEGORIES` in `scripts/train_classifier.py`
2. `CATEGORIES` in `scripts/export_onnx.py`
3. `CLASSIFIER_CATEGORIES` in `config.py`

Current order: `["chat", "code", "reasoning", "document"]`.

The classifier can be confidently wrong on very short or casual prompts (for example a greeting predicted as `document`). The rule-based guards in `core/router.py` cover the most common cases; better training data is the long-term fix.

### Retraining

Use a separate virtual environment so the heavy training dependencies stay out of your backend environment and your PyInstaller build.

```bash
python3 -m venv ~/venvs/loqmi-train
source ~/venvs/loqmi-train/bin/activate
pip install torch transformers datasets scikit-learn accelerate pandas numpy onnx onnxruntime
```

1. **Prepare the data.** `data/dataset.csv` needs `prompt` and `category` columns. Rows whose category is not in `CATEGORIES` are dropped automatically, as are duplicates and nulls.
2. **Clear old outputs** so stale checkpoints and 5-class files are not mixed in:
   ```bash
   rm -rf core/classifier_model core/classifier_onnx
   ```
3. **Train** (from `backend/`):
   ```bash
   python scripts/train_classifier.py
   ```
   It trains from the base MobileBERT checkpoint with a fresh classification head, stops early if accuracy plateaus, and prints a per-class report. Read the per-class recall, not just overall accuracy. First run needs internet to download the base model.
4. **Export to ONNX:**
   ```bash
   python scripts/export_onnx.py
   rm -f core/classifier_onnx/model.onnx.bak   # keep stale models out of the bundle
   ```
5. **Verify** that the model has the right number of outputs:
   ```bash
   python -c "import onnxruntime as ort; s=ort.InferenceSession('core/classifier_onnx/model.onnx'); print([(o.name,o.shape) for o in s.get_outputs()])"
   ```
   The last dimension must equal the number of categories.
6. **Test routing** with real prompts, especially ones that previously misrouted:
   ```bash
   python -c "from core.classifier import classify; print(classify('hey how are you'))"
   ```
7. **Rebuild the PyInstaller bundle** (see [Packaging](#packaging)).

Library note: the training script targets recent `transformers` releases. It uses `processing_class=` instead of `tokenizer=` and does not pass `logging_dir` or `use_mps_device`. On older versions you may need to adjust those arguments.

### History

The classifier originally had a fifth "translation" category. It was removed because it misrouted ordinary chat prompts. The translation rows were dropped from the training set and the model was retrained with four classes.

## Web research

When web search is on for a chat, a turn has two phases (`core/research.py`). There is no tool-calling loop.

1. **Research.** The model writes a short JSON plan (does this need the internet, and which queries?). Plain code then searches, fetches the top pages, extracts text, ranks passages with BM25 and keeps the best ones. The model writes concise, numbered research notes from those passages.
2. **Answer.** The routed model answers from the notes. It never sees raw web pages or tool syntax. A numbered **Sources** list is appended to the saved reply.

Who does the planning and note-writing:

- If a small **helper** model is installed (category `helper`, for example Qwen 3 4B Instruct), it is loaded next to your main model when RAM allows, so the main model stays resident.
- Otherwise the chat model does the research. When memory is tight, the main model is temporarily swapped out and restored afterwards.

If a search fails, the answering model is told so and answers from general knowledge, flagged as possibly out of date. Providers: DuckDuckGo (default, free), SearXNG, Brave and Tavily (API keys set in Settings). Tunables such as the number of queries and pages live in `config.py` under "Search helper / research pipeline".

## Model catalog and downloads

The catalog lives in the SQLite table `suggested_models`, not in a module-level constant. It is seeded from `model/model.json` (or a hosted JSON when that file is absent) and can be reloaded at runtime with `POST /setup/refresh-catalog`, no restart needed. The seed only runs when the table is empty, so after editing `model.json` use **Settings > Model Catalog > Refresh catalog**.

`config.py` reads the catalog live and imports `core.memory` inside each function. That is deliberate: a top-level import caused a circular import that silently froze the catalog as empty.

`model.json` is either a flat list of models or an object keyed by category. Each model needs `id`, `category`, `name`, `filename`, `url`, and optionally `size_gb`, `n_ctx`, `ram_required`, `hardware_target`, `description`, `recommended`. Entries in different categories may share one `filename` (the default chat and document models do), so they share one file on disk.

Before publishing catalog changes, check that every URL resolves and that its size matches `size_gb`:

```bash
python scripts/check_catalog_urls.py
```

**How downloads work** (`api/routes/setup.py`):

- A model streams into `<filename>.part`. Only a complete, verified file is renamed to its final name, so a half-finished download can never appear as "installed".
- Before the rename, the byte count is checked and the file must start with the `GGUF` magic bytes.
- If the client disconnects (for example you close the Settings page), the `.part` file is kept. Starting the download again resumes it with an HTTP `Range` request when the server supports that.
- `GET /setup/status` reports `downloading` and `progress_percent`, so the UI can show progress after you navigate away and come back.
- The Hugging Face search endpoints call the Hub API with `httpx` directly. The `huggingface_hub` client is not used there because offline flags set for the embedding model also block its API calls.

## Chat template compatibility

Some GGUF files ship a chat template that rejects a `system` role and fails with `Conversation roles must alternate user/assistant/user/assistant/...` (older Mistral-style templates do this). `core/llm_compat.py` wraps `create_chat_completion`: if the template raises, it retries once with the system prompt folded into the newest user message. Anything that sends a system message should call `llm_compat.create_chat_completion` instead of calling the model directly.

## Embeddings and OCR

- **Embeddings** (`rag/embeddings.py`): `sentence-transformers/all-MiniLM-L6-v2` on CPU. It loads from the app data folder first, and downloads once (about 90 MB) if it is missing. No global offline environment flags are set. If the download fails, uploads return a clear message asking the user to connect once.
- **OCR** (`core/ocr_support.py`): finds Tesseract on `PATH` and in common install locations (a GUI app launched from Finder does not inherit your shell `PATH`, so `/opt/homebrew/bin` is checked explicitly). You can override the path with the `TESSERACT_CMD` environment variable. Availability is re-checked on demand, so installing Tesseract does not need a restart.

## Process lifecycle

The backend is spawned by the desktop app, so it must never outlive it.

- **Normal quit:** the app sends SIGTERM to the backend's process group, waits up to 3 seconds, then SIGKILLs anything left. Uvicorn shuts down and the OS frees the loaded model's memory.
- **Force quit or crash:** the app cannot clean up, so `core/lifecycle.py` runs a watchdog thread. When the environment variable `loqmi_PARENT_WATCH=1` is set (the app sets it), the backend polls its parent PID, and if the parent disappears it sends itself SIGTERM, then hard-exits after 5 seconds if shutdown hangs.
- **Running by hand** (no env var): the watchdog is disabled, so `uvicorn` in a terminal behaves normally.
- **Concurrency:** one main model is loaded at a time, plus an optional helper model. `core/model_manager.py` guards loads, unloads and generation with a shared reentrant lock, so concurrent requests cannot race a model switch.
- **Persistence:** the user message is saved before generation starts and removed again if the turn fails before any reply, and any partial assistant reply is saved even if the client disconnects mid-stream.

The watchdog assumes the backend's parent is the desktop app. That holds for the PyInstaller **onedir** build. If you ever switch to onefile, the parent becomes the PyInstaller bootloader and the check needs rethinking.

## Packaging

`loqmi_backend.spec` builds the backend as a PyInstaller **onedir** bundle that the Tauri app launches.

```bash
pip install pyinstaller
rm -f core/classifier_onnx/model.onnx.bak
rm -rf build dist
pyinstaller loqmi_backend.spec
./dist/loqmi-backend/loqmi-backend --selftest   # checks the classifier and embeddings load
```

`--selftest` loads the classifier and the embedding model from inside the frozen bundle and prints `SELFTEST OK`. It needs internet on its first run (to fetch the embedding model). Run it after every spec change: missing modules in a frozen app only show up at runtime, and this finds them without building the whole desktop app.

Then copy the bundle into the frontend project:

```bash
rm -rf ../frontend/src-tauri/binaries/loqmi-backend-dist
cp -R dist/loqmi-backend ../frontend/src-tauri/binaries/loqmi-backend-dist
```

Checklist before building:

- The executable name must be `loqmi-backend`, because the app launches `loqmi-backend-dist/loqmi-backend`.
- `core/classifier_onnx/` must be included, and `core/classifier_model/` (the PyTorch checkpoint) must **not** be.
- Delete `core/classifier_onnx/model.onnx.bak` first.
- `sentence-transformers` needs `transformers` and `torch` at import time, so the spec must **not** exclude them, and it copies package metadata for the libraries they check at startup. This makes the bundle large.
- `server_entry.py` points TLS at `certifi`'s certificate bundle when frozen, and calls `multiprocessing.freeze_support()`.
- The desktop app runs the copy in `src-tauri/binaries/`, not your Python source. **Rebuild and recopy after every backend change.**

The spec currently assumes macOS (Apple Silicon, Metal) and a local conda environment. It lists the macOS `llama_cpp` dylib names and pins an `onnxruntime` dylib filename, so update that when you bump `onnxruntime`.

## Debugging the packaged backend

The desktop app starts the bundled backend itself and its output is not visible. To see logs and tracebacks, start the backend by hand first. The app notices the port is in use and talks to your copy:

```bash
pkill -f loqmi-backend
/Applications/loqmi.app/Contents/Resources/loqmi-backend-dist/loqmi-backend 2>&1 | tee ~/Desktop/loqmi-backend.log
```

Then open the app and reproduce the problem. If something works with `uvicorn` from source but fails in the app, the cause is almost always a package that is missing from the frozen bundle.

## Data and storage

All runtime state lives in an OS-appropriate app-data directory:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/loqmi` |
| Windows | `%APPDATA%/loqmi` |
| Linux | `$XDG_DATA_HOME/loqmi` (or `~/.local/share/loqmi`) |

Subdirectories and files: `models/` (GGUF weights, plus any `.part` downloads in progress), `chroma_db/` (vector store), `sentence_transformers/` (embedding model cache), `attachments/` (full-size screenshots), `chat_docs/` (files attached to chats), `chats.db` (SQLite), `logs/`.

## Platform support

Only **macOS on Apple Silicon** has been built and tested. The main work needed for other platforms:

- **Windows:** the Tauri shell's process handling is Unix-only (process groups and `killpg`), the parent watchdog relies on `getppid()` changing (it does not on Windows), the PyInstaller spec lists macOS library names, and Tesseract and a CPU `llama-cpp-python` wheel need to be sorted out.
- **Linux:** closer to working, since the process code is Unix, but untested.

## Upgrading

If you are upgrading from a version that had the translation category, no migration is required. Leftover settings keys such as `active_model:translation` and `context_window:translation`, and old rows in `downloaded_models`, are ignored. Translation previously reused the chat model files, so there is nothing to delete from disk. Old chats are unaffected because messages do not store their routed category.

If you maintain your own copy of `model.json`, remove any `translation` entries and run `POST /setup/refresh-catalog`.

## Known issues

Being worked through:

- HuggingFace token and search API keys are stored in plaintext in SQLite (local machine only, but worth knowing).
- Some model download URLs point at third-party HuggingFace mirrors rather than official repos. `scripts/check_catalog_urls.py` confirms they resolve, but they should be replaced with official sources over time.
- The classifier can be confidently wrong on short or casual prompts. Guards in `core/router.py` cover common cases; the training data needs improving.
- Tesseract is not bundled. Screenshot and scanned-PDF text recognition needs it installed separately.
- The bundle is large because PyTorch and `transformers` are packaged for the embedding model. Moving embeddings to an ONNX export would remove them.
- The PyInstaller spec is macOS-only and machine-specific.
- No automated tests yet. `test.py` is a manual smoke script.
- Release builds have been tested on one machine. Reports from clean machines are welcome.

Needs verification (carried over from earlier notes, not re-checked recently):

- `POST /documents/ingest` may run blocking embedding work on the event loop and freeze the server on large files.
- Logging is partly ad-hoc `print()` calls, including debug output in `rag/retriever.py`.

## Contributing

Issues and pull requests are welcome. Before opening a PR:

1. Open an issue first for anything larger than a small fix.
2. Keep changes focused, and update this README when behavior or endpoints change.
3. If you touch the classifier labels, update all three places listed in [The intent classifier](#the-intent-classifier) and include the classification report in the PR.
4. If you change `model/model.json`, run `python scripts/check_catalog_urls.py`.
5. If you change `loqmi_backend.spec` or add dependencies, run the packaged `--selftest` (see [Packaging](#packaging)).

## License

Loqmi is licensed under the MIT License. Note that downloaded models carry their own licenses, which belong to their authors.