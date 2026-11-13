# loqmi Backend

FastAPI server that powers **loqmi**, a privacy-first desktop AI assistant. It runs local GGUF models through llama.cpp, routes each prompt to the right model by intent, does RAG over your documents with ChromaDB, optionally searches the web, and exposes everything over a local HTTP + SSE API on `127.0.0.1:8765`.

Nothing leaves your machine unless you explicitly turn on web search.

> The desktop shell and UI live in the frontend repo: see the [frontend README](../frontend/README.md).

## Table of contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Running in development](#running-in-development)
- [API reference](#api-reference)
- [The intent classifier](#the-intent-classifier)
- [Model catalog](#model-catalog)
- [Process lifecycle](#process-lifecycle)
- [Packaging](#packaging)
- [Data and storage](#data-and-storage)
- [Upgrading](#upgrading)
- [Known issues](#known-issues)
- [Contributing](#contributing)
- [License](#license)

## How it works

```
request -> intent classifier (MobileBERT, ONNX)
        -> router picks a model category: chat | code | reasoning | document
        -> llama.cpp streams tokens
        -> optional RAG context (ChromaDB + sentence-transformers)
        -> optional web search (DuckDuckGo / SearXNG / Brave / Tavily, ReAct-style tool loop)
        -> SSE stream to the client
```

Four categories are supported: **chat**, **code**, **reasoning** and **document**. Each category has its own system prompt and an active model chosen by the user. Requests that used to be treated as a separate "translation" category (for example "translate this to French") are now handled by the chat model.

If a category's model has not been downloaded, the model manager falls back to the chat model.

## Project structure

```
api/
  server.py            FastAPI app, CORS, startup
  routes/              chats, documents, setup, attachments, context-window
core/
  classifier.py        ONNX intent classifier (MobileBERT)
  classifier_onnx/     shipped model.onnx + tokenizer (see "The intent classifier")
  router.py            classify -> load model -> stream inference -> stats
  model_manager.py     loads one GGUF model at a time, guarded by a lock
  inference.py         prompts, history windowing, RAG, tool loop
  memory.py            SQLite: chats, messages, settings, catalog, documents
  web_search.py        search providers and page fetching
  catalog_sync.py      seeds the model catalog
  lifecycle.py         parent-process watchdog (see "Process lifecycle")
rag/                   ingestion, embeddings, retrieval
model/model.json       default model catalog (seed data)
data/dataset.csv       classifier training data
scripts/               train_classifier.py, export_onnx.py
config.py              paths, constants, catalog accessors
main.py                standalone REPL for testing the pipeline
loqmi_backend.spec PyInstaller spec
```

## Requirements

- Python 3.11 (3.10+ should work)
- `llama-cpp-python` built for your platform. On Apple Silicon, build with Metal:
  ```bash
  CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python --no-cache-dir
  ```
- API keys are optional: a HuggingFace token (gated model downloads) and Brave/Tavily keys (web search) can be set in the app's Settings screen.

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
| DELETE | `/chats/{id}` | Delete a session and its messages |
| GET | `/chats/{id}/messages` | Full history, each message's `content` is an array of content blocks |
| POST | `/chats/{id}/messages` | Send a message, SSE stream of `token`, `status`, `stats` and `error` events |
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

- `content` is persisted. It is an array of blocks: `text`, or `image` (with a summary, a thumbnail and an attachment id).
- `inference_override_text` is what the model sees for this turn only (for example typed text plus a screenshot's OCR text). It is never stored.
- `override_category` skips the classifier. Valid values are `chat`, `code`, `reasoning`, `document`.

### Documents (RAG)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/documents/ingest` | Multipart upload, chunk and embed into ChromaDB |
| GET | `/documents` | List indexed documents with size, chunk count and upload date |
| DELETE | `/documents/{filename}` | Remove a document and its chunks |

### Attachments

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/attachments/screenshot` | Upload an image, returns a summary, thumbnail and raw OCR text. Never touches the RAG knowledge base |
| GET | `/attachments/{id}` | Full-resolution original |

### Setup and models

| Method | Path | Purpose |
|--------|------|---------|
| GET/POST | `/setup/token` | Get (masked) / save the HuggingFace token |
| GET/POST | `/setup/search-keys` | Get (masked) / save Brave and Tavily keys |
| GET | `/setup/status` | Per-model download state for every category |
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

## Model catalog

The catalog lives in the SQLite table `suggested_models`, not in a module-level constant. It is seeded from `model/model.json` (or a hosted JSON when that file is absent) and can be reloaded at runtime with `POST /setup/refresh-catalog`, no restart needed.

`config.py` reads the catalog live and imports `core.memory` inside each function. That is deliberate: a top-level import caused a circular import that silently froze the catalog as empty.

`model.json` is either a flat list of models or an object keyed by category. Each model needs `id`, `category`, `name`, `filename`, `url`, and optionally `size_gb`, `n_ctx`, `ram_required`, `hardware_target`, `description`, `recommended`.

## Process lifecycle

The backend is spawned by the desktop app, so it must never outlive it.

- **Normal quit:** the app sends SIGTERM to the backend's process group, waits up to 3 seconds, then SIGKILLs anything left. Uvicorn shuts down and the OS frees the loaded model's memory.
- **Force quit or crash:** the app cannot clean up, so `core/lifecycle.py` runs a watchdog thread. When the environment variable `loqmi_PARENT_WATCH=1` is set (the app sets it), the backend polls its parent PID, and if the parent disappears it sends itself SIGTERM, then hard-exits after 5 seconds if shutdown hangs.
- **Running by hand** (no env var): the watchdog is disabled, so `uvicorn` in a terminal behaves normally.
- **Concurrency:** one model is loaded at a time. `core/model_manager.py` guards loads, unloads and generation with a shared reentrant lock, so concurrent requests cannot race a model switch.
- **Persistence:** the user message is saved before generation starts, and any partial assistant reply is saved even if the client disconnects mid-stream.

The watchdog assumes the backend's parent is the desktop app. That holds for the PyInstaller **onedir** build. If you ever switch to onefile, the parent becomes the PyInstaller bootloader and the check needs rethinking.

## Packaging

`loqmi_backend.spec` builds the backend as a PyInstaller **onedir** bundle that the Tauri app launches.

```bash
pip install pyinstaller
rm -rf build dist
pyinstaller loqmi_backend.spec
./dist/loqmi-backend/loqmi-backend      # smoke test
curl http://127.0.0.1:8765/health               # in another terminal
```

Then copy the bundle into the frontend project:

```bash
rm -rf ../frontend/src-tauri/binaries/loqmi-backend-dist
cp -R dist/loqmi-backend ../frontend/src-tauri/binaries/loqmi-backend-dist
```

Checklist before building:

- The executable name must be `loqmi-backend`, because the app launches `loqmi-backend-dist/loqmi-backend`.
- `core/classifier_onnx/` must be included, and `core/classifier_model/` (the PyTorch checkpoint) must **not** be.
- Delete `core/classifier_onnx/model.onnx.bak` first.
- The desktop app runs the copy in `src-tauri/binaries/`, not your Python source. **Rebuild and recopy after every backend change.**

The spec currently assumes macOS (Apple Silicon, Metal) and a local conda environment, and needs path cleanup before it is reproducible on other machines.

## Data and storage

All runtime state lives in an OS-appropriate app-data directory:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/loqmi` |
| Windows | `%APPDATA%/loqmi` |
| Linux | `$XDG_DATA_HOME/loqmi` (or `~/.local/share/loqmi`) |

Subdirectories and files: `models/` (GGUF weights), `chroma_db/` (vector store), `sentence_transformers/` (embedding model cache), `chats.db` (SQLite), `logs/`.

## Upgrading

If you are upgrading from a version that had the translation category, no migration is required. Leftover settings keys such as `active_model:translation` and `context_window:translation`, and old rows in `downloaded_models`, are ignored. Translation previously reused the chat model files, so there is nothing to delete from disk. Old chats are unaffected because messages do not store their routed category.

If you maintain your own copy of `model.json`, remove any `translation` entries and run `POST /setup/refresh-catalog`.

## Known issues

Being worked through ahead of the open-source release:

- HuggingFace token and search API keys are stored in plaintext in SQLite (local machine only, but worth knowing).
- Some model download URLs point at third-party HuggingFace mirrors rather than official repos. They should be replaced.
- The classifier has no confidence threshold. A low-confidence prediction should fall back to `chat`.
- `override_category` is not validated server-side. An unknown value fails deep inside the model manager instead of returning a clear 400.
- No automated tests yet. `test.py` is a manual smoke script.
- The PyInstaller spec is machine-specific, and the bundle is large (the full app is about 1.3 GB). Check that training dependencies are not leaking in.

Needs verification (carried over from earlier notes, not re-checked recently):

- `POST /documents/ingest` may run blocking embedding work on the event loop and freeze the server on large files.
- `pytesseract` may be missing from `requirements.txt`, which would break OCR ingestion at runtime.
- Logging is partly ad-hoc `print()` calls, including debug output in `rag/retriever.py`.

## Contributing

Issues and pull requests are welcome. Before opening a PR:

1. Open an issue first for anything larger than a small fix.
2. Keep changes focused, and update this README when behavior or endpoints change.
3. If you touch the classifier labels, update all three places listed in [The intent classifier](#the-intent-classifier) and include the classification report in the PR.

## License

Loqmi is licensed under the MIT License. Note that downloaded models carry their own licenses, which belong to their authors.