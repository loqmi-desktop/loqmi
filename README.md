# Loqmi 🧠

**Privacy-first desktop AI assistant using local GGUF models and RAG.**

Loqmi is a desktop application that runs Large Language Models (LLMs) entirely on your own hardware. A lightweight Next.js/Tauri frontend talks to a FastAPI/Python backend on `127.0.0.1`, so your chats, documents and screenshots stay on your machine.

## Key Features

* **Local inference:** Powered by `llama.cpp`. No API keys, no subscriptions, no cloud.
* **Smart intent routing:** An ONNX MobileBERT classifier routes each prompt to the most suitable model (Chat, Code, Reasoning or Document), with guards that keep routing stable inside a conversation.
* **Local RAG (Retrieval-Augmented Generation):** Attach PDFs and text files to a chat, or build a knowledge base, and ask questions about them using a local ChromaDB vector store. Answers show which files and pages they used.
* **Screenshot understanding:** Attach a screenshot and loqmi reads its text locally (OCR) so the model can answer questions about it.
* **Web search (optional):** A research pipeline plans queries, searches (DuckDuckGo, SearXNG, Brave or Tavily), reads the top pages and hands the notes to your model, with a sources list at the end of the answer.
* **Model manager:** Download curated models or search Hugging Face from inside the app, switch the active model per category, and tune each category's context window.

## Privacy

Everything runs on your machine. The network is used only for:

* downloading models you choose (from Hugging Face),
* a one-time download of a small (~90 MB) embedding model the first time you upload a document,
* web search, and only when you turn it on for a chat.

## Platform support

| Platform | Status |
|----------|--------|
| macOS 12+ on Apple Silicon (M1 and later) | Supported. This is the only platform tested so far |
| macOS on Intel | Not supported (builds are arm64 only) |
| Windows | Not yet. A CPU-only build is planned |
| Linux | Not yet |

## Download

Pre-built `.dmg` files for macOS (Apple Silicon) are on the [Releases](https://github.com/loqmi-desktop/loqmi/releases) page. Each release lists a SHA-256 checksum.

> These builds are **not signed or notarized** with an Apple Developer ID, so macOS Gatekeeper will warn you the first time. See [Installing an unsigned build](./frontend/README.md#installing-an-unsigned-build) for the steps.

The download is large because the Python runtime and its libraries (including PyTorch, used for document embeddings) are bundled. Models are **not** bundled: you download them on the first-run Setup screen.

### Recommended and optional extras

Nothing below is required to chat, but each one unlocks a feature:

| Extra | What it enables | How |
|-------|-----------------|-----|
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Reading text from screenshots and scanned PDFs | `brew install tesseract` |
| Hugging Face token | Gated models and faster downloads | Paste it in **Settings** |
| Brave or Tavily API key | More reliable web search than the free DuckDuckGo fallback | Paste it in **Settings** |
| Internet, once | The ~90 MB embedding model is fetched on your first document upload, then works offline | Automatic |

Rough memory needs: the default 7B to 8B models in 4-bit form want about 8 GB of unified memory (16 GB is more comfortable, especially with web search or a second helper model loaded). Larger code and reasoning models list their needs in the model catalog.

## Project Structure

This is a monorepo containing the desktop shell and the inference engine.

* [`/frontend`](./frontend/README.md): the Next.js 15 + React 19 UI wrapped in a Tauri v2 desktop shell.
* [`/backend`](./backend/README.md): the FastAPI inference server that handles model routing, RAG, web research and SSE streaming.

## Build from source

Short version (details are in the two READMEs above):

```bash
# 1. Build the backend bundle
cd backend && rm -f core/classifier_onnx/model.onnx.bak && rm -rf build dist && pyinstaller loqmi_backend.spec && cd ..

# 2. Copy it into the frontend
rm -rf frontend/src-tauri/binaries/loqmi-backend-dist && cp -R backend/dist/loqmi-backend frontend/src-tauri/binaries/loqmi-backend-dist

# 3. Build the app (use --bundles dmg for the installer)
cd frontend && pnpm install && pnpm tauri build --bundles app
```

Apps you build yourself are not quarantined by macOS, so they open without the Gatekeeper warning.

## Status

Loqmi is an early open-source release. Release builds have so far been tested on the author's Apple Silicon Mac, so reports from other setups (clean machines, different RAM sizes, different models) are very welcome. Known problems are listed at the bottom of each sub-README.

## Contributing

Issues and pull requests are welcome. For anything larger than a small fix, please open an issue first. The contributing checklists are in the [backend](./backend/README.md#contributing) and [frontend](./frontend/README.md#contributing) READMEs.

## License

Loqmi is licensed under the [MIT License](LICENSE). Downloaded models carry their own licenses, which belong to their respective authors.