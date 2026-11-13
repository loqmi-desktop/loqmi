# Loqmi 🧠

**Privacy-first desktop AI assistant using local GGUF models and RAG.**

Loqmi is a highly decoupled desktop application that allows you to run powerful Large Language Models (LLMs) entirely on your own hardware. It uses a lightweight Next.js/Tauri frontend to communicate with a robust FastAPI/Python backend, ensuring zero data leaves your machine.

## Key Features
*   **100% Local Inference:** Powered by `llama.cpp`, meaning no API keys, no subscriptions, and absolute data privacy.
*   **Smart Intent Routing:** Uses an ONNX MobileBERT classifier to automatically route your prompts to the most appropriate model (Chat, Code, Reasoning, or Document).
*   **Local RAG (Retrieval-Augmented Generation):** Drag and drop your private PDFs and documents to chat with them using a local ChromaDB vector store.
*   **Web Search:** Optional ReAct-style tool loop using DuckDuckGo, SearXNG, Brave, or Tavily to fetch live data.

## Project Structure
This is a monorepo containing both the frontend shell and the backend inference engine.

*   [`/frontend`](./frontend/README.md) - The Next.js 15 + React 19 UI wrapped in a Tauri v2 desktop shell.
*   [`/backend`](./backend/README.md) - The FastAPI inference server handling model routing, RAG, and SSE streaming.

## Download
Pre-built `.dmg` binaries for macOS (Apple Silicon) are available in the [Releases](https://github.com/YOUR-ORG/loqmi/releases) section. 

*(Note: These builds are not currently signed with an Apple Developer ID. See the [Frontend README](./frontend/README.md#installing-an-unsigned-build) for instructions on bypassing Gatekeeper).*

## License
Loqmi is licensed under the [MIT License](LICENSE). Downloaded models carry their own licenses belonging to their respective authors.