# loqmi Frontend

Next.js 15 + React 19 + Tailwind CSS 4 UI for **loqmi**, a privacy-first desktop AI assistant, shipped inside a [Tauri v2](https://tauri.app) shell. All inference happens in the local backend. The frontend is a pure client for it.

> The Python backend lives in its own folder: see the [backend README](../backend/README.md) for the API reference.

## Table of contents

- [Features](#features)
- [Structure](#structure)
- [Prerequisites](#prerequisites)
- [Running in development](#running-in-development)
- [Desktop app behavior](#desktop-app-behavior)
- [Building](#building)
- [Installing an unsigned build](#installing-an-unsigned-build)
- [Distributing](#distributing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)
- [Contributing](#contributing)
- [License](#license)

## Features

1. **Setup** (`/`): first-run model download. Lists the model catalog grouped by category (chat, code, reasoning, document) and streams per-model download progress.
2. **Chat** (`/chat?id=...`): collapsible sidebar with chat history, message area with expandable routing badges (which model answered, plus token stats), a composer with attachments, and a slide-out documents panel. Header toggles for RAG and web search, and a routing-override and search-provider dropdown in the composer.
3. **Documents** (`/documents`): knowledge base with drag-and-drop upload, a document table (size, chunks, upload date) and bulk delete.
4. **Settings** (`/settings`): HuggingFace token manager, Brave and Tavily API keys, model catalog (download, activate, delete per category) and per-category context-window controls.

Screenshots attached in the composer go through a separate attachments endpoint and never enter the RAG knowledge base.

## Structure

```
app/                  pages (setup, chat, documents, settings) and layouts
components/           sidebar, chat-interface, composer, chat-header,
                      chat-message, documents-panel, error-banner, progress-bar
lib/api.ts            all backend types plus fetch and SSE plumbing
lib/connection.tsx    global backend connection status
lib/chat-context.tsx  per-chat RAG and web-search state
src-tauri/            Tauri shell (Rust); launches and supervises the backend
  src/lib.rs          window handling, backend spawn and shutdown
  tauri.conf.json     app config and bundled resources
  binaries/           the PyInstaller backend bundle (not committed, see below)
```

## Prerequisites

| Tool | Notes |
|------|-------|
| Node.js and [pnpm](https://pnpm.io) | `pnpm install` also installs `@tauri-apps/cli` as a dev dependency |
| Rust (stable) | Install with [rustup](https://rustup.rs). If `cargo` is not found, run `source "$HOME/.cargo/env"` |
| Xcode Command Line Tools | `xcode-select --install` (macOS) |
| The backend bundle | Built from `../backend`, see below |

The Rust side depends on the `libc` crate (used to signal the backend's process group). It is declared in `src-tauri/Cargo.toml`.

## Running in development

```bash
pnpm install
pnpm dev                # UI only, http://localhost:3000
```

The UI degrades gracefully with inline errors when the backend is not running.

For the full desktop experience:

```bash
pnpm tauri dev
```

`tauri.conf.json` sets `beforeDevCommand` to `pnpm dev`, so this starts Next.js and the desktop window together.

**Which backend does it use?** In debug builds the app launches the bundle from `src-tauri/binaries/loqmi-backend-dist/` (in dev, Tauri's resource directory is not populated). Two ways to provide one:

- Build the bundle and copy it in, as described in the [backend README](../backend/README.md#packaging).
- Or run the backend from source (`uvicorn api.server:app --host 127.0.0.1 --port 8765`) before starting the app. If port 8765 is already in use, the app logs a warning and does not spawn its own.

## Desktop app behavior

The shell follows macOS conventions and supervises the backend so it never lingers.

| Action | What happens |
|--------|--------------|
| Click the red close button | The window hides. The app stays in the Dock and the backend keeps running |
| Click the Dock icon | The hidden window comes back |
| Cmd+Q, or Dock > Quit | The backend gets SIGTERM (to its whole process group), 3 seconds to exit, then SIGKILL. Its memory, including any loaded model, is freed |
| Force quit or crash of the app | The backend's parent-process watchdog notices within about a second and shuts itself down |
| Port 8765 already in use | The app does not spawn a second backend, and logs a warning |

Notes:

- The backend is started in its own process group, with `loqmi_PARENT_WATCH=1` set so its watchdog is active.
- Closing the window does **not** unload the model. Use Cmd+Q to free memory.
- If you suspect a stale backend (for example after a crash during development):
  ```bash
  lsof -i :8765
  pkill -f loqmi-backend
  ```

## Building

```bash
pnpm tauri build --bundles app
```

`beforeBuildCommand` runs `pnpm build` (a static export to `out/`), so the frontend is always fresh. The type-checker runs as part of that build, so type errors fail the build.

To create a DMG:

```bash
pnpm tauri build --bundles dmg
```

Output is under `src-tauri/target/release/bundle/`. In a release build the backend is read from `Contents/Resources/loqmi-backend-dist/loqmi-backend` inside the app.

Things to know:

- The backend bundle must exist in `src-tauri/binaries/loqmi-backend-dist/` before building. It is mapped into the app by the `bundle.resources` entry in `tauri.conf.json`.
- Builds are **arm64** (Apple Silicon) when made on an Apple Silicon Mac. Intel Macs cannot run them.
- The app is large (about 1.3 GB) because the Python runtime and libraries are bundled. Models are **not** bundled. Users download them on the Setup screen.
- The minimum supported macOS version is 12.0.
- Do not chain `open` after the build with `;`. Use `&&` so a failed build does not launch a stale app.

## Installing an unsigned build

loqmi is open source, and release builds are **not signed or notarized with an Apple Developer ID** (that requires a paid Apple Developer Program membership). On Apple Silicon, Tauri applies an ad-hoc signature so the binary can run, but macOS Gatekeeper still treats a downloaded copy as coming from an unidentified developer.

Users have two options:

**Option 1: build from source.** Apps you build yourself are not quarantined, so they open normally. Follow [Building](#building).

**Option 2: allow the downloaded app.**

1. Drag `loqmi.app` to `/Applications`.
2. Try to open it. macOS will refuse.
3. Open **System Settings > Privacy & Security**, scroll to the security section and click **Open Anyway** next to the loqmi message. (On older macOS versions, right-click the app and choose **Open** instead.)
4. Alternatively, remove the quarantine flag from a terminal:
   ```bash
   xattr -dr com.apple.quarantine /Applications/loqmi.app
   ```

Only do this for builds you trust. Verify the SHA-256 checksum published with each release:

```bash
shasum -a 256 loqmi_*.dmg
```

## Distributing

For an open-source release without a paid developer account:

1. Build the DMG on a clean checkout, and test it on a Mac that has no Python, conda or models installed (or a fresh user account).
2. Publish it on **GitHub Releases** (free, and files up to 2 GB are allowed), together with its SHA-256 checksum.
3. Put the install steps from [Installing an unsigned build](#installing-an-unsigned-build) on your download page and in the release notes.
4. State the requirements: Apple Silicon, macOS 12 or later, and the disk and RAM needs of the models users choose.
5. Include third-party notices. Downloaded models carry their own licenses, which belong to their authors.

If the project grows, signing and notarizing with an Apple Developer ID removes the Gatekeeper prompts. Nested binaries inside the backend bundle also need to be signed for notarization to pass.

## Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Backend URL | `NEXT_PUBLIC_loqmi_API_URL` (for example in `.env.local`) | `http://127.0.0.1:8765` |
| Dev server URL | `build.devUrl` in `tauri.conf.json` | `http://localhost:3000` |
| Bundle identifier | `identifier` in `tauri.conf.json` | Must not end in `.app` (macOS reserves that extension) |
| Backend resource path | `bundle.resources` in `tauri.conf.json` | `binaries/loqmi-backend-dist` -> `loqmi-backend-dist` |

The backend allows the Tauri origins (`tauri://localhost`, `http://localhost:1420`) plus `http://localhost:3000` for dev.

Backend connection details: requests use plain `fetch`. SSE streams are read with `fetch` and `ReadableStream` (not `EventSource`, because several endpoints stream over POST) and use a 120-second **idle** timeout that resets on every chunk, so a hung stream fails with a clear message while a slow first token (model load) does not.

## Troubleshooting

| Symptom | Likely cause and fix |
|---------|----------------------|
| `Command "tauri" not found` | The CLI is a dev dependency. Run `pnpm install` in the project root, not inside `src-tauri/` |
| `failed to run 'cargo metadata'` | Rust is not on your PATH. `source "$HOME/.cargo/env"` |
| `tauri dev` stuck on "Waiting for your frontend dev server" | `beforeDevCommand` must be `pnpm dev`, or start `pnpm dev` yourself |
| "Can't reach loqmi backend" banner | The backend did not start or is still booting (give it 10 to 20 seconds). Check `lsof -i :8765` and the terminal for `[app_lib][ERROR]` lines |
| `failed to spawn backend` in the log | The bundle is missing from `src-tauri/binaries/loqmi-backend-dist/` |
| Build fails with a type error about a category | A leftover reference to a removed category (for example `translation`) in `app/` or `components/` |
| The app opened after a failed build is an old version | A stale `.app` from an earlier build was launched. Use `&&` after the build command |
| Backend still running after quitting | Quit with Cmd+Q, not just the red button. See the cleanup commands above |

## Known issues

Being worked through ahead of the open-source release:

- Accessibility gaps: custom dropdowns lack keyboard handling and `aria-expanded`, toggles lack `aria-pressed`, and some `alert()` and `confirm()` calls remain.
- The Documents page has a "Re-index All" control with no backend endpoint to call. Hide or disable it.
- v0.dev scaffolding leftovers (placeholder assets, `generator: 'v0.app'`, the `package.json` name) are being cleaned up.
- The Settings "Add Custom GGUF" and "Pull from Hub" flows have limited backend support.

Needs verification (carried over from earlier notes, not re-checked recently):

- The Setup screen's model checkboxes may be disabled by a leftover ternary in `app/page.tsx`.
- RAG and web-search toggle state may not sync from the backend, because `lib/chat-context.tsx` may read a route param while the chat id lives in the query string.

## Contributing

Issues and pull requests are welcome. Before opening a PR:

1. Run `pnpm build` locally. It type-checks the whole app, which `pnpm dev` does not.
2. If you change backend types in `lib/api.ts`, keep them in sync with the [backend README](../backend/README.md).
3. Test desktop-shell changes on a release build (`pnpm tauri build --bundles app`) as well as `pnpm tauri dev`, because the two resolve the backend path differently.

## License

## License

Loqmi is licensed under the [MIT License](../LICENSE).