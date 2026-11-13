// Backend base URL. Override with NEXT_PUBLIC_loqmi_API_URL (e.g. in
// .env.local) if the backend runs on a non-default host or port.
const API_BASE =
  process.env.NEXT_PUBLIC_loqmi_API_URL || 'http://127.0.0.1:8765'

export type APIError = {
  message: string
  status?: number
}

export class BackendError extends Error {
  constructor(
    public message: string,
    public status?: number,
  ) {
    super(message)
    this.name = 'BackendError'
  }
}

export interface SuggestedModel {
  id: string;
  category: string;
  name: string;
  filename: string;
  url: string;
  size_gb?: number;
  n_ctx: number;
  ram_required?: string;
  hardware_target?: string;
  description?: string;
  recommended: boolean;
}

export interface HuggingFaceRepo {
  repo_id: string;
  downloads: number;
  available_files: string[];
}


/**
 * Check if the backend is reachable via its /health endpoint.
 */
export async function checkBackendStatus(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    })
    return response.ok
  } catch {
    return false
  }
}

/**
 * Generic fetch wrapper with error handling (for plain JSON endpoints only —
 * NOT for the SSE streaming endpoints, those need raw fetch + stream reading,
 * see streamSSE() below).
 */
export async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  try {
    const url = `${API_BASE}${endpoint}`
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      signal: AbortSignal.timeout(30000),
      ...options,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: `HTTP ${response.status}`,
      }))
      throw new BackendError(
        error.detail || `HTTP ${response.status}`,
        response.status,
      )
    }

    // Some endpoints (DELETE, PATCH) return 204 No Content — guard against
    // trying to parse an empty body as JSON.
    if (response.status === 204) {
      return undefined as T
    }

    return await response.json()
  } catch (error) {
    if (error instanceof BackendError) {
      throw error
    }
    if (error instanceof TypeError) {
      throw new BackendError(
        'Cannot connect to loqmi backend. Make sure the app is running.',
      )
    }
    throw new BackendError(
      error instanceof Error ? error.message : 'Unknown error',
    )
  }
}

/**
 * Shared SSE stream reader. Used for any endpoint that streams
 * `data: {...}\n\n` events terminated by a literal `data: [DONE]` line.
 * Works for both GET-based SSE (download endpoints) and POST-based SSE
 * (message sending) since both are consumed via fetch + ReadableStream here
 * — EventSource can't be used for the POST case, so we use this everywhere
 * for consistency.
 *
 * Uses an *idle* timeout (reset on every received chunk) rather than a total
 * one: the first token can legitimately take a while while the model loads,
 * but a stream that goes silent for SSE_IDLE_TIMEOUT_MS is dead.
 */
const SSE_IDLE_TIMEOUT_MS = 120_000

export async function* streamSSE(
  endpoint: string,
  options: RequestInit = {},
): AsyncGenerator<any, void, unknown> {
  const url = `${API_BASE}${endpoint}`
  const abortController = new AbortController()
  let idleTimer: ReturnType<typeof setTimeout> | null = null

  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer)
    idleTimer = setTimeout(
      () => abortController.abort(),
      SSE_IDLE_TIMEOUT_MS,
    )
  }

  resetIdleTimer()

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      signal: abortController.signal,
      ...options,
    })

    if (!response.ok || !response.body) {
      const error = await response.json().catch(() => ({
        detail: `HTTP ${response.status}`,
      }))
      throw new BackendError(
        error.detail || `HTTP ${response.status}`,
        response.status,
      )
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      resetIdleTimer()

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      // Keep the last (possibly incomplete) chunk in the buffer
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const dataStr = trimmed.slice(5).trim()

        if (dataStr === '[DONE]') {
          return
        }

        try {
          yield JSON.parse(dataStr)
        } catch {
          // Skip malformed lines rather than crashing the whole stream
          continue
        }
      }
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new BackendError(
        'The backend stream went silent. It may have crashed or hung — try again.',
      )
    }
    throw error
  } finally {
    if (idleTimer) clearTimeout(idleTimer)
  }
}

/**
 * Setup API endpoints
 * Real routes (api/routes/setup.py):
 *   GET    /setup/status
 *   GET    /setup/download                       (SSE)
 *   POST   /setup/models/{category}/{model_id}    (SSE)
 *   DELETE /setup/models/{category}/{model_id}
 *   PUT    /setup/models/{category}/select
 * There is NO bare GET /setup/models — model list comes from /setup/status.
 */

export type ModelInfo = {
  id: string
  name: string
  filename: string
  downloaded: boolean
  size_gb: number
  n_ctx?: number
  ram_required?: string
  hardware_target?: string
  description?: string
  recommended: boolean
  is_active: boolean
  shared_with?: string
}

export type SetupStatus = {
  models: {
    chat: ModelInfo[]
    code: ModelInfo[]
    reasoning: ModelInfo[]
    document: ModelInfo[]
    helper?: ModelInfo[] // Kept optional for your future phase
    image?: ModelInfo[] // Kept optional for your future phase
  }
  all_ready: boolean
}

export type TokenStatus = {
  has_token: boolean
  masked_token: string | null
}

export const setupAPI = {

  async getStatus(): Promise<SetupStatus> {
    return apiFetch('/setup/status')
  },

  async getToken(): Promise<TokenStatus> {
    return apiFetch('/setup/token')
  },

  async saveToken(token: string): Promise<void> {
    return apiFetch('/setup/token', {
      method: 'POST',
      body: JSON.stringify({ token }),
    })
  },
  /**
   * Bulk download of all missing recommended models. SSE stream.
   * Event shapes: start, progress, done, skip, error, complete.
   */
  downloadAll(): AsyncGenerator<any, void, unknown> {
    return streamSSE('/setup/download', { method: 'GET' })
  },

  /**
   * Targeted download of one specific model variant. SSE stream.
   * Returns 409 if that model is already downloading.
   */
  downloadModel(
    category: string,
    modelId: string,
  ): AsyncGenerator<any, void, unknown> {
    return streamSSE(`/setup/models/${category}/${modelId}`, {
      method: 'POST',
    })
  },

  async deleteModel(category: string, modelId: string): Promise<void> {
    return apiFetch(`/setup/models/${category}/${modelId}`, {
      method: 'DELETE',
    })
  },

  /**
   * Set the active model for a category. 400s if the model isn't
   * downloaded yet — disable the UI control in that case rather than
   * relying on this throwing.
   */
  async selectModel(category: string, modelId: string): Promise<void> {
    return apiFetch(`/setup/models/${category}/select`, {
      method: 'PUT',
      body: JSON.stringify({ model_id: modelId }),
    })
  },

  async getSearchKeys(): Promise<SearchKeysResponse> {
    return apiFetch('/setup/search-keys')
  },

  async saveSearchKey(provider: 'brave' | 'tavily', key: string): Promise<void> {
    return apiFetch('/setup/search-keys', {
      method: 'POST',
      body: JSON.stringify({ provider, key }),
    })
  },
  getSuggestedModels: async (): Promise<SuggestedModel[]> => {
    const res = await fetch(`${API_BASE}/setup/catalog`);
    if (!res.ok) throw new BackendError('Failed to fetch catalog');
    return res.json();
  },

  refreshCatalog: async (): Promise<{ status: string; count: number }> => {
    const res = await fetch(`${API_BASE}/setup/refresh-catalog`, { method: 'POST' });
    if (!res.ok) throw new BackendError('Failed to refresh catalog');
    return res.json();
  },

  getDownloadedModels: async (): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/setup/downloaded`);
    if (!res.ok) throw new BackendError('Failed to fetch downloaded models');
    return res.json();
  },

  searchHuggingFace: async (query: string): Promise<HuggingFaceRepo[]> => {
    const res = await fetch(`${API_BASE}/setup/hf-search?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new BackendError('Failed to search Hugging Face');
    return res.json();
  },

  getHuggingFaceFiles: async (repoId: string): Promise<string[]> => {
    const res = await fetch(`${API_BASE}/setup/hf-files?repo_id=${encodeURIComponent(repoId)}`);
    if (!res.ok) throw new BackendError('Failed to fetch repository files');
    return res.json();
  }
}

/**
 * Chat API endpoints
 * Real routes (api/routes/chats.py):
 *   POST   /chats                        -> { chat_id }
 *   GET    /chats                        -> list of chats
 *   DELETE /chats/{chat_id}
 *   GET    /chats/{chat_id}/messages
 *   POST   /chats/{chat_id}/messages     (SSE)
 *   PATCH  /chats/{chat_id}/rag          -> { enabled }
 */

/**
 * A message's `content` is an array of blocks rather than a flat string.
 * - `text`  blocks are whatever the user typed (or the assistant's reply).
 * - `image` blocks represent an attached screenshot: `msg` is the
 *   LLM-generated summary (shown nowhere directly, but is what gets
 *   replayed into future turns as history), `svg` is a base64 thumbnail
 *   shown inline, and `attachment_id` can be used to fetch the full-res
 *   original from GET /attachments/{id}.
 *
 * NOTE: raw OCR text is deliberately NOT part of this type — it's used
 * once (see chatAPI.sendMessage's `inferenceOverrideText` param) and is
 * never persisted or returned by the backend after the initial upload.
 */
export type ContentBlock =
  | { label: 'text'; content: string }
  | { label: 'image'; msg: string; svg: string; attachment_id: string }
  | { label: 'file'; doc_id: string; filename: string; pages: number }

export type ChatDocument = {
  doc_id: string
  filename: string
  size_bytes: number
  pages: number
  chars: number
  status: 'processing' | 'ready' | 'failed'
  detail: string          // stage text while processing, error message when failed
  created_at: string
}

export type AnswerSource = { file: string; page: number | null; scope: 'chat' | 'kb' }

/** Extensions that become chat-scoped documents. Everything else image-like goes the screenshot route. */
export const DOC_EXTENSIONS = ['.pdf', '.txt', '.md']
export const isChatDocument = (f: File) =>
  DOC_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext))
export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: ContentBlock[]
  created_at: string
}

export const chatAPI = {
  async listChats(): Promise<Array<{
    id: string
    title: string
    rag_enabled?: boolean
    search_enabled?: boolean
    search_method?: string
    created_at?: string
  }>> {
    return apiFetch('/chats')
  },

  async createChat(): Promise<{ chat_id: string }> {
    return apiFetch('/chats', { method: 'POST' })
  },

  async deleteChat(chatId: string): Promise<void> {
    return apiFetch(`/chats/${chatId}`, { method: 'DELETE' })
  },

  async getMessages(chatId: string): Promise<ChatMessage[]> {
    return apiFetch(`/chats/${chatId}/messages`)
  },

  /**
   * Sends a message and streams the assistant's reply.
   *
   * `content` is what gets persisted and displayed — the compact block
   * array (image blocks carry only their summary, never raw OCR text).
   *
   * `inferenceOverrideText`, if provided, is what the model actually
   * sees for THIS turn only — e.g. the user's typed text plus a
   * screenshot's full raw OCR text. It is never persisted; the backend
   * uses it to build this turn's prompt and then discards it. Pass an
   * empty string when there's nothing to add beyond the text already in
   * `content`.
   *
   * Event shapes from the backend:
   *   {"type": "token", "text": "..."}
   *   {"type": "status", "message": "..."}
   *   {"type": "stats", "category": "...", "classify_ms": ..., "inference_s": ...}
   *   {"type": "error", "message": "..."}
   * Stream is terminated by a literal `data: [DONE]` line (handled inside streamSSE).
   *
   * Usage:
   *   for await (const event of chatAPI.sendMessage(chatId, blocks, overrideText)) {
   *     if (event.type === 'token') { ... }
   *     if (event.type === 'stats') { ... }
   *     if (event.type === 'error') { ... }
   *   }
   */
  sendMessage(
    chatId: string,
    content: ContentBlock[],
    inferenceOverrideText: string,
    overrideCategory?: string,
  ): AsyncGenerator<any, void, unknown> {
    return streamSSE(`/chats/${chatId}/messages`, {
      method: 'POST',
      body: JSON.stringify({
        content,
        inference_override_text: inferenceOverrideText || null,
        override_category: overrideCategory || null,
      }),
    })
  },

  async setRagEnabled(chatId: string, enabled: boolean): Promise<void> {
    return apiFetch(`/chats/${chatId}/rag`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled }),
    })
  },

   async setSearchSettings(chatId: string, enabled: boolean, method: string): Promise<void> {
    return apiFetch(`/chats/${chatId}/search`, {
      method: 'PATCH',
      body: JSON.stringify({ enabled, method }),
    })
  },
}

/**
 * Attachments API endpoints
 * Real routes (api/routes/attachments.py):
 *   POST /attachments/screenshot   -> { attachment_id, msg, svg, raw_text }
 *   GET  /attachments/{id}         -> full-resolution image file
 *
 * Distinct from documentsAPI: screenshots never touch the RAG knowledge
 * base. Use this for any image attachment; keep using documentsAPI for
 * actual documents (PDF, TXT, MD) that should be searchable via RAG.
 */

export type ScreenshotUploadResult = {
  attachment_id: string
  msg: string
  svg: string
  /** Full OCR text. Use it to build this turn's inferenceOverrideText,
   *  then let it go — never store it anywhere. */
  raw_text: string
}

export const attachmentsAPI = {
  async uploadScreenshot(file: File): Promise<ScreenshotUploadResult> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_BASE}/attachments/screenshot`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(60000),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: `Screenshot upload failed: HTTP ${response.status}`,
      }))
      throw new BackendError(error.detail, response.status)
    }

    return await response.json()
  },

  fullImageUrl(attachmentId: string): string {
    return `${API_BASE}/attachments/${attachmentId}`
  },
}

export const chatDocumentsAPI = {
  async upload(chatId: string, file: File): Promise<ChatDocument> {
    const formData = new FormData()
    formData.append('file', file)
    // Returns immediately (202); processing continues on the backend. Poll get().
    const response = await fetch(`${API_BASE}/chats/${chatId}/documents`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(120_000), // large files take a while to transfer
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: `Upload failed: HTTP ${response.status}` }))
      throw new BackendError(error.detail, response.status)
    }
    return response.json()
  },

  get(chatId: string, docId: string): Promise<ChatDocument> {
    return apiFetch(`/chats/${chatId}/documents/${docId}`)
  },

  async list(chatId: string): Promise<ChatDocument[]> {
    const r = await apiFetch<{ documents: ChatDocument[] }>(`/chats/${chatId}/documents`)
    return r.documents
  },

  remove(chatId: string, docId: string): Promise<void> {
    return apiFetch(`/chats/${chatId}/documents/${docId}`, { method: 'DELETE' })
  },

  promote(chatId: string, docId: string): Promise<{ filename: string; chunks_stored: number }> {
    return apiFetch(`/chats/${chatId}/documents/${docId}/promote`, { method: 'POST' })
  },
}

export type CategoryContextInfo = {
  category: string
  model_id: string
  model_name: string
  current: number      // effective value in use right now
  ceiling: number       // model's real trained max, from GGUF metadata
  recommended: number   // soft suggestion based on device RAM — not a hard cap
  options: number[]     // selectable presets, capped at `ceiling`
}
 
export const contextWindowAPI = {
  async getAll(): Promise<CategoryContextInfo[]> {
    return apiFetch('/context-window')
  },
 
  async setValue(category: string, value: number): Promise<void> {
    return apiFetch(`/context-window/${category}`, {
      method: 'PUT',
      body: JSON.stringify({ value }),
    })
  },
}

/**
 * Documents API endpoints
 * Real routes (api/routes/documents.py):
 *   POST   /documents/ingest        (multipart, field name "file")
 *   GET    /documents               -> { documents: DocumentMetadata[] }
 *   DELETE /documents/{filename}
 * The list carries full per-document metadata (size, chunks, upload date)
 * stored in SQLite by the backend at ingest time.
 */

export type DocumentMetadata = {
  filename: string
  file_size_bytes: number
  chunk_count: number
  upload_date: string
}

export const documentsAPI = {
  async listDocuments(): Promise<{ documents: DocumentMetadata[] }> {
    return apiFetch('/documents')
  },

  async uploadDocument(
    file: File,
  ): Promise<{ filename: string; chunks_stored: number ; extracted_text: string }> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_BASE}/documents/ingest`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(300_000),
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: `Upload failed: HTTP ${response.status}`,
      }))
      throw new BackendError(error.detail, response.status)
    }

    return await response.json()
  },

  /**
   * filename here is the actual source filename (e.g. "report.pdf"),
   * not a generated document id — the backend has no document id concept.
   * Returns 404 if no chunks exist for that filename; treat that as
   * "already gone" rather than a hard failure.
   */
  async deleteDocument(filename: string): Promise<void> {
    try {
      return await apiFetch(`/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      })
    } catch (err) {
      if (err instanceof BackendError && err.status === 404) {
        return
      }
      throw err
    }
  },

  async setRagSettings(chatId: string, enabled: boolean): Promise<any> {
  const res = await fetch(`${API_BASE}/chats/${chatId}/rag`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }), // check your backend's expected payload body format
  })
  if (!res.ok) throw new BackendError('Failed to update RAG settings')
  return res.json()
}

  // NOTE: there is no /documents/reindex endpoint on the backend.
  // "Re-index All" in the UI has nothing to call yet — either hide that
  // button or disable it with a tooltip until a real endpoint exists.
}

/**
 * NOTE ON settingsAPI:
 * The previous version of this file had a whole separate settingsAPI
 * object pointing at /settings/models/*, /settings/models/custom,
 * /settings/models/pull — none of these routes exist anywhere in the
 * backend. Model management for Settings screen should reuse setupAPI
 * above (getStatus, downloadModel, deleteModel, selectModel) since that's
 * the same data the Setup screen uses. "Add Custom GGUF" and "Pull from
 * Hub" have no backend support at all — render them disabled, don't call
 * anything for them.
 */

// ── Search settings types ──
export type SearchKeyStatus = { has_key: boolean; masked_key: string | null }
export type SearchKeysResponse = { brave: SearchKeyStatus; tavily: SearchKeyStatus }