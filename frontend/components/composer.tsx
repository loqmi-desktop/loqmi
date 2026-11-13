'use client'

import { useState, useRef, useEffect } from 'react'
import {
  Send, Plus, X, FileText, Sparkles, Brain, Code, Globe, Check, Loader2, AlertCircle,
} from 'lucide-react'
import { useChatState } from '@/lib/chat-context'
import {
  setupAPI,
  chatAPI,
  chatDocumentsAPI,
  isChatDocument,
  BackendError,
  type SearchKeysResponse,
  type ChatDocument,
} from '@/lib/api'

interface ComposerProps {
  /** The chat this composer belongs to. Needed to attach PDFs / text files. */
  chatId?: string
  onSendMessage: (
    message: string,
    files: File[],
    overrideCategory?: string,
    docs?: ChatDocument[],
  ) => void
  isLoading?: boolean
  isDisabled?: boolean
}

const ROUTING_OPTIONS = [
  { id: 'auto', name: 'Auto Route', icon: Sparkles },
  { id: 'chat', name: 'General Chat', icon: Brain },
  { id: 'code', name: 'Code Generation', icon: Code },
  { id: 'reasoning', name: 'Deep Reasoning', icon: Brain },
  { id: 'document', name: 'Document Analysis', icon: FileText },
]

const SEARCH_METHODS = [
  { id: 'duckduckgo', name: 'DuckDuckGo', requiresKey: false },
  { id: 'searxng', name: 'SearXNG', requiresKey: false },
  { id: 'brave', name: 'Brave Search', requiresKey: true },
  { id: 'tavily', name: 'Tavily', requiresKey: true },
]

export function Composer({
  chatId,
  onSendMessage,
  isLoading = false,
  isDisabled = false,
}: ComposerProps) {
  const [input, setInput] = useState('')
  // Images: kept as File objects and processed (OCR + summary) when the message is sent.
  const [attachedFiles, setAttachedFiles] = useState<File[]>([])
  // PDFs / text files: uploaded immediately and processed in the background.
  const [docs, setDocs] = useState<ChatDocument[]>([])
  const [uploadError, setUploadError] = useState<string | null>(null)

  const [selectedRoute, setSelectedRoute] = useState('auto')
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [searchDropdownOpen, setSearchDropdownOpen] = useState(false)
  const [availableKeys, setAvailableKeys] = useState<SearchKeysResponse | null>(null)

  const { isSearchEnabled, searchMethod, setSearchMethod } = useChatState()

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const routeDropdownRef = useRef<HTMLDivElement>(null)
  const searchDropdownRef = useRef<HTMLDivElement>(null)

  const docsProcessing = docs.some((d) => d.status === 'processing')
  const docsReady = docs.filter((d) => d.status === 'ready')

  useEffect(() => {
    if (isSearchEnabled) {
      setupAPI.getSearchKeys().then(setAvailableKeys).catch(() => setAvailableKeys(null))
    }
  }, [isSearchEnabled])

  // Poll the backend while any attachment is still being read / indexed
  useEffect(() => {
    if (!chatId || !docsProcessing) return
    const timer = setInterval(async () => {
      const pending = docs.filter((d) => d.status === 'processing')
      const updated = await Promise.all(
        pending.map((d) => chatDocumentsAPI.get(chatId, d.doc_id).catch(() => d)),
      )
      setDocs((prev) => prev.map((d) => updated.find((u) => u.doc_id === d.doc_id) ?? d))
    }, 800)
    return () => clearInterval(timer)
  }, [chatId, docsProcessing, docs])

  // Close either dropdown on Escape or on click/tap outside of it
  useEffect(() => {
    if (!dropdownOpen && !searchDropdownOpen) return

    const handlePointerDown = (e: PointerEvent) => {
      const target = e.target as Node
      if (dropdownOpen && routeDropdownRef.current && !routeDropdownRef.current.contains(target)) {
        setDropdownOpen(false)
      }
      if (searchDropdownOpen && searchDropdownRef.current && !searchDropdownRef.current.contains(target)) {
        setSearchDropdownOpen(false)
      }
    }
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setDropdownOpen(false)
        setSearchDropdownOpen(false)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [dropdownOpen, searchDropdownOpen])

  // Auto-resize the textarea as the user types
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [input])

  const handleSelectSearchMethod = async (methodId: string) => {
    setSearchMethod(methodId)
    setSearchDropdownOpen(false)
    if (chatId) {
      try {
        await chatAPI.setSearchSettings(chatId, true, methodId)
      } catch (err) {
        console.error('Failed to persist search method:', err)
      }
    }
  }

  const canSend =
    (input.trim().length > 0 || attachedFiles.length > 0 || docsReady.length > 0) &&
    !docsProcessing &&
    !isLoading &&
    !isDisabled

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!canSend) return

    // Determine if we are bypassing the classifier
    const overrideParam = selectedRoute === 'auto' ? undefined : selectedRoute

    onSendMessage(input.trim(), attachedFiles, overrideParam, docsReady)

    setInput('')
    setAttachedFiles([])
    // Sent documents stay attached to the chat on the backend (follow-up questions keep working),
    // so only failed uploads remain visible here.
    setDocs((prev) => prev.filter((d) => d.status === 'failed'))
    setUploadError(null)
    // Note: we keep the selected route active for their next message

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files ? Array.from(e.target.files) : []
    if (fileInputRef.current) fileInputRef.current.value = ''
    setUploadError(null)

    // Images keep the screenshot flow (processed when the message is sent)
    setAttachedFiles((prev) => [...prev, ...picked.filter((f) => !isChatDocument(f))])

    // PDFs / text files upload right away so they can be processed while the user types
    for (const file of picked.filter(isChatDocument)) {
      if (!chatId) {
        setUploadError('The chat is still loading. Try again in a moment.')
        break
      }
      try {
        const doc = await chatDocumentsAPI.upload(chatId, file)
        setDocs((prev) => (prev.some((d) => d.doc_id === doc.doc_id) ? prev : [...prev, doc]))
      } catch (err) {
        setUploadError(err instanceof BackendError ? err.message : `Couldn't upload ${file.name}.`)
      }
    }
  }

  const removeFile = (indexToRemove: number) => {
    setAttachedFiles((prev) => prev.filter((_, index) => index !== indexToRemove))
  }

  const removeDoc = (doc: ChatDocument) => {
    setDocs((prev) => prev.filter((d) => d.doc_id !== doc.doc_id))
    if (chatId && doc.status !== 'failed') {
      chatDocumentsAPI.remove(chatId, doc.doc_id).catch(() => {})
    }
  }

  const ActiveIcon = ROUTING_OPTIONS.find((o) => o.id === selectedRoute)?.icon || Sparkles
  const activeSearchMethodName = SEARCH_METHODS.find((m) => m.id === searchMethod)?.name || 'DuckDuckGo'

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* ── Document attachments (PDF / text): status chips ── */}
      {(docs.length > 0 || uploadError) && (
        <div className="flex flex-wrap gap-2 mb-2 px-2">
          {docs.map((doc) => (
            <div
              key={doc.doc_id}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm border ${
                doc.status === 'failed'
                  ? 'border-destructive/50 text-destructive bg-destructive/5'
                  : 'border-border bg-secondary text-secondary-foreground'
              }`}
            >
              {doc.status === 'processing' && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
              {doc.status === 'ready' && <FileText className="h-4 w-4 text-primary" />}
              {doc.status === 'failed' && <AlertCircle className="h-4 w-4" />}
              <div className="flex flex-col leading-tight">
                <span className="truncate max-w-[180px]">{doc.filename}</span>
                <span className="text-xs text-muted-foreground">
                  {doc.status === 'processing' && (doc.detail || 'Reading…')}
                  {doc.status === 'ready' && `${doc.pages} ${doc.pages === 1 ? 'page' : 'pages'}`}
                  {doc.status === 'failed' && doc.detail}
                </span>
              </div>
              <button
                type="button"
                onClick={() => removeDoc(doc)}
                aria-label={`Remove ${doc.filename}`}
                className="text-muted-foreground hover:text-foreground ml-1"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ))}
          {uploadError && (
            <p role="alert" className="text-xs text-destructive self-center">
              {uploadError}
            </p>
          )}
        </div>
      )}

      {/* ── Image attachments preview ── */}
      {attachedFiles.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2 px-2">
          {attachedFiles.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-2 bg-secondary text-secondary-foreground px-3 py-1.5 rounded-md text-sm border border-border"
            >
              <FileText className="h-4 w-4 text-primary" />
              <span className="truncate max-w-[150px]">{file.name}</span>
              <button
                type="button"
                onClick={() => removeFile(index)}
                aria-label={`Remove ${file.name}`}
                className="text-muted-foreground hover:text-foreground ml-1"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ── Input Area ── */}
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          className="hidden"
          accept=".pdf,.txt,.md,image/*"
          multiple
        />

        <div className="relative bg-secondary border border-border rounded-lg transition-colors focus-within:ring-2 focus-within:ring-primary focus-within:border-transparent p-2 flex flex-col gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message loqmi..."
            disabled={isDisabled || isLoading}
            rows={1}
            className="w-full px-2 py-1 bg-transparent text-foreground placeholder-muted-foreground focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed resize-none overflow-y-auto block max-h-[200px]"
            style={{ minHeight: '24px' }}
          />

          <div className="flex items-center justify-between border-t border-border/40 pt-2 px-1">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isDisabled || isLoading}
                className="text-muted-foreground hover:text-foreground hover:bg-background rounded-md p-1.5 disabled:opacity-50 transition-colors"
                title="Attach a PDF, text file or screenshot"
                aria-label="Add attachment"
              >
                <Plus className="h-4 w-4" />
              </button>

              {/* Route Selector Dropdown */}
              <div className="relative" ref={routeDropdownRef}>
                <button
                  type="button"
                  onClick={() => setDropdownOpen(!dropdownOpen)}
                  disabled={isDisabled || isLoading}
                  aria-haspopup="menu"
                  aria-expanded={dropdownOpen}
                  className="flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs bg-background hover:bg-background/80 font-medium text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                >
                  <ActiveIcon className="h-3.5 w-3.5 text-primary" />
                  <span>{ROUTING_OPTIONS.find((o) => o.id === selectedRoute)?.name}</span>
                </button>
                {dropdownOpen && (
                  <div
                    role="menu"
                    aria-label="Routing override"
                    className="absolute bottom-full left-0 mb-2 w-48 bg-popover border border-border rounded-md shadow-lg py-1 z-50"
                  >
                    {ROUTING_OPTIONS.map((option) => {
                      const Icon = option.icon
                      return (
                        <button
                          key={option.id}
                          type="button"
                          role="menuitemradio"
                          aria-checked={selectedRoute === option.id}
                          onClick={() => {
                            setSelectedRoute(option.id)
                            setDropdownOpen(false)
                          }}
                          className={`w-full flex items-center gap-2 px-3 py-2 text-xs text-left hover:bg-secondary transition-colors ${
                            selectedRoute === option.id
                              ? 'text-foreground font-semibold bg-secondary/50'
                              : 'text-muted-foreground'
                          }`}
                        >
                          <Icon className="h-3.5 w-3.5" />
                          {option.name}
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Search Method Dropdown — only visible when header toggle is on */}
              {isSearchEnabled && (
                <div className="relative" ref={searchDropdownRef}>
                  <button
                    type="button"
                    onClick={() => setSearchDropdownOpen(!searchDropdownOpen)}
                    disabled={isDisabled || isLoading}
                    aria-haspopup="menu"
                    aria-expanded={searchDropdownOpen}
                    className="flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs bg-background hover:bg-background/80 font-medium text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                  >
                    <Globe className="h-3.5 w-3.5 text-primary" />
                    <span>{activeSearchMethodName}</span>
                  </button>
                  {searchDropdownOpen && (
                    <div
                      role="menu"
                      aria-label="Search method"
                      className="absolute bottom-full left-0 mb-2 w-52 bg-popover border border-border rounded-md shadow-lg py-1 z-50"
                    >
                      {SEARCH_METHODS.map((method) => {
                        const hasKey =
                          !method.requiresKey || (availableKeys as any)?.[method.id]?.has_key
                        return (
                          <button
                            key={method.id}
                            type="button"
                            role="menuitemradio"
                            aria-checked={searchMethod === method.id}
                            disabled={!hasKey}
                            onClick={() => hasKey && handleSelectSearchMethod(method.id)}
                            className={`w-full flex items-center justify-between gap-2 px-3 py-2 text-xs text-left transition-colors ${
                              !hasKey
                                ? 'text-muted-foreground/40 cursor-not-allowed'
                                : searchMethod === method.id
                                  ? 'text-foreground font-semibold bg-secondary/50 hover:bg-secondary'
                                  : 'text-muted-foreground hover:bg-secondary'
                            }`}
                            title={!hasKey ? 'Add an API key in Settings to enable this' : undefined}
                          >
                            <span>{method.name}</span>
                            {searchMethod === method.id && hasKey && <Check className="h-3 w-3" />}
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={!canSend}
              className="bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground p-2 rounded-md transition-colors disabled:cursor-not-allowed shrink-0"
              title={docsProcessing ? 'Waiting for your file to finish processing' : 'Send message'}
              aria-label="Send message"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      </form>

      {isDisabled && (
        <p className="text-xs text-muted-foreground mt-2 text-center">
          Backend is not available. Reconnecting...
        </p>
      )}
    </div>
  )
}