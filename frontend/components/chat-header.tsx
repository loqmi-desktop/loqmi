'use client'

import { useState } from 'react'
import { Settings, FileText, Database, Globe } from 'lucide-react'
import Link from 'next/link'
import { chatAPI, BackendError } from '@/lib/api'
import { ErrorBanner } from '@/components/error-banner'

interface ChatHeaderProps {
  chatId: string | null
  isRagEnabled: boolean
  setIsRagEnabled: (enabled: boolean) => void
  isSearchEnabled: boolean
  setIsSearchEnabled: (enabled: boolean) => void
  searchMethod: string
  onToggleDocuments?: () => void
  documentsOpen?: boolean
}

export function ChatHeader({
  chatId,
  isRagEnabled,
  setIsRagEnabled,
  isSearchEnabled,
  setIsSearchEnabled,
  searchMethod,
  onToggleDocuments,
  documentsOpen,
}: ChatHeaderProps) {
  const [error, setError] = useState<string | null>(null)

  const handleRagToggle = async () => {
    if (!chatId) return
    const newState = !isRagEnabled
    setIsRagEnabled(newState)
    try {
      await chatAPI.setRagEnabled(chatId, newState)
    } catch (err) {
      setIsRagEnabled(!newState)
      setError(
        err instanceof BackendError
          ? err.message
          : 'Failed to update RAG setting. Please try again.',
      )
    }
  }

  const handleSearchToggle = async () => {
    if (!chatId) return
    const newState = !isSearchEnabled
    setIsSearchEnabled(newState)
    try {
      await chatAPI.setSearchSettings(chatId, newState, searchMethod)
    } catch (err) {
      setIsSearchEnabled(!newState)
      setError(
        err instanceof BackendError
          ? err.message
          : 'Failed to update search setting. Please try again.',
      )
    }
  }

  return (
    <div className="border-b border-border px-6 py-4 flex items-center justify-between bg-background">
      <h1 className="text-lg font-semibold text-foreground">loqmi</h1>

      <div className="flex items-center gap-3">
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        {/* Internet Search Toggle */}
        <button
          onClick={handleSearchToggle}
          disabled={!chatId}
          aria-pressed={isSearchEnabled}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors border ${
            isSearchEnabled
              ? 'bg-primary/10 border-primary text-primary'
              : 'border-border text-muted-foreground hover:bg-secondary'
          } disabled:opacity-50`}
          title="Toggle Internet Search"
        >
          <Globe className="h-4 w-4" />
          <span className="text-sm font-medium">
            {isSearchEnabled ? 'Internet: On' : 'Internet: Off'}
          </span>
        </button>

        {/* RAG Toggle */}
        <button
          onClick={handleRagToggle}
          disabled={!chatId}
          aria-pressed={isRagEnabled}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors border ${
            isRagEnabled
              ? 'bg-primary/10 border-primary text-primary'
              : 'border-border text-muted-foreground hover:bg-secondary'
          } disabled:opacity-50`}
          title="Toggle Semantic Search (RAG)"
        >
          <Database className="h-4 w-4" />
          <span className="text-sm font-medium">
            {isRagEnabled ? 'RAG: On' : 'RAG: Off'}
          </span>
        </button>

        <Link
          href="/documents"
          className="flex items-center gap-2 px-3 py-2 rounded-lg transition-colors text-muted-foreground hover:text-foreground hover:bg-secondary"
          title="Manage Knowledge Base"
        >
          <FileText className="h-4 w-4" />
          <span className="text-sm font-medium">Knowledge Base</span>
        </Link>

        <Link
          href="/settings"
          className="text-muted-foreground hover:text-foreground transition-colors p-2 hover:bg-secondary rounded-lg"
          title="Settings"
          aria-label="Settings"
        >
          <Settings className="h-5 w-5" />
        </Link>
      </div>
    </div>
  )
}
