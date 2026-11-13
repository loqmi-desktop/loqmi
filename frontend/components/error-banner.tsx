'use client'

import { AlertCircle, X } from 'lucide-react'
import { useState } from 'react'

export interface ErrorBannerProps {
  message: string
  onDismiss?: () => void
  type?: 'error' | 'warning' | 'info'
}

export function ErrorBanner({
  message,
  onDismiss,
  type = 'error',
}: ErrorBannerProps) {
  // Track which message was dismissed rather than a boolean: a parent that
  // re-renders with the SAME message stays dismissed, but a new error
  // (different text) is always visible.
  const [dismissedMessage, setDismissedMessage] = useState<string | null>(null)

  if (dismissedMessage === message) return null

  const handleDismiss = () => {
    setDismissedMessage(message)
    onDismiss?.()
  }

  const bgColor =
    type === 'error' ? 'bg-red-950' : type === 'warning' ? 'bg-yellow-950' : 'bg-blue-950'
  const borderColor =
    type === 'error' ? 'border-red-800' : type === 'warning' ? 'border-yellow-800' : 'border-blue-800'
  const textColor =
    type === 'error' ? 'text-red-200' : type === 'warning' ? 'text-yellow-200' : 'text-blue-200'
  const iconColor =
    type === 'error' ? 'text-red-400' : type === 'warning' ? 'text-yellow-400' : 'text-blue-400'

  return (
    <div
      className={`flex items-center gap-3 ${bgColor} border ${borderColor} rounded-lg px-4 py-3 ${textColor} mb-4`}
    >
      <AlertCircle className={`h-5 w-5 shrink-0 ${iconColor}`} />
      <p className="flex-1 text-sm">{message}</p>
      <button
        onClick={handleDismiss}
        className="shrink-0 text-muted-foreground hover:text-foreground transition-colors p-1"
        aria-label="Dismiss"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}
