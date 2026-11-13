'use client'

import React, { createContext, useContext, useEffect, useRef, useState } from 'react'
import { checkBackendStatus } from './api'

type ConnectionContextType = {
  isConnected: boolean
  isLoading: boolean
}

const ConnectionContext = createContext<ConnectionContextType | undefined>(
  undefined,
)

const POLL_INTERVAL_MS = 5000

export function ConnectionProvider({
  children,
}: {
  children: React.ReactNode
}) {
  // Start disconnected instead of claiming a connection we haven't verified —
  // the first real check resolves within a second when the backend is up.
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    let cancelled = false

    const scheduleNext = (whenHidden: boolean) => {
      if (cancelled) return
      // Poll less aggressively while the window is hidden — the backend
      // status doesn't change just because nobody is looking.
      const delay = whenHidden ? POLL_INTERVAL_MS * 6 : POLL_INTERVAL_MS
      timerRef.current = setTimeout(checkConnection, delay)
    }

    const checkConnection = async () => {
      try {
        const status = await checkBackendStatus()
        if (!cancelled) {
          setIsConnected(status)
        }
      } catch {
        if (!cancelled) {
          setIsConnected(false)
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false)
          scheduleNext(document.hidden)
        }
      }
    }

    const handleVisibility = () => {
      // Re-check immediately when the window becomes visible again
      if (!document.hidden && timerRef.current) {
        clearTimeout(timerRef.current)
        checkConnection()
      }
    }

    checkConnection()
    document.addEventListener('visibilitychange', handleVisibility)

    return () => {
      cancelled = true
      if (timerRef.current) clearTimeout(timerRef.current)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [])

  return (
    <ConnectionContext.Provider value={{ isConnected, isLoading }}>
      {children}
    </ConnectionContext.Provider>
  )
}

export function useConnection() {
  const context = useContext(ConnectionContext)
  if (!context) {
    throw new Error('useConnection must be used within ConnectionProvider')
  }
  return context
}
