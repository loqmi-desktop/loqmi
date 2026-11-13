'use client'

import React, { createContext, useContext, useState, useEffect, Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import { chatAPI } from '@/lib/api'

interface ChatContextType {
  isRagEnabled: boolean
  setIsRagEnabled: (enabled: boolean) => void
  isSearchEnabled: boolean
  setIsSearchEnabled: (enabled: boolean) => void
  searchMethod: string
  setSearchMethod: (method: string) => void
  chatId: string | null
}

const ChatContext = createContext<ChatContextType | undefined>(undefined)

// 1. Isolate the hook logic inside a component that will be wrapped in Suspense
function ChatProviderInner({ children }: { children: React.ReactNode }) {
  const searchParams = useSearchParams()
  const chatId = searchParams?.get('id') || null
  
  const [isRagEnabled, setIsRagEnabled] = useState(false)
  const [isSearchEnabled, setIsSearchEnabled] = useState(false)
  const [searchMethod, setSearchMethod] = useState('duckduckgo')

  // Sync state from backend on load
  useEffect(() => {
    const syncState = async () => {
      if (!chatId) {
        setIsRagEnabled(false)
        setIsSearchEnabled(false)
        setSearchMethod('duckduckgo')
        return
      }
      try {
        const chats = await chatAPI.listChats()
        
        // FIX: Check for both snake_case and camelCase ID variations
        const currentChat = chats.find((c: any) => c.id === chatId || c.chat_id === chatId)
        
        if (currentChat) {
          setIsRagEnabled(!!currentChat.rag_enabled)
          setIsSearchEnabled(!!currentChat.search_enabled)
          setSearchMethod(currentChat.search_method || 'duckduckgo')
        }
      } catch (err) {
        console.error('Failed to sync chat state:', err)
      }
    }
    syncState()
  }, [chatId])

  // FIX: Intercept state changes and push them to the backend immediately
  const handleSetSearchEnabled = async (enabled: boolean) => {
    setIsSearchEnabled(enabled) // Optimistic UI update
    if (chatId) {
      try {
        await chatAPI.setSearchSettings(chatId, enabled, searchMethod)
      } catch (err) {
        console.error('Failed to update search toggle on backend', err)
      }
    }
  }

  const handleSetSearchMethod = async (method: string) => {
    setSearchMethod(method) // Optimistic UI update
    if (chatId) {
      try {
        await chatAPI.setSearchSettings(chatId, isSearchEnabled, method)
      } catch (err) {
        console.error('Failed to update search method on backend', err)
      }
    }
  }

  const handleSetRagEnabled = async (enabled: boolean) => {
    setIsRagEnabled(enabled)
    if (chatId) {
      try {
        await chatAPI.setRagEnabled(chatId, enabled) // Assuming this endpoint exists based on your structure
      } catch (err) {
        console.error('Failed to update RAG toggle on backend', err)
      }
    }
  }

  return (
    <ChatContext.Provider
      value={{
        isRagEnabled,
        setIsRagEnabled: handleSetRagEnabled,
        isSearchEnabled,
        setIsSearchEnabled: handleSetSearchEnabled,
        searchMethod,
        setSearchMethod: handleSetSearchMethod,
        chatId,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

// 2. Export the Provider wrapped in a Suspense boundary to satisfy Next.js 15
export function ChatProvider({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={null}>
      <ChatProviderInner>{children}</ChatProviderInner>
    </Suspense>
  )
}

export function useChatState() {
  const context = useContext(ChatContext)
  if (!context) throw new Error('useChatState must be used within a ChatProvider')
  return context
}