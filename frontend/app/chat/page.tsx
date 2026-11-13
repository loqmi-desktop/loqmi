// app/chat/page.tsx
'use client'

import { Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import { ChatInterface } from '@/components/chat-interface'

// 1. Extract the URL reading into an inner client component
function ChatContent() {
  const searchParams = useSearchParams()
  const id = searchParams.get('id') || undefined

  // Pass the extracted ID down. 
  // Using key={id || 'new'} ensures the component cleanly remounts when switching chats.
  return <ChatInterface key={id || 'new'} initialChatId={id} />
}

// 2. Wrap it in Suspense so Next.js static export doesn't fail
export default function ChatPage() {
  return (
    <Suspense fallback={
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        Loading chat...
      </div>
    }>
      <ChatContent />
    </Suspense>
  )
}