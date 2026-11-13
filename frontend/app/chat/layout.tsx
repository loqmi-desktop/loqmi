// NO 'use client' here — this is a server component
import { Suspense } from 'react'
import { ChatProvider } from '@/lib/chat-context'
import { ChatLayoutClient } from './layout-client'

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="h-screen w-screen flex items-center justify-center bg-background text-muted-foreground">
          Loading interface...
        </div>
      }
    >
      <ChatProvider>
        <ChatLayoutClient>{children}</ChatLayoutClient>
      </ChatProvider>
    </Suspense>
  )
}