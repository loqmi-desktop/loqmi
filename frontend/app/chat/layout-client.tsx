"use client";

import { useState } from "react";
import { Sidebar } from "@/components/sidebar";
import { ChatHeader } from "@/components/chat-header";
import { DocumentsPanel } from "@/components/documents-panel";
import { useConnection } from "@/lib/connection-context";
import { useSearchParams } from "next/navigation";
import { useChatState } from "@/lib/chat-context";

export function ChatLayoutClient({ children }: { children: React.ReactNode }) {
  const { isConnected } = useConnection();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [documentsOpen, setDocumentsOpen] = useState(false);
  const { chatId, isRagEnabled, setIsRagEnabled, setIsSearchEnabled, isSearchEnabled, searchMethod } = useChatState();
  const searchParams = useSearchParams();
  const activeChatId = searchParams.get("id") || null;

  return (
    <div className="flex h-screen bg-background">
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <div className="flex-1 flex flex-col overflow-hidden">
        <ChatHeader
          chatId={activeChatId}
          isRagEnabled={isRagEnabled}
          setIsRagEnabled={setIsRagEnabled}
          isSearchEnabled={isSearchEnabled}
          setIsSearchEnabled={setIsSearchEnabled}
          searchMethod={searchMethod}
          onToggleDocuments={() => setDocumentsOpen(!documentsOpen)}
          documentsOpen={documentsOpen}
        />
        <div className="flex-1 flex overflow-hidden relative">
          <div className="flex-1 flex flex-col overflow-hidden">{children}</div>
          {documentsOpen && (
            <div className="w-96 border-l border-border bg-sidebar flex flex-col">
              <div className="p-4 border-b border-border flex items-center justify-between">
                <h2 className="font-semibold text-foreground">Documents</h2>
                <button
                  onClick={() => setDocumentsOpen(false)}
                  className="text-muted-foreground hover:text-foreground transition-colors p-1"
                  aria-label="Close documents"
                >
                  ✕
                </button>
              </div>
              <div className="flex-1 overflow-auto">
                <DocumentsPanel isConnected={isConnected} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
