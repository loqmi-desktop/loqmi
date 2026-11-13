// components/chat-interface.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useConnection } from "@/lib/connection-context";
import {
  chatAPI,
  BackendError,
  attachmentsAPI,
  type ContentBlock,
  type ChatDocument,
  type AnswerSource,
} from "@/lib/api";
import { ErrorBanner } from "@/components/error-banner";
import { ChatMessage } from "@/components/chat-message";
import { SourceChips } from "@/components/source-chips";
import { Composer } from "@/components/composer";
import { useChatState } from "@/lib/chat-context";

interface ChatInterfaceProps {
  initialChatId?: string;
}

type Message = {
  id: string;
  role: "user" | "assistant";
  content: ContentBlock[];
  model_category?: "chat" | "code" | "reasoning" | "document";
  model_name?: string;
  stats?: any;
  sources?: AnswerSource[]; // files / pages this answer drew on (live only, not persisted)
  created_at?: string;
};

export function ChatInterface({ initialChatId }: ChatInterfaceProps) {
  const router = useRouter();
  const { isSearchEnabled, searchMethod } = useChatState();
  const { isConnected } = useConnection();
  const hasCreatedChat = useRef(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const shouldAutoScroll = useRef(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loadChat = async () => {
      try {
        setError(null);
        setIsLoading(true);

        if (initialChatId) {
          // If we have an ID from the URL, fetch the history
          const history = await chatAPI.getMessages(initialChatId);
          setMessages(history);
        } else {
          // If no ID, create a brand new chat
          if (hasCreatedChat.current) return; // Prevent multiple creations
          hasCreatedChat.current = true;
          const newChat = await chatAPI.createChat();
          setMessages([]);

          // Silently update the URL so the user can refresh/bookmark it later.
          // replace() so we don't break the back button history
          router.replace(`/chat?id=${newChat.chat_id}`);
        }
      } catch (err) {
        setError(
          err instanceof BackendError ? err.message : "Failed to load chat",
        );
      } finally {
        setIsLoading(false);
      }
    };

    if (isConnected) loadChat();
  }, [isConnected, initialChatId, router]);

  useEffect(() => {
    if (shouldAutoScroll.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
    }
  }, [messages]);

  useEffect(() => {
    const container = messageContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      // Math.ceil prevents floating-point bugs on zoomed screens
      const distanceToBottom =
        container.scrollHeight -
        Math.ceil(container.scrollTop) -
        container.clientHeight;
      shouldAutoScroll.current = distanceToBottom < 50;
    };

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSendMessage = async (
    content: string,
    files: File[] = [],
    overrideCategory?: string,
    docs: ChatDocument[] = [],
  ) => {
    if (!initialChatId || (!content.trim() && files.length === 0 && docs.length === 0)) return;

    try {
      setError(null);
      setIsSending(true);
      shouldAutoScroll.current = true;

      try {
        await chatAPI.setSearchSettings(
          initialChatId,
          isSearchEnabled,
          searchMethod,
        );
      } catch (err) {
        console.error("Failed to persist search settings before send:", err);
        // Non-fatal: proceed with whatever the backend currently has stored
      }

      // `blocks` is what gets persisted and displayed: compact, forever.
      // `inferenceExtra` is ephemeral text that only augments THIS turn's
      // prompt (raw screenshot OCR text) and is never stored.
      const blocks: ContentBlock[] = [];
      let inferenceExtra = "";

      // Only files and no question: give the model something to do.
      const text =
        content.trim() ||
        (docs.length > 0 && files.length === 0 ? "Summarize the attached file." : "");
      if (text) blocks.push({ label: "text", content: text });

      for (const doc of docs) {
        // A lightweight pointer. The file's text is injected by the backend on every turn.
        blocks.push({
          label: "file",
          doc_id: doc.doc_id,
          filename: doc.filename,
          pages: doc.pages,
        });
      }

      for (const file of files) {
        // Images only: OCR + summarize, never touches the RAG store.
        // Only the summary block is persisted; raw_text is used once then discarded.
        const result = await attachmentsAPI.uploadScreenshot(file);
        blocks.push({
          label: "image",
          msg: result.msg,
          svg: result.svg,
          attachment_id: result.attachment_id,
        });
        if (result.raw_text) {
          inferenceExtra += `\n\n[Screenshot OCR text]\n${result.raw_text}`;
        }
      }

      const inferenceOverrideText = (text + inferenceExtra).trim();

      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: blocks,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);

      const assistantId = (Date.now() + 1).toString();
      let assistantContent = "";

      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: [{ label: "text", content: "" }],
          model_category: undefined,
          created_at: new Date().toISOString(),
        },
      ]);

      for await (const event of chatAPI.sendMessage(
        initialChatId,
        blocks,
        inferenceOverrideText,
        overrideCategory,
      )) {
        if (event.type === "token") {
          assistantContent += event.text;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: [{ label: "text", content: assistantContent }],
                  }
                : m,
            ),
          );
        } else if (event.type === "status") {
          // Progress indicator, shown only until real tokens start arriving
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId && !assistantContent
                ? {
                    ...m,
                    content: [
                      { label: "text", content: `🔎 ${event.message}` },
                    ],
                  }
                : m,
            ),
          );
        } else if (event.type === "reset") {
          // A tool round finished: discard its chatter, the real answer follows
          assistantContent = "";
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: [{ label: "text", content: "" }] }
                : m,
            ),
          );
        } else if (event.type === "sources") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, sources: event.items } : m,
            ),
          );
        } else if (event.type === "stats") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    model_category: event.category,
                    model_name: event.model_name,
                    stats: event,
                  }
                : m,
            ),
          );
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
      setMessages((prev) => prev.slice(0, -2));
    } finally {
      setIsSending(false);
    }
  };

  if (!isConnected) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-md w-full">
          <ErrorBanner message="Can't reach loqmi backend. Make sure the app's backend service is running." />
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-muted-foreground text-sm">Loading chat...</div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* ── Message area ───────────────────────────────────────────────── */}
      <div
        ref={messageContainerRef}
        className={`flex-1 overflow-y-auto flex flex-col ${
          messages.length === 0
            ? "items-center justify-center"
            : "items-stretch justify-start"
        }`}
      >
        {error && (
          <div className="w-full max-w-3xl mx-auto px-6 pt-4">
            <ErrorBanner message={error} onDismiss={() => setError(null)} />
          </div>
        )}

        {messages.length === 0 ? (
          /* ── Empty state, centered ───────────────────────────────────── */
          <div className="flex flex-col items-center text-center px-6 max-w-lg">
            <h2 className="text-xl font-semibold text-foreground mb-2">
              Ask anything. Nothing leaves your machine.
            </h2>
            <p className="text-muted-foreground text-sm mb-6">
              loqmi runs entirely on your device — no cloud, no tracking.
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {["Explain a concept", "Write a function"].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => handleSendMessage(suggestion)}
                  className="px-3 py-1.5 rounded border border-border text-muted-foreground hover:text-foreground hover:border-primary/50 text-xs font-mono transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-4">
              Use the + button to attach a PDF and ask questions about it.
            </p>
          </div>
        ) : (
          /* ── Message list, stacked from top ──────────────────────────── */
          <div className="w-full max-w-3xl mx-auto px-6 py-6 space-y-8">
            {messages.map((message) => (
              <div key={message.id}>
                <ChatMessage
                  role={message.role}
                  content={message.content}
                  modelCategory={message.model_category}
                  modelName={message.model_name}
                  stats={message.stats}
                />
                {message.role === "assistant" && (
                  <div className="pl-12">
                    <SourceChips sources={message.sources} />
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* ── Composer, pinned to bottom ─────────────────────────────────── */}
      <div className="border-t border-border bg-background px-6 py-4">
        <div className="max-w-3xl mx-auto">
          <Composer
            chatId={initialChatId}
            onSendMessage={handleSendMessage}
            isLoading={isSending}
            isDisabled={!isConnected}
          />
          <p className="text-xs text-muted-foreground text-center mt-2">
            Local responses are private and run entirely on your device.
          </p>
        </div>
      </div>
    </div>
  );
}