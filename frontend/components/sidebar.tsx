"use client";

import Link from "next/link";
import {
  ChevronLeft,
  Plus,
  FileText,
  Settings,
  MessageSquare,
  Trash2,
} from "lucide-react";
import { usePathname, useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { chatAPI, BackendError } from "@/lib/api";
import { ErrorBanner } from "@/components/error-banner";

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

export function Sidebar({ isOpen, onToggle }: SidebarProps) {
  const router = useRouter();
  const pathname = usePathname();

  // 1. Swap useParams for useSearchParams
  const searchParams = useSearchParams();
  const currentChatId = searchParams.get("id"); // Get the ID from the query string

  const [recentChats, setRecentChats] = useState<
    Array<{ id: string; title: string }>
  >([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRecentChats = async () => {
      try {
        const chats = await chatAPI.listChats();
        setRecentChats(chats);
      } catch (err) {
        if (err instanceof BackendError) {
          setError(err.message);
        } else {
          setError("Failed to load recent chats");
        }
      }
    };

    fetchRecentChats();
  }, [pathname]);

  const handleDeleteChat = async (e: React.MouseEvent, chatId: string) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      await chatAPI.deleteChat(chatId);
      setRecentChats((prev) => prev.filter((chat) => chat.id !== chatId));

      // 2. Check against the query param instead of the path
      if (currentChatId === chatId) {
        router.replace("/chat");
      }
    } catch (err) {
      setError(
        err instanceof BackendError
          ? err.message
          : "Failed to delete chat. Please try again.",
      );
    }
  };

  return (
    <div
      className={`${
        isOpen ? "w-64" : "w-14"
      } bg-sidebar border-r border-sidebar-border flex flex-col transition-all duration-200 overflow-hidden h-full`}
    >
      <div className="p-4 border-b border-sidebar-border flex items-center justify-between gap-2">
        <div className={`flex items-center gap-2 ${!isOpen && "hidden"}`}>
          <div className="w-8 h-8 bg-primary rounded flex items-center justify-center text-xs font-bold text-primary-foreground">
            LM
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-semibold text-foreground text-sm truncate">
              loqmi
            </div>
            <div className="text-xs text-muted-foreground truncate">
              Offline Intelligence
            </div>
          </div>
        </div>
        {isOpen && (
          <button
            onClick={onToggle}
            className="text-muted-foreground hover:text-foreground transition-colors p-1 flex-shrink-0"
            aria-label="Toggle sidebar"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        )}
        {!isOpen && (
          <button
            onClick={onToggle}
            className="text-muted-foreground hover:text-foreground transition-colors p-1 w-full flex justify-center"
            aria-label="Expand sidebar"
          >
            <ChevronLeft className="h-4 w-4 rotate-180" />
          </button>
        )}
      </div>

      {error && (
        <div className="px-2 pt-2">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      <div className="p-2">
        <Link
          href="/chat"
          className={`${
            isOpen ? "px-3 py-2" : "p-2"
          } w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground font-medium rounded-lg hover:bg-primary/90 transition-colors`}
        >
          <Plus className="h-4 w-4 flex-shrink-0" />
          {isOpen && <span>New Chat</span>}
        </Link>
      </div>

      <nav className="px-2 py-4 space-y-1">
        <NavItem
          icon={<MessageSquare className="h-4 w-4" />}
          label="Chats"
          isOpen={isOpen}
          // 3. Simplified the active check since we don't have dynamic /chat/... paths anymore
          isActive={pathname === "/chat"}
          href="/chat"
        />
        <NavItem
          icon={<FileText className="h-4 w-4" />}
          label="Documents"
          isOpen={isOpen}
          isActive={pathname === "/documents"}
          href="/documents"
        />
        <NavItem
          icon={<Settings className="h-4 w-4" />}
          label="Settings"
          isOpen={isOpen}
          isActive={pathname === "/settings"}
          href="/settings"
        />
      </nav>

      {isOpen && (
        <div className="flex-1 overflow-y-auto px-3 py-4 border-t border-sidebar-border">
          <div className="text-xs uppercase text-muted-foreground mb-2 font-mono px-2">
            Recent chats
          </div>

          <div className="space-y-1">
            {recentChats.length === 0 ? (
              <div className="text-xs text-muted-foreground px-2">
                No chats yet
              </div>
            ) : (
              recentChats.map((chat) => (
                <Link
                  key={chat.id}
                  // 4. Update the link to use a query parameter
                  href={`/chat?id=${chat.id}`}
                  className={`group flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                    currentChatId === chat.id
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-sidebar-foreground hover:bg-secondary"
                  }`}
                >
                  <MessageSquare className="h-4 w-4 shrink-0 opacity-70" />

                  <span className="flex-1 truncate text-sm">{chat.title}</span>

                  <button
                    onClick={(e) => handleDeleteChat(e, chat.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:text-destructive shrink-0"
                    title="Delete chat"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </Link>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ... NavItem component remains exactly the same ...
interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  isOpen: boolean;
  isActive: boolean;
  href: string;
}

function NavItem({ icon, label, isOpen, isActive, href }: NavItemProps) {
  return (
    <Link
      href={href}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
        isActive
          ? "bg-sidebar-primary text-sidebar-primary-foreground"
          : "text-sidebar-foreground hover:bg-secondary"
      } ${!isOpen && "justify-center"}`}
      title={!isOpen ? label : undefined}
    >
      {icon}
      {isOpen && <span className="text-sm">{label}</span>}
    </Link>
  );
}