"use client";

import { useEffect } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";

const EXTERNAL = /^(https?:|mailto:|tel:)/i;

export function ExternalLinkHandler() {
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      const anchor = (e.target as HTMLElement | null)?.closest?.("a");
      if (!anchor || !anchor.href) return;

      const url = anchor.href;
      if (!EXTERNAL.test(url)) return;

      // Same-origin links (dev server) stay inside the app
      try {
        if (new URL(url).origin === window.location.origin) return;
      } catch {}

      e.preventDefault();
      openUrl(url).catch((err) =>
        console.error("Failed to open external link:", err),
      );
    };

    document.addEventListener("click", onClick, true); // capture phase
    return () => document.removeEventListener("click", onClick, true);
  }, []);

  return null;
}