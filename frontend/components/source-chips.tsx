import { FileText } from "lucide-react";
import type { AnswerSource } from "@/lib/api";

/** Which files (and pages) the answer drew on. Render under an assistant message. */
export function SourceChips({ sources }: { sources?: AnswerSource[] }) {
  if (!sources || sources.length === 0) return null;

  // Group pages per file: "report.pdf · p. 3, 7"
  const byFile = new Map<string, { pages: number[]; scope: AnswerSource["scope"] }>();
  for (const s of sources) {
    const entry = byFile.get(s.file) ?? { pages: [], scope: s.scope };
    if (s.page != null && !entry.pages.includes(s.page)) entry.pages.push(s.page);
    byFile.set(s.file, entry);
  }

  return (
    <div className="flex flex-wrap gap-2 mt-3" aria-label="Sources used for this answer">
      {[...byFile.entries()].map(([file, { pages, scope }]) => (
        <span
          key={file}
          title={scope === "kb" ? "From your knowledge base" : "From a file attached to this chat"}
          className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md border border-border bg-secondary text-xs text-muted-foreground"
        >
          <FileText className="h-3 w-3" />
          <span className="truncate max-w-[200px]">{file}</span>
          {pages.length > 0 && <span>p. {pages.sort((a, b) => a - b).join(", ")}</span>}
        </span>
      ))}
    </div>
  );
}