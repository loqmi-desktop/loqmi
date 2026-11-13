"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Download, ChevronDown, ChevronUp, Check } from "lucide-react";
import { setupAPI, BackendError, type ModelInfo } from "@/lib/api";
import { useConnection } from "@/lib/connection-context";
import { ErrorBanner } from "@/components/error-banner";
import { ProgressBar } from "@/components/progress-bar";
import { StatusBadge } from "@/components/status-badge";

type Category = "chat" | "code" | "reasoning" | "document";

const CATEGORIES: Category[] = ["chat", "code", "reasoning", "document"];

const CATEGORY_INFO: Record<Category, { title: string; blurb: string; required: boolean }> = {
  chat: {
    title: "General chat",
    blurb: "Answers everyday questions. Also covers anything you haven't set up a specialist for.",
    required: true,
  },
  code: { title: "Coding", blurb: "Used when you ask for code or debugging help.", required: false },
  reasoning: { title: "Deep reasoning", blurb: "Used for step-by-step analysis and maths.", required: false },
  document: { title: "Documents", blurb: "Used for summarising and questions about your files.", required: false },
};

type ModelsByCategory = Record<Category, ModelInfo[]>;

const EMPTY: ModelsByCategory = { chat: [], code: [], reasoning: [], document: [] };

// A model that is only a pointer to the chat model needs no download of its own.
const isDownloadable = (m: ModelInfo) => m.shared_with !== "chat";

export default function SetupPage() {
  const router = useRouter();
  const { isConnected } = useConnection();

  const [models, setModels] = useState<ModelsByCategory>(EMPTY);
  // One pick per category. `null` means "don't add this one" (not allowed for chat).
  const [picks, setPicks] = useState<Record<Category, string | null>>({
    chat: null, code: null, reasoning: null, document: null,
  });
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isDownloading, setIsDownloading] = useState(false);
  const [showCustomize, setShowCustomize] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isConnected) return;
    let cancelled = false;

    (async () => {
      try {
        setError(null);
        const status = await setupAPI.getStatus();
        if (cancelled) return;

        const grouped = Object.fromEntries(
          CATEGORIES.map((c) => [c, status.models[c] ?? []]),
        ) as ModelsByCategory;

        // First run is over as soon as there is a chat model on disk. The other
        // categories fall back to it, so they are upgrades, not requirements.
        if (grouped.chat.some((m) => m.downloaded)) {
          router.replace("/chat");
          return;
        }

        setModels(grouped);
        setPicks({
          chat: pickDefault(grouped.chat),
          // Optional categories start unselected on the Essentials path.
          code: null, reasoning: null, document: null,
        });
      } catch (err) {
        if (!cancelled) setError(err instanceof BackendError ? err.message : "Couldn't load the model list.");
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [isConnected, router]);

  const toDownload = useMemo(() => {
    const out: { category: Category; model: ModelInfo }[] = [];
    for (const c of CATEGORIES) {
      const id = picks[c];
      const m = models[c].find((x) => x.id === id);
      if (m && !m.downloaded && isDownloadable(m)) out.push({ category: c, model: m });
    }
    return out;
  }, [picks, models]);

  const totalGb = toDownload.reduce((sum, x) => sum + (x.model.size_gb || 0), 0);

  const applyBundle = (bundle: "essentials" | "full") => {
    setPicks({
      chat: picks.chat ?? pickDefault(models.chat),
      code: bundle === "full" ? pickDefault(models.code) : null,
      reasoning: bundle === "full" ? pickDefault(models.reasoning) : null,
      document: bundle === "full" ? pickDefault(models.document) : null,
    });
  };

  const bundle: "essentials" | "full" | "custom" = (() => {
    const optional = (["code", "reasoning", "document"] as Category[]).map((c) => picks[c]);
    if (optional.every((p) => p === null)) return "essentials";
    if (optional.every((p, i) => p === pickDefault(models[(["code", "reasoning", "document"] as Category[])[i]]))) return "full";
    return "custom";
  })();

  const start = async () => {
    if (!picks.chat) return;
    setError(null);
    setIsDownloading(true);

    try {
      // Chat first: once it is on disk the app is usable even if a later download fails.
      for (const { category, model } of toDownload) {
        let failed: string | null = null;
        for await (const ev of setupAPI.downloadModel(category, model.id)) {
          if (ev.type === "progress") setProgress((p) => ({ ...p, [model.id]: ev.percent }));
          else if (ev.type === "done") setProgress((p) => ({ ...p, [model.id]: 100 }));
          else if (ev.type === "error") failed = ev.message;
        }
        if (failed) {
          setError(`Couldn't download ${model.name}: ${failed}`);
          if (category === "chat") return; // can't continue without the chat model
          continue;
        }
        try { await setupAPI.selectModel(category, model.id); } catch { /* may auto-activate */ }
      }

      // If chat was already installed but inactive, make sure it is the active one.
      const chosen = models.chat.find((m) => m.id === picks.chat);
      if (chosen?.downloaded) {
        try { await setupAPI.selectModel("chat", chosen.id); } catch { /* ignore */ }
      }

      router.replace("/chat");
    } catch (err) {
      setError(err instanceof BackendError ? err.message : "Download failed. Check your connection and try again.");
    } finally {
      setIsDownloading(false);
    }
  };

  if (!isConnected) {
    return (
      <Shell>
        <ErrorBanner message="Can't reach the loqmi backend. It may still be starting — give it a few seconds." />
      </Shell>
    );
  }

  if (isLoading) {
    return (
      <Shell>
        <p className="text-muted-foreground text-center">Checking what's installed…</p>
      </Shell>
    );
  }

  const noModels = CATEGORIES.every((c) => models[c].length === 0);

  return (
    <Shell>
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      <header className="mb-8">
        <h1 className="text-3xl font-semibold text-foreground mb-3">Set up loqmi</h1>
        <p className="text-muted-foreground max-w-prose">
          loqmi runs AI models on your computer, so your chats and files stay here. Pick what to
          download first. You can add or swap models later in Settings.
        </p>
      </header>

      {noModels ? (
        <ErrorBanner message="The model list is empty. Open Settings and refresh the catalog, or check your internet connection." />
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 mb-4" role="radiogroup" aria-label="Setup type">
            <BundleCard
              selected={bundle === "essentials"}
              disabled={isDownloading}
              title="Essentials"
              detail="One general-purpose model. Smallest download, handles everything."
              onSelect={() => applyBundle("essentials")}
            />
            <BundleCard
              selected={bundle === "full"}
              disabled={isDownloading}
              title="Full set"
              detail="Adds specialist models for code, reasoning and documents."
              onSelect={() => applyBundle("full")}
            />
          </div>

          <button
            type="button"
            onClick={() => setShowCustomize((v) => !v)}
            aria-expanded={showCustomize}
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground mb-4"
          >
            {showCustomize ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            Choose models yourself
          </button>

          {showCustomize && (
            <div className="space-y-8 mb-8">
              {CATEGORIES.map((c) => (
                <CategoryPicker
                  key={c}
                  category={c}
                  models={models[c]}
                  value={picks[c]}
                  disabled={isDownloading}
                  progress={progress}
                  onChange={(id) => setPicks((p) => ({ ...p, [c]: id }))}
                />
              ))}
            </div>
          )}

          {isDownloading && (
            <div className="space-y-3 mb-6">
              {toDownload.map(({ model }) => (
                <ProgressBar
                  key={model.id}
                  value={progress[model.id] ?? 0}
                  label={model.name}
                  showPercent
                />
              ))}
            </div>
          )}

          <button
            onClick={start}
            disabled={isDownloading || !picks.chat}
            className="w-full py-3 px-4 bg-primary text-primary-foreground font-medium rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
          >
            <Download className="h-4 w-4" />
            {isDownloading
              ? "Downloading…"
              : toDownload.length
                ? `Download ${totalGb ? totalGb.toFixed(1) + " GB" : "models"} and start`
                : "Start chatting"}
          </button>

          <p className="text-xs text-muted-foreground text-center mt-4">
            Models download once from Hugging Face. After that, loqmi works offline.
          </p>
        </>
      )}
    </Shell>
  );
}

// ── helpers & small components ────────────────────────────────────────────────

function pickDefault(list: ModelInfo[]): string | null {
  const usable = list.filter(isDownloadable);
  return (usable.find((m) => m.recommended) ?? usable[0])?.id ?? null;
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main className="min-h-screen bg-background flex justify-center px-4 py-12">
      <div className="w-full max-w-2xl">{children}</div>
    </main>
  );
}

function BundleCard(props: {
  selected: boolean; disabled: boolean; title: string; detail: string; onSelect: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={props.selected}
      disabled={props.disabled}
      onClick={props.onSelect}
      className={`text-left p-4 rounded-lg border transition-colors disabled:opacity-60 ${
        props.selected ? "border-primary bg-secondary" : "border-border bg-card hover:bg-secondary"
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="font-medium text-foreground">{props.title}</span>
        {props.selected && <Check className="h-4 w-4 text-primary" />}
      </div>
      <p className="text-sm text-muted-foreground">{props.detail}</p>
    </button>
  );
}

function CategoryPicker(props: {
  category: Category;
  models: ModelInfo[];
  value: string | null;
  disabled: boolean;
  progress: Record<string, number>;
  onChange: (id: string | null) => void;
}) {
  const info = CATEGORY_INFO[props.category];
  const name = `pick-${props.category}`;
  const options = props.models.filter(isDownloadable);

  return (
    <fieldset disabled={props.disabled}>
      <legend className="text-sm font-medium text-foreground mb-1">{info.title}</legend>
      <p className="text-sm text-muted-foreground mb-3">{info.blurb}</p>

      <div className="space-y-2">
        {!info.required && (
          <label className="flex items-center gap-3 p-3 border border-border rounded-lg bg-card cursor-pointer has-[:checked]:border-primary">
            <input
              type="radio"
              name={name}
              checked={props.value === null}
              onChange={() => props.onChange(null)}
              className="accent-primary"
            />
            <span className="text-sm text-foreground">Use the general chat model for this</span>
          </label>
        )}

        {options.map((m) => (
          <label
            key={m.id}
            className="flex items-start gap-3 p-3 border border-border rounded-lg bg-card cursor-pointer hover:bg-secondary has-[:checked]:border-primary"
          >
            <input
              type="radio"
              name={name}
              checked={props.value === m.id}
              onChange={() => props.onChange(m.id)}
              className="mt-1 accent-primary"
            />
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium text-foreground">{m.name}</span>
                {m.recommended && <StatusBadge text="Recommended" variant="active" />}
                {m.downloaded && <StatusBadge text="Installed" variant="success" />}
              </div>
              {m.description && <p className="text-sm text-muted-foreground mt-1">{m.description}</p>}
              <div className="flex flex-wrap gap-2 mt-2">
                <StatusBadge text={`${m.size_gb} GB`} />
                {m.ram_required && <StatusBadge text={m.ram_required} />}
              </div>
            </div>
          </label>
        ))}
      </div>
    </fieldset>
  );
}