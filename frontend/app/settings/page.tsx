"use client";

import { useEffect, useState } from "react";
import {
  setupAPI,
  contextWindowAPI,
  BackendError,
  ModelInfo,
  CategoryContextInfo,
  SuggestedModel,
  HuggingFaceRepo,
  SetupStatus,
} from "@/lib/api";
import { useConnection } from "@/lib/connection-context";
import { ErrorBanner } from "@/components/error-banner";
import { ProgressBar } from "@/components/progress-bar";
import { StatusBadge } from "@/components/status-badge";
import {
  Trash2,
  Download,
  Key,
  CheckCircle2,
  Cpu,
  HardDrive,
  BrainCircuit,
  Globe,
  SlidersHorizontal,
  Search
} from "lucide-react";
import { Sidebar } from "@/components/sidebar";

// Routable categories (classifier labels, context windows, Hugging Face assignment)
const CATEGORIES = [
  "chat",
  "code",
  "reasoning",
  "document",
] as const;
type Category = (typeof CATEGORIES)[number];

// Everything that has downloadable models. "helper" is NOT a routing category.
const MODEL_CATEGORIES = [...CATEGORIES, "helper"] as const;
type ModelCategory = (typeof MODEL_CATEGORIES)[number];

const CATEGORY_LABEL: Record<ModelCategory, string> = {
  chat: "Chat",
  code: "Code",
  reasoning: "Reasoning",
  document: "Document",
  helper: "Search Helper",
};

const toModelMap = (status: SetupStatus) => {
  const raw = status.models as Record<string, ModelInfo[] | undefined>;
  const out = {} as Record<ModelCategory, ModelInfo[]>;
  MODEL_CATEGORIES.forEach((cat) => {
    out[cat] = raw[cat] ?? [];
  });
  return out;
};

const formatTokens = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : String(n));

export default function SettingsPage() {
  const { isConnected } = useConnection();

  // State
  const [models, setModels] = useState<Record<ModelCategory, ModelInfo[]>>({
    chat: [],
    code: [],
    reasoning: [],
    document: [],
    helper: [],
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Tabs
  const [activeTab, setActiveTab] = useState<"library" | "catalog">("library");

  // Catalog Sub-tabs & Hugging Face State
  const [catalogView, setCatalogView] = useState<"suggested" | "huggingface">("suggested");
  const [suggestedModels, setSuggestedModels] = useState<SuggestedModel[]>([]);

  // HF Search State
  const [hfQuery, setHfQuery] = useState("");
  const [hfIsSearching, setHfIsSearching] = useState(false);
  const [hfResults, setHfResults] = useState<HuggingFaceRepo[]>([]);

  // HF Download Flow State
  const [hfSelectedRepo, setHfSelectedRepo] = useState<HuggingFaceRepo | null>(null);
  const [hfFiles, setHfFiles] = useState<string[]>([]);
  const [hfSelectedFile, setHfSelectedFile] = useState<string>("");
  const [hfOverrideCategory, setHfOverrideCategory] = useState<Category | "auto">("auto");

  // Downloading & Deleting State
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const [processingId, setProcessingId] = useState<string | null>(null);

  // Token State
  const [hfToken, setHfToken] = useState("");
  const [hasSavedToken, setHasSavedToken] = useState(false);
  const [isSavingToken, setIsSavingToken] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Search Key State
  const [braveKey, setBraveKey] = useState("");
  const [hasSavedBraveKey, setHasSavedBraveKey] = useState(false);
  const [isSavingBraveKey, setIsSavingBraveKey] = useState(false);

  const [tavilyKey, setTavilyKey] = useState("");
  const [hasSavedTavilyKey, setHasSavedTavilyKey] = useState(false);
  const [isSavingTavilyKey, setIsSavingTavilyKey] = useState(false);

  // Context Window State (routable categories only; the search helper has a fixed window)
  const [contextWindows, setContextWindows] = useState<Record<Category, CategoryContextInfo | null>>({
    chat: null,
    code: null,
    reasoning: null,
    document: null,
  });
  const [savingContextWindow, setSavingContextWindow] = useState<Category | null>(null);

  // ── INIT ────────────────────────────────────────────────────────
  useEffect(() => {
    const fetchData = async () => {
      try {
        setError(null);
        const [statusData, tokenData, searchKeysData, contextWindowData, catalogData] = await Promise.all([
          setupAPI.getStatus(),
          setupAPI.getToken(),
          setupAPI.getSearchKeys(),
          contextWindowAPI.getAll(),
          setupAPI.getSuggestedModels().catch(() => []) // Fallback to empty if not fetched yet
        ]);

        setModels(toModelMap(statusData));
        setSuggestedModels(catalogData);

        setHasSavedToken(tokenData.has_token);
        if (tokenData.has_token && tokenData.masked_token) {
          setHfToken(tokenData.masked_token);
        }

        setHasSavedBraveKey(searchKeysData.brave.has_key);
        if (searchKeysData.brave.has_key && searchKeysData.brave.masked_key) {
          setBraveKey(searchKeysData.brave.masked_key);
        }

        setHasSavedTavilyKey(searchKeysData.tavily.has_key);
        if (searchKeysData.tavily.has_key && searchKeysData.tavily.masked_key) {
          setTavilyKey(searchKeysData.tavily.masked_key);
        }

        const cwMap: Record<Category, CategoryContextInfo | null> = {
          chat: null,
          code: null,
          reasoning: null,
          document: null,
        };
        contextWindowData.forEach((info) => {
          if (info.category in cwMap) {
            cwMap[info.category as Category] = info;
          }
        });
        setContextWindows(cwMap);
      } catch (err) {
        setError(err instanceof BackendError ? err.message : "Failed to load settings data.");
      } finally {
        setIsLoading(false);
      }
    };

    if (isConnected) fetchData();
  }, [isConnected]);

  // ── HANDLERS ────────────────────────────────────────────────────
  const handleSaveToken = async () => {
    if (!hfToken.trim()) return;
    setIsSavingToken(true);
    try {
      await setupAPI.saveToken(hfToken);
      setHasSavedToken(true);
      setHfToken(`${hfToken.slice(0, 4)}...${hfToken.slice(-4)}`);
    } catch (err) {
      setError("Failed to save Hugging Face token.");
    } finally {
      setIsSavingToken(false);
    }
  };

  const handleSaveBraveKey = async () => {
    if (!braveKey.trim()) return;
    setIsSavingBraveKey(true);
    try {
      await setupAPI.saveSearchKey("brave", braveKey);
      setHasSavedBraveKey(true);
      setBraveKey(`${braveKey.slice(0, 4)}...${braveKey.slice(-4)}`);
    } catch (err) {
      setError("Failed to save Brave Search API key.");
    } finally {
      setIsSavingBraveKey(false);
    }
  };

  const handleSaveTavilyKey = async () => {
    if (!tavilyKey.trim()) return;
    setIsSavingTavilyKey(true);
    try {
      await setupAPI.saveSearchKey("tavily", tavilyKey);
      setHasSavedTavilyKey(true);
      setTavilyKey(`${tavilyKey.slice(0, 4)}...${tavilyKey.slice(-4)}`);
    } catch (err) {
      setError("Failed to save Tavily API key.");
    } finally {
      setIsSavingTavilyKey(false);
    }
  };

  const handleActivateModel = async (category: ModelCategory, modelId: string) => {
    try {
      setError(null);
      await setupAPI.selectModel(category, modelId);
      setModels((prev) => ({
        ...prev,
        [category]: prev[category].map((m) => ({
          ...m,
          is_active: m.id === modelId,
        })),
      }));
    } catch (err) {
      setError(err instanceof BackendError ? err.message : "Failed to activate model");
    }
  };

  const handleDeleteModel = async (category: ModelCategory, modelId: string) => {
    if (!confirm("Delete this model? This action cannot be undone.")) return;
    try {
      setError(null);
      setProcessingId(modelId);
      await setupAPI.deleteModel(category, modelId);

      const statusData = await setupAPI.getStatus();
      setModels(toModelMap(statusData));
    } catch (err) {
      setError(err instanceof BackendError ? err.message : "Failed to delete model");
    } finally {
      setProcessingId(null);
    }
  };

  const handleDownload = async (category: ModelCategory, modelId: string) => {
    try {
      setError(null);
      setDownloadProgress((prev) => ({ ...prev, [modelId]: 0 }));

      for await (const event of setupAPI.downloadModel(category, modelId)) {
        if (event.type === "progress") {
          setDownloadProgress((prev) => ({
            ...prev,
            [modelId]: event.percent,
          }));
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }

      const statusData = await setupAPI.getStatus();
      setModels(toModelMap(statusData));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed.");
    } finally {
      setDownloadProgress((prev) => {
        const next = { ...prev };
        delete next[modelId];
        return next;
      });
    }
  };

  const handleRefreshCatalog = async () => {
    try {
      setError(null);
      await setupAPI.refreshCatalog();
      setSuggestedModels(await setupAPI.getSuggestedModels());
      setModels(toModelMap(await setupAPI.getStatus()));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh catalog.");
    }
  };

  const handleSetContextWindow = async (category: Category, value: number) => {
    const previous = contextWindows[category];
    setError(null);
    setSavingContextWindow(category);

    setContextWindows((prev) => ({
      ...prev,
      [category]: prev[category] ? { ...prev[category]!, current: value } : prev[category],
    }));

    try {
      await contextWindowAPI.setValue(category, value);
    } catch (err) {
      setContextWindows((prev) => ({ ...prev, [category]: previous }));
      setError(err instanceof BackendError ? err.message : "Failed to update context window");
    } finally {
      setSavingContextWindow(null);
    }
  };

  // ── HUGGING FACE HANDLERS ───────────────────────────────────────
  const handleHfSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hfQuery.trim()) return;

    setHfIsSearching(true);
    setHfSelectedRepo(null);
    try {
      const results = await setupAPI.searchHuggingFace(hfQuery);
      setHfResults(results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to search Hugging Face.");
    } finally {
      setHfIsSearching(false);
    }
  };

  const handleSelectHfRepo = async (repo: HuggingFaceRepo) => {
    setHfSelectedRepo(repo);
    setHfFiles([]);
    try {
      const files = await setupAPI.getHuggingFaceFiles(repo.repo_id);
      setHfFiles(files);
      if (files.length > 0) setHfSelectedFile(files[0]);
    } catch (err) {
      setError("Failed to fetch files for this repository.");
    }
  };

  const handleDownloadHfModel = async () => {
    if (!hfSelectedRepo || !hfSelectedFile) return;

    let targetCategory = hfOverrideCategory;
    if (targetCategory === "auto") {
      const searchString = hfSelectedRepo.repo_id.toLowerCase();
      if (searchString.includes("code") || searchString.includes("coder")) targetCategory = "code";
      else if (searchString.includes("reason") || searchString.includes("think") || searchString.includes("r1")) targetCategory = "reasoning";
      else targetCategory = "chat";
    }

    await handleDownload(targetCategory as Category, hfSelectedRepo.repo_id);
    setHfSelectedRepo(null);
  };

  // ── RENDERERS ───────────────────────────────────────────────────
  if (!isConnected)
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="max-w-2xl w-full">
          <ErrorBanner message="Can't reach loqmi backend." />
        </div>
      </div>
    );

  if (isLoading)
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-muted-foreground">Loading settings...</div>
      </div>
    );

  const activeHelper = models.helper.find((m) => m.downloaded && m.is_active);
  const installedIds = new Set(
    MODEL_CATEGORIES.flatMap((c) => models[c].filter((m) => m.downloaded).map((m) => m.id)),
  );

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <div className="flex-1 overflow-y-auto pb-20">
        <div className="max-w-5xl mx-auto p-8">
          {error && (
            <div className="mb-6">
              <ErrorBanner message={error} onDismiss={() => setError(null)} />
            </div>
          )}

          <div className="mb-10">
            <h1 className="text-3xl font-bold text-foreground mb-2">Settings & Models</h1>
            <p className="text-muted-foreground text-sm">
              Manage your local AI environment. Download specialized models and configure active defaults.
            </p>
          </div>

          {/* ── HF TOKEN MANAGER ── */}
          <div className="bg-card border border-border rounded-lg p-6 mb-10 shadow-sm">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-secondary rounded-full">
                <Key className="h-5 w-5 text-foreground" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-foreground">Hugging Face API Token</h3>
                <p className="text-xs text-muted-foreground mt-1 mb-4">
                  Required for faster download speeds and accessing gated models. Your token is stored securely in your local database.
                </p>
                <div className="flex gap-3 max-w-md">
                  <input
                    type="password"
                    placeholder="hf_..."
                    value={hfToken}
                    onChange={(e) => {
                      setHfToken(e.target.value);
                      setHasSavedToken(false);
                    }}
                    className="flex-1 px-3 py-2 bg-background border border-border rounded-md text-sm text-foreground focus:outline-none focus:border-primary"
                  />
                  <button
                    onClick={handleSaveToken}
                    disabled={isSavingToken || !hfToken.trim()}
                    className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-md hover:bg-primary/90 disabled:opacity-50 transition-colors"
                  >
                    {isSavingToken ? "Saving..." : hasSavedToken ? "Update" : "Save"}
                  </button>
                </div>
                {hasSavedToken && (
                  <p className="text-xs text-green-500 mt-2 flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3" /> Token active
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* ── SEARCH API KEYS MANAGER ── */}
          <div className="bg-card border border-border rounded-lg p-6 mb-10 shadow-sm">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-secondary rounded-full">
                <Globe className="h-5 w-5 text-foreground" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-foreground">Internet Search API Keys</h3>
                <p className="text-xs text-muted-foreground mt-1 mb-4">
                  Optional. Without a key, internet search falls back to free DuckDuckGo/SearXNG search. Adding a key unlocks that provider as a more reliable option.
                </p>
                <p className="text-xs text-muted-foreground mb-4">
                  <span className="font-medium text-foreground">Search helper: </span>
                  {activeHelper
                    ? activeHelper.name
                    : "none installed. Your chat model will do the research (download one from the Model Catalog)."}
                </p>

                <div className="mb-5">
                  <label className="text-xs font-medium text-muted-foreground mb-2 block">Brave Search API Key</label>
                  <div className="flex gap-3 max-w-md">
                    <input
                      type="password"
                      placeholder="BSA..."
                      value={braveKey}
                      onChange={(e) => {
                        setBraveKey(e.target.value);
                        setHasSavedBraveKey(false);
                      }}
                      className="flex-1 px-3 py-2 bg-background border border-border rounded-md text-sm text-foreground focus:outline-none focus:border-primary"
                    />
                    <button
                      onClick={handleSaveBraveKey}
                      disabled={isSavingBraveKey || !braveKey.trim()}
                      className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-md hover:bg-primary/90 disabled:opacity-50 transition-colors"
                    >
                      {isSavingBraveKey ? "Saving..." : hasSavedBraveKey ? "Update" : "Save"}
                    </button>
                  </div>
                  {hasSavedBraveKey && (
                    <p className="text-xs text-green-500 mt-2 flex items-center gap-1">
                      <CheckCircle2 className="h-3 w-3" /> Key active
                    </p>
                  )}
                </div>

                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-2 block">Tavily API Key</label>
                  <div className="flex gap-3 max-w-md">
                    <input
                      type="password"
                      placeholder="tvly-..."
                      value={tavilyKey}
                      onChange={(e) => {
                        setTavilyKey(e.target.value);
                        setHasSavedTavilyKey(false);
                      }}
                      className="flex-1 px-3 py-2 bg-background border border-border rounded-md text-sm text-foreground focus:outline-none focus:border-primary"
                    />
                    <button
                      onClick={handleSaveTavilyKey}
                      disabled={isSavingTavilyKey || !tavilyKey.trim()}
                      className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-md hover:bg-primary/90 disabled:opacity-50 transition-colors"
                    >
                      {isSavingTavilyKey ? "Saving..." : hasSavedTavilyKey ? "Update" : "Save"}
                    </button>
                  </div>
                  {hasSavedTavilyKey && (
                    <p className="text-xs text-green-500 mt-2 flex items-center gap-1">
                      <CheckCircle2 className="h-3 w-3" /> Key active
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* ── CONTEXT WINDOW MANAGER ── */}
          <div className="bg-card border border-border rounded-lg p-6 mb-10 shadow-sm">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-secondary rounded-full">
                <SlidersHorizontal className="h-5 w-5 text-foreground" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-foreground">Context Window</h3>
                <p className="text-xs text-muted-foreground mt-1 mb-4">
                  How much conversation history each model category can hold at once. Larger windows use more memory and take longer to load.
                </p>

                <div className="divide-y divide-border/50">
                  {CATEGORIES.map((category) => {
                    const info = contextWindows[category];
                    if (!info) return null;
                    const isSaving = savingContextWindow === category;

                    return (
                      <div key={category} className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 py-3 first:pt-0 last:pb-0">
                        <div>
                          <p className="text-sm font-medium text-foreground capitalize">{category}</p>
                          <p className="text-xs text-muted-foreground">
                            {info.model_name} · supports up to {formatTokens(info.ceiling)} tokens
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          {isSaving && <span className="text-xs text-muted-foreground">Saving...</span>}
                          <select
                            value={info.current}
                            disabled={isSaving}
                            onChange={(e) => handleSetContextWindow(category, Number(e.target.value))}
                            className="px-3 py-1.5 bg-background border border-border rounded-md text-sm text-foreground focus:outline-none focus:border-primary disabled:opacity-50 min-w-[140px]"
                          >
                            {info.options.map((opt) => (
                              <option key={opt} value={opt}>
                                {formatTokens(opt)}
                                {opt === info.recommended ? " (Recommended)" : ""}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* ── TABS ── */}
          <div className="flex border-b border-border mb-8">
            <button
              onClick={() => setActiveTab("library")}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === "library"
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              My Local Library
            </button>
            <button
              onClick={() => setActiveTab("catalog")}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === "catalog"
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              Model Catalog
            </button>
          </div>

          {/* ── TAB CONTENT ── */}
          {activeTab === "library" ? (
            <div className="space-y-12">
              {MODEL_CATEGORIES.map((category) => {
                const catModels = models[category].filter((m) => m.downloaded);
                if (catModels.length === 0) return null;

                return (
                  <div key={category} className="space-y-4">
                    <div className="border-b border-border pb-2">
                      <h3 className="text-sm uppercase tracking-wider font-mono text-muted-foreground">
                        {CATEGORY_LABEL[category]} Models
                      </h3>
                      {category === "helper" && (
                        <p className="text-xs text-muted-foreground mt-1">
                          Optional. Plans and summarises web searches in parallel so your main model stays loaded.
                          If none is installed, the chat model does this job.
                        </p>
                      )}
                    </div>

                    <div className="grid grid-cols-1 gap-4">
                      {catModels.map((model) => (
                        <div key={model.id} className="flex flex-col md:flex-row md:items-start gap-4 p-5 border border-border rounded-lg bg-card hover:border-border/80 transition-colors">
                          <div className="pt-1">
                            <input
                              type="radio"
                              name={`active-${category}`}
                              checked={model.is_active}
                              onChange={() => handleActivateModel(category, model.id)}
                              className="h-4 w-4 accent-primary cursor-pointer"
                            />
                          </div>

                          <div className="flex-1 space-y-3">
                            <div className="flex items-center gap-3">
                              <h4 className="font-semibold text-base text-foreground">{model.name}</h4>
                              {model.recommended && <StatusBadge text="Recommended" variant="active" />}
                              {model.is_active && <StatusBadge text="Active Default" variant="success" />}
                            </div>
                            <p className="text-sm text-muted-foreground">{model.description || "No description provided."}</p>
                          </div>

                          <div className="flex flex-col items-end justify-center min-w-[120px]">
                            <button
                              onClick={() => handleDeleteModel(category, model.id)}
                              disabled={processingId === model.id}
                              className="p-2 text-muted-foreground hover:text-destructive transition-colors disabled:opacity-50 mt-1"
                              title="Delete from disk"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="space-y-6">

              {/* Sub-tab Toggle */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex gap-2 p-1 bg-muted rounded-lg w-fit">
                  <button
                    onClick={() => setCatalogView("suggested")}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${catalogView === "suggested" ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground"}`}
                  >
                    Curated Catalog
                  </button>
                  <button
                    onClick={() => setCatalogView("huggingface")}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${catalogView === "huggingface" ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground"}`}
                  >
                    Hugging Face Search
                  </button>
                </div>
                {catalogView === "suggested" && (
                  <button
                    onClick={handleRefreshCatalog}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Refresh catalog
                  </button>
                )}
              </div>

              {catalogView === "suggested" && (
                <div className="grid grid-cols-1 gap-4">
                  {suggestedModels.map((model) => (
                    <div key={model.id} className="flex flex-col md:flex-row gap-4 p-5 border border-border rounded-lg bg-card">
                       <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h4 className="font-semibold text-foreground">{model.name}</h4>
                            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-secondary text-muted-foreground">{model.category}</span>
                            {model.recommended && <StatusBadge text="Recommended" variant="active" />}
                          </div>
                          <p className="text-sm text-muted-foreground mb-3">{model.description}</p>
                          <div className="flex flex-wrap gap-4 pt-2">
                            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                              <HardDrive className="h-3.5 w-3.5" />
                              <span>{model.size_gb} GB</span>
                            </div>
                            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                              <Cpu className="h-3.5 w-3.5" />
                              <span>{model.ram_required || "Unknown RAM"}</span>
                            </div>
                            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                              <BrainCircuit className="h-3.5 w-3.5" />
                              <span>{model.n_ctx ? `${(model.n_ctx / 1000).toFixed(0)}k Context` : "Unknown Context"}</span>
                            </div>
                          </div>
                       </div>

                       <div className="flex items-center justify-end min-w-[120px]">
                         {installedIds.has(model.id) ? (
                           <span className="text-xs text-green-500 flex items-center gap-1">
                             <CheckCircle2 className="h-3.5 w-3.5" /> Installed
                           </span>
                         ) : downloadProgress[model.id] !== undefined ? (
                           <div className="w-full max-w-[150px]">
                              <ProgressBar value={downloadProgress[model.id]} label="Downloading..." showPercent />
                           </div>
                         ) : (
                           <button
                             onClick={() => handleDownload(model.category as ModelCategory, model.id)}
                             className="px-4 py-2 bg-secondary text-foreground hover:bg-secondary/80 text-sm font-medium rounded-md transition-colors flex items-center gap-2"
                           >
                             <Download className="h-4 w-4" /> Download
                           </button>
                         )}
                       </div>
                    </div>
                  ))}
                </div>
              )}

              {catalogView === "huggingface" && (
                <div className="space-y-6">
                  <form onSubmit={handleHfSearch} className="flex gap-2">
                    <input
                      type="text"
                      value={hfQuery}
                      onChange={(e) => setHfQuery(e.target.value)}
                      placeholder="Search for models (e.g., Llama 3 8B GGUF)"
                      className="flex-1 px-4 py-2 bg-background border border-border rounded-md text-foreground focus:outline-none focus:border-primary"
                    />
                    <button type="submit" disabled={hfIsSearching} className="px-6 py-2 bg-primary text-primary-foreground rounded-md flex items-center gap-2">
                      <Search className="h-4 w-4" />
                      {hfIsSearching ? "Searching..." : "Search Hub"}
                    </button>
                  </form>

                  {/* Hugging Face Results List */}
                  {!hfSelectedRepo ? (
                    <div className="grid grid-cols-1 gap-3">
                      {hfResults.map((repo) => (
                        <div key={repo.repo_id} onClick={() => handleSelectHfRepo(repo)} className="p-4 border border-border rounded-lg cursor-pointer hover:border-primary/50 bg-card transition-colors">
                          <h4 className="font-medium text-foreground">{repo.repo_id}</h4>
                          <p className="text-xs text-muted-foreground mt-1">{repo.downloads.toLocaleString()} downloads</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    /* The Download Configuration Modal/View */
                    <div className="p-6 border border-border bg-card rounded-lg space-y-5">
                      <div className="flex justify-between items-start border-b border-border pb-4">
                        <div>
                          <h4 className="font-semibold text-lg text-foreground">{hfSelectedRepo.repo_id}</h4>
                          <p className="text-xs text-muted-foreground mt-1">Configure your download for this custom model.</p>
                        </div>
                        <button onClick={() => setHfSelectedRepo(null)} className="text-sm text-muted-foreground hover:text-foreground transition-colors">Cancel</button>
                      </div>

                      <div className="space-y-4">
                        <div>
                          <label className="text-sm font-medium mb-1.5 block text-foreground">1. Select Quantization File</label>
                          <select
                            value={hfSelectedFile}
                            onChange={(e) => setHfSelectedFile(e.target.value)}
                            className="w-full px-3 py-2 bg-background border border-border rounded-md text-sm text-foreground focus:outline-none focus:border-primary"
                          >
                            {hfFiles.length === 0 ? <option>Loading files...</option> : null}
                            {hfFiles.map(f => <option key={f} value={f}>{f}</option>)}
                          </select>
                          <p className="text-xs text-muted-foreground mt-1">Select the specific .gguf file. Q4_K_M is generally recommended for balance.</p>
                        </div>

                        <div>
                          <label className="text-sm font-medium mb-1.5 block text-foreground">2. Assign Category</label>
                          <select
                            value={hfOverrideCategory}
                            onChange={(e) => setHfOverrideCategory(e.target.value as Category | "auto")}
                            className="w-full px-3 py-2 bg-background border border-border rounded-md text-sm text-foreground focus:outline-none focus:border-primary"
                          >
                            <option value="auto">Auto-detect from name</option>
                            {CATEGORIES.map(c => <option key={c} value={c} className="capitalize">{c}</option>)}
                          </select>
                          <p className="text-xs text-muted-foreground mt-1">loqmi routes prompts based on this category.</p>
                        </div>
                      </div>

                      <div className="pt-2">
                        {downloadProgress[hfSelectedRepo.repo_id] !== undefined ? (
                           <ProgressBar value={downloadProgress[hfSelectedRepo.repo_id]} label="Downloading custom model..." showPercent />
                        ) : (
                          <button onClick={handleDownloadHfModel} className="w-full py-2.5 bg-primary text-primary-foreground font-medium rounded-md hover:bg-primary/90 transition-colors flex items-center justify-center gap-2">
                            <Download className="h-4 w-4" />
                            Download Custom Model
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}