"use client";

import { useEffect, useState, useRef, useCallback, Suspense } from "react";
import { documentsAPI, DocumentMetadata, BackendError } from "@/lib/api";
import { useConnection } from "@/lib/connection-context";
import { ErrorBanner } from "@/components/error-banner";
import {
  Trash2,
  UploadCloud,
  FileText,
  Database,
  CheckSquare,
  Square,
} from "lucide-react";
import { Sidebar } from "@/components/sidebar";

// Helper functions for formatting data
const formatBytes = (bytes: number) => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
};

const formatDate = (isoString: string) => {
  return new Date(isoString).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export default function DocumentsPage() {
  const { isConnected } = useConnection();
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Bulk selection state
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());

  // Drag and drop state
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDocuments = useCallback(async () => {
    try {
      setError(null);
      const data = await documentsAPI.listDocuments();
      setDocuments(data.documents || []);
      // Clear selection if documents were deleted elsewhere
      setSelectedFiles(new Set());
    } catch (err) {
      setError(
        err instanceof BackendError ? err.message : "Failed to load documents",
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isConnected) loadDocuments();
  }, [isConnected, loadDocuments]);

  // ── Upload Handlers ──
  const processFiles = async (files: FileList | File[]) => {
    if (files.length === 0) return;

    setIsUploading(true);
    setError(null);

    let successCount = 0;
    for (let i = 0; i < files.length; i++) {
      try {
        await documentsAPI.uploadDocument(files[i]);
        successCount++;
      } catch (err) {
        setError(
          `Failed to upload ${files[i].name}. Proceeding with remaining...`,
        );
      }
    }

    setIsUploading(false);
    if (successCount > 0) loadDocuments(); // Refresh table
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    await processFiles(e.dataTransfer.files);
  };

  // ── Selection Handlers ──
  const toggleSelection = (filename: string) => {
    const newSelection = new Set(selectedFiles);
    if (newSelection.has(filename)) newSelection.delete(filename);
    else newSelection.add(filename);
    setSelectedFiles(newSelection);
  };

  const toggleSelectAll = () => {
    if (selectedFiles.size === documents.length) {
      setSelectedFiles(new Set());
    } else {
      setSelectedFiles(new Set(documents.map((d) => d.filename)));
    }
  };

  // ── Delete Handler ──
  const handleDeleteSelected = async () => {
    if (selectedFiles.size === 0) return;
    if (
      !confirm(
        `Are you sure you want to delete ${selectedFiles.size} document(s)? This removes them from your local knowledge base.`,
      )
    )
      return;

    setIsLoading(true);
    setError(null);

    for (const filename of Array.from(selectedFiles)) {
      try {
        await documentsAPI.deleteDocument(filename);
      } catch (err) {
        console.error(`Failed to delete ${filename}`, err);
      }
    }

    await loadDocuments();
  };

  // ── Render ──
  if (!isConnected) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <ErrorBanner message="Can't reach loqmi backend." />
      </div>
    );
  }

  const allSelected =
    documents.length > 0 && selectedFiles.size === documents.length;

  return (
    <Suspense
      fallback={
        <div className="h-screen bg-background flex items-center justify-center text-muted-foreground">
          Loading Knowledge Base...
        </div>
      }
    >
      <div className="flex h-screen bg-background overflow-hidden">
        <Sidebar
          isOpen={sidebarOpen}
          onToggle={() => setSidebarOpen(!sidebarOpen)}
        />
        <div className="flex-1 overflow-y-auto pb-20">
          <div className="max-w-5xl mx-auto p-8">
            {error && (
              <div className="mb-6">
                <ErrorBanner message={error} onDismiss={() => setError(null)} />
              </div>
            )}

            <div className="flex items-center justify-between mb-8">
              <div>
                <h1 className="text-3xl font-bold text-foreground mb-2 flex items-center gap-3">
                  <Database className="h-7 w-7 text-primary" />
                  Knowledge Base
                </h1>
                <p className="text-muted-foreground text-sm">
                  Manage documents in your local vector database. These files
                  are securely embedded and used for RAG generation.
                </p>
              </div>

              {selectedFiles.size > 0 && (
                <button
                  onClick={handleDeleteSelected}
                  className="px-4 py-2 bg-destructive/10 text-destructive border border-destructive/20 hover:bg-destructive hover:text-destructive-foreground font-medium rounded-md transition-colors flex items-center gap-2 text-sm"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete Selected ({selectedFiles.size})
                </button>
              )}
            </div>

            {/* ── Drag & Drop Upload Zone ── */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`mb-10 p-12 border-2 border-dashed rounded-xl flex flex-col items-center justify-center cursor-pointer transition-all ${
                isDragging
                  ? "border-primary bg-primary/5"
                  : "border-border bg-card hover:border-primary/50 hover:bg-secondary/50"
              }`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => e.target.files && processFiles(e.target.files)}
                className="hidden"
                multiple
                accept=".pdf,.txt,.md,.png,.jpg,.jpeg"
              />

              {isUploading ? (
                <div className="flex flex-col items-center animate-pulse">
                  <UploadCloud className="h-10 w-10 text-primary mb-4" />
                  <p className="font-semibold text-foreground">
                    Embedding Documents...
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    This takes heavy compute. Please wait.
                  </p>
                </div>
              ) : (
                <>
                  <UploadCloud className="h-10 w-10 text-muted-foreground mb-4" />
                  <p className="font-semibold text-foreground">
                    Click or drag files to embed
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Supports PDF, TXT, MD, PNG, JPG
                  </p>
                </>
              )}
            </div>

            {/* ── Document Table ── */}
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-secondary/50 border-b border-border">
                    <tr>
                      <th className="px-6 py-4 font-medium text-muted-foreground w-12">
                        <button
                          onClick={toggleSelectAll}
                          className="hover:text-foreground"
                        >
                          {allSelected ? (
                            <CheckSquare className="h-4 w-4" />
                          ) : (
                            <Square className="h-4 w-4" />
                          )}
                        </button>
                      </th>
                      <th className="px-6 py-4 font-medium text-muted-foreground">
                        Document Name
                      </th>
                      <th className="px-6 py-4 font-medium text-muted-foreground">
                        File Size
                      </th>
                      <th className="px-6 py-4 font-medium text-muted-foreground">
                        Vector Chunks
                      </th>
                      <th className="px-6 py-4 font-medium text-muted-foreground">
                        Date Embedded
                      </th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-border">
                    {isLoading ? (
                      <tr>
                        <td
                          colSpan={5}
                          className="px-6 py-8 text-center text-muted-foreground"
                        >
                          Loading knowledge base...
                        </td>
                      </tr>
                    ) : documents.length === 0 ? (
                      <tr>
                        <td
                          colSpan={5}
                          className="px-6 py-12 text-center text-muted-foreground"
                        >
                          <FileText className="h-8 w-8 mx-auto mb-3 opacity-20" />
                          No documents in the database.
                          <br />
                          Upload files above to start using RAG.
                        </td>
                      </tr>
                    ) : (
                      documents.map((doc) => {
                        const isSelected = selectedFiles.has(doc.filename);
                        return (
                          <tr
                            key={doc.filename}
                            className={`transition-colors hover:bg-secondary/30 ${isSelected ? "bg-primary/5" : ""}`}
                            onClick={() => toggleSelection(doc.filename)}
                          >
                            <td
                              className="px-6 py-4 cursor-pointer"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                onClick={() => toggleSelection(doc.filename)}
                                className="text-muted-foreground hover:text-foreground"
                              >
                                {isSelected ? (
                                  <CheckSquare className="h-4 w-4 text-primary" />
                                ) : (
                                  <Square className="h-4 w-4" />
                                )}
                              </button>
                            </td>
                            <td className="px-6 py-4 font-medium text-foreground flex items-center gap-3">
                              <FileText className="h-4 w-4 text-muted-foreground" />
                              {doc.filename}
                            </td>
                            <td className="px-6 py-4 text-muted-foreground font-mono text-xs">
                              {formatBytes(doc.file_size_bytes)}
                            </td>
                            <td className="px-6 py-4">
                              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-secondary text-secondary-foreground border border-border">
                                {doc.chunk_count} chunks
                              </span>
                            </td>
                            <td className="px-6 py-4 text-muted-foreground text-xs">
                              {formatDate(doc.upload_date)}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Suspense>
  );
}
