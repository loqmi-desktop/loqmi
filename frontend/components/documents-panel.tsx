'use client'

import { useState, useCallback, useEffect } from 'react'
import { Upload, FileText, Trash2 } from 'lucide-react'
import { documentsAPI, DocumentMetadata, BackendError } from '@/lib/api'
import { ErrorBanner } from './error-banner'

interface DocumentsPanelProps {
  isConnected: boolean
}

export function DocumentsPanel({ isConnected }: DocumentsPanelProps) {
  const [documents, setDocuments] = useState<DocumentMetadata[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null)

  // Load documents on mount
  const loadDocuments = useCallback(async () => {
    try {
      setError(null)
      const data = await documentsAPI.listDocuments()
      setDocuments(data.documents)
    } catch (err) {
      if (err instanceof BackendError) {
        setError(err.message)
      } else {
        setError('Failed to load documents')
      }
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Initial load
  useEffect(() => {
    if (isConnected && isLoading) {
      loadDocuments()
    }
  }, [isConnected, isLoading, loadDocuments])

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return

    const file = files[0]
    if (!file.name.match(/\.(pdf|txt|md|png|jpg|jpeg)$/i)) {
      setError('Only PDF, TXT, MD, PNG, and JPG files are supported')
      return
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File must be smaller than 50MB')
      return
    }

    try {
      setError(null)
      setIsUploading(true)
      await documentsAPI.uploadDocument(file)
      await loadDocuments()
    } catch (err) {
      if (err instanceof BackendError) {
        setError(err.message)
      } else {
        setError('Upload failed')
      }
    } finally {
      setIsUploading(false)
    }
  }

  const handleDelete = async (filename: string) => {
    if (!confirm('Delete this document?')) return

    try {
      setError(null)
      setDeletingFilename(filename)
      await documentsAPI.deleteDocument(filename)
      setDocuments((prev) => prev.filter((d) => d.filename !== filename))
    } catch (err) {
      if (err instanceof BackendError) {
        setError(err.message)
      } else {
        setError('Failed to delete document')
      }
    } finally {
      setDeletingFilename(null)
    }
  }

  if (!isConnected) {
    return (
      <div className="p-4">
        <ErrorBanner message="Backend unreachable" type="warning" />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {error && (
        <div className="p-4">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {/* Upload area */}
      <div className="p-4 border-b border-border">
        <label className="flex items-center justify-center w-full px-4 py-8 border-2 border-dashed border-border rounded-lg cursor-pointer hover:bg-secondary/50 transition-colors">
          <div className="flex flex-col items-center justify-center gap-2">
            <Upload className="h-6 w-6 text-muted-foreground" />
            <div className="text-center">
              <p className="text-xs font-medium text-foreground">Drop files or click to upload</p>
              <p className="text-xs text-muted-foreground">PDF, TXT, MD, PNG, JPG (Max 50MB)</p>
            </div>
          </div>
          <input
            type="file"
            className="hidden"
            onChange={(e) => handleFileUpload(e.target.files)}
            disabled={isUploading}
            accept=".pdf,.txt,.md,.png,.jpg,.jpeg"
          />
        </label>
      </div>

      {/* Documents list */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="text-xs text-muted-foreground">Loading documents...</div>
        ) : documents.length === 0 ? (
          <div className="text-xs text-muted-foreground">No documents yet</div>
        ) : (
          <div className="space-y-2">
            <h3 className="text-xs uppercase tracking-wider text-muted-foreground mb-3 font-mono">
              Indexed Files ({documents.length})
            </h3>
            {documents.map((doc) => (
              <div
                key={doc.filename}
                className="flex items-start gap-2 p-2 rounded border border-border/50 bg-card/50 hover:bg-card transition-colors group"
              >
                <FileText className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-foreground truncate">{doc.filename}</p>
                </div>
                <button
                  onClick={() => handleDelete(doc.filename)}
                  disabled={deletingFilename === doc.filename}
                  className="text-muted-foreground hover:text-destructive transition-colors p-1 opacity-0 group-hover:opacity-100 shrink-0 disabled:opacity-50"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>


    </div>
  )
}
