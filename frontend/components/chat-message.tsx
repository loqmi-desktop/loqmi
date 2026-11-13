'use client'

import { useState, memo } from 'react'
import { ChevronDown, Copy, Check, Maximize2, BrainCircuit, User, Bot, FileText } from 'lucide-react'
import { JetBrains_Mono } from 'next/font/google'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { attachmentsAPI, type ContentBlock } from '@/lib/api'

// Required CSS for Math rendering. KaTeX does not provide TypeScript declarations
// for its stylesheet, so suppress the side-effect import type check.
import 'katex/dist/katex.min.css'

const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], display: 'swap' })

interface StatsEvent {
  category: 'chat' | 'code' | 'reasoning' | 'document'
  model_name: string
  classify_ms: number
  inference_s: number
}

interface ChatMessageProps {
  role: 'user' | 'assistant'
  content: ContentBlock[]
  modelCategory?: string
  modelName?: string
  stats?: StatsEvent
}

const categoryLabel: Record<string, string> = {
  chat: 'Chat',
  code: 'Code',
  reasoning: 'Reasoning',
  document: 'Document'
}

const languageLabel: Record<string, string> = {
  cpp: 'C++', c: 'C', py: 'Python', python: 'Python', ts: 'TypeScript', tsx: 'TSX',
  js: 'JavaScript', jsx: 'JSX', javascript: 'JavaScript', typescript: 'TypeScript',
  rs: 'Rust', rust: 'Rust', go: 'Go', golang: 'Go', java: 'Java', kt: 'Kotlin',
  kotlin: 'Kotlin', swift: 'Swift', rb: 'Ruby', ruby: 'Ruby', php: 'PHP', cs: 'C#',
  csharp: 'C#', sh: 'Shell', bash: 'Bash', zsh: 'Shell', sql: 'SQL', json: 'JSON',
  yaml: 'YAML', yml: 'YAML', html: 'HTML', css: 'CSS', md: 'Markdown',
}

const CodeBlock = memo(function CodeBlock({
  language,
  code,
}: {
  language: string
  code: string
}) {
  const [copied, setCopied] = useState(false)
  const lineCount = code.split('\n').length

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {}
  }

  const displayLabel = languageLabel[language.toLowerCase()] || language || 'Text'

  return (
    <div className="my-4 rounded-lg border border-border overflow-hidden shadow-sm">
      <div className="flex items-center justify-between px-3 py-1.5 bg-muted/50 border-b border-border">
        <span className="text-xs font-mono text-muted-foreground tracking-wide">
          {displayLabel}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3" />
              Copied
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" />
              Copy
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        language={language || 'text'}
        style={oneDark}
        showLineNumbers={lineCount > 8}
        lineNumberStyle={{ opacity: 0.35, minWidth: '2.25em', userSelect: 'none' }}
        customStyle={{
          margin: 0,
          padding: '1rem',
          background: 'rgb(13 17 23 / 0.4)',
          fontSize: '0.85rem',
        }}
        codeTagProps={{ className: jetbrainsMono.className }}
        wrapLongLines
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
})

/* ── Markdown renderer with Thought Parsing & Math Support ───── */
function MarkdownContent({ content }: { content: string }) {
  // Extract <think> blocks (handles streaming state where </think> might not exist yet)
  const thinkMatch = content.match(/<think>([\s\S]*?)(?:<\/think>|$)/i)
  const hasThink = !!thinkMatch
  const thinkContent = hasThink ? thinkMatch[1].trim() : ''
  const mainContent = content.replace(/<think>[\s\S]*?(?:<\/think>|$)/i, '').trim()

  const MarkdownRenderer = ({ text }: { text: string }) => (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
      rehypePlugins={[rehypeKatex]}
      components={{
        code({ className, children, ...props }) {
          const raw = String(children).replace(/\n$/, '')
          const match = /language-(\w+)/.exec(className || '')

          if (match || raw.includes('\n')) {
            return <CodeBlock language={match?.[1] ?? ''} code={raw} />
          }
          return (
            <code className="px-1.5 py-0.5 rounded bg-muted/70 text-foreground/90 font-mono text-[0.85em]" {...props}>
              {children}
            </code>
          )
        },
        p({ children }) {
          return <p className="mb-4 last:mb-0 leading-relaxed break-words">{children}</p>
        },
        ul({ children }) { return <ul className="mb-4 list-disc pl-6 space-y-1 marker:text-muted-foreground">{children}</ul> },
        ol({ children }) { return <ol className="mb-4 list-decimal pl-6 space-y-1 marker:text-muted-foreground">{children}</ol> },
      }}
    >
      {text}
    </ReactMarkdown>
  )

  return (
    <div className="text-foreground text-sm">
      {hasThink && (
        <details className="mb-4 group border border-border/60 rounded-lg bg-muted/20">
          <summary className="flex items-center gap-2 px-4 py-2.5 cursor-pointer font-mono text-xs text-muted-foreground hover:text-foreground select-none">
            <BrainCircuit className="w-4 h-4" />
            <span>Reasoning Process</span>
            <ChevronDown className="w-3 h-3 ml-auto group-open:rotate-180 transition-transform" />
          </summary>
          <div className="px-4 py-3 border-t border-border/60 text-muted-foreground opacity-90 text-[0.9em] border-l-2 border-l-primary/40 bg-background/50">
             <MarkdownRenderer text={thinkContent} />
          </div>
        </details>
      )}
      {mainContent && <MarkdownRenderer text={mainContent} />}
    </div>
  )
}

function PlainText({ text }: { text: string }) {
  return <p className="text-foreground text-sm leading-relaxed whitespace-pre-wrap break-words">{text}</p>
}

function ImageBlock({ svg, attachmentId }: { svg: string; attachmentId: string }) {
  const openFull = () => window.open(attachmentsAPI.fullImageUrl(attachmentId), '_blank')

  return (
    <button onClick={openFull} className="group relative block rounded-lg overflow-hidden border border-border max-w-xs hover:border-primary/50 transition-colors mt-2" type="button">
      <img src={svg} alt="Attached screenshot" className="block w-full h-auto" />
      <span className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
        <Maximize2 className="h-5 w-5 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
      </span>
    </button>
  )
}

/** A file attached to the chat (PDF / text). Only a pointer is stored; the text lives on the backend. */
function FileBlock({ filename, pages }: { filename: string; pages: number }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-lg border border-border bg-background/60 px-3 py-2 mt-1">
      <FileText className="h-4 w-4 text-primary shrink-0" />
      <div className="flex flex-col leading-tight min-w-0">
        <span className="text-sm text-foreground truncate max-w-[220px]">{filename}</span>
        <span className="text-xs text-muted-foreground">
          {pages} {pages === 1 ? 'page' : 'pages'}
        </span>
      </div>
    </div>
  )
}

function BlockList({ blocks, markdown }: { blocks: ContentBlock[]; markdown: boolean }) {
  return (
    <div className="space-y-3">
      {blocks.map((block, i) => {
        if (block.label === 'image') {
          return <ImageBlock key={i} svg={block.svg} attachmentId={block.attachment_id} />
        }
        if (block.label === 'file') {
          return <FileBlock key={i} filename={block.filename} pages={block.pages} />
        }
        return markdown ? (
          <MarkdownContent key={i} content={block.content} />
        ) : (
          <PlainText key={i} text={block.content} />
        )
      })}
    </div>
  )
}

export function ChatMessage({ role, content, modelCategory, modelName, stats }: ChatMessageProps) {
  const [badgeExpanded, setBadgeExpanded] = useState(false)
  const isUser = role === 'user'
  const displayModelName = modelName || (modelCategory ? categoryLabel[modelCategory] : 'Unknown Model')
  // Images and files always count as content; text blocks only when non-empty
  const hasContent = content.some((b) => (b.label === 'text' ? b.content.trim().length > 0 : true))

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>

      {/* Avatar Indicator */}
      <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 border ${
        isUser
          ? 'bg-primary/10 border-primary/20 text-primary'
          : 'bg-muted/50 border-border text-foreground'
      }`}>
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? 'flex justify-end' : ''}`}>
        {isUser ? (
          <div className="max-w-xl bg-primary/10 border border-primary/20 rounded-2xl rounded-tr-sm px-5 py-3 shadow-sm">
            <BlockList blocks={content} markdown={false} />
          </div>
        ) : (
          <div className="w-full max-w-4xl pt-1">
            {hasContent ? (
              <BlockList blocks={content} markdown={true} />
            ) : (
              <span className="inline-flex gap-1 items-center h-6">
                <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce" />
                <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <span className="w-2 h-2 bg-primary/60 rounded-full animate-bounce [animation-delay:-0.3s]" />
              </span>
            )}

            {(modelName || modelCategory) && (
              <div className="mt-3">
                <button
                  onClick={() => setBadgeExpanded((v) => !v)}
                  aria-expanded={badgeExpanded}
                  className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] uppercase font-mono text-muted-foreground/70 border border-transparent hover:border-border hover:bg-muted/30 transition-all group"
                >
                  <Bot className="w-3 h-3" />
                  <span>
                    {displayModelName}
                    {stats ? ` · ${stats.inference_s.toFixed(1)}S` : ''}
                  </span>
                  {stats && (
                    <ChevronDown className={`w-3 h-3 transition-transform ${badgeExpanded ? 'rotate-180' : ''}`} />
                  )}
                </button>

                {badgeExpanded && stats && (
                  <div className="mt-2 ml-2 text-xs font-mono text-muted-foreground space-y-1 border-l-2 border-border/50 pl-3">
                    <div className="flex gap-4">
                      <span className="text-foreground/40 w-16">Classify:</span>
                      <span className="text-foreground/80">{(stats.classify_ms / 1000).toFixed(2)}s</span>
                    </div>
                    <div className="flex gap-4">
                      <span className="text-foreground/40 w-16">Inference:</span>
                      <span className="text-foreground/80">{stats.inference_s.toFixed(2)}s</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}