'use client'

import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogContent,
  DialogClose,
} from '@/components/ui/dialog'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type MessageRole = 'user' | 'assistant'

interface DebugInfo {
  context?: string
  retrieval?: { chunks_used?: number; collections?: unknown[] }
  base_model?: boolean
}

interface Message {
  id: string
  role: MessageRole
  content: string
  debug?: DebugInfo
}

interface Section {
  id: string
  label: string
  items?: unknown[]
}

interface DatasetResponse {
  sections?: Section[]
}

async function askApi(
  query: string,
  options: { debug: boolean; base_model: boolean }
): Promise<{ response: string; context?: string; retrieval?: DebugInfo['retrieval']; base_model?: boolean }> {
  const params = new URLSearchParams()
  if (options.debug) params.set('debug', 'true')
  if (options.base_model) params.set('base_model', 'true')
  const url = `${API_URL}/ask?${params.toString()}`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(await res.text().catch(() => res.statusText))
  return res.json()
}

async function fetchDataset(): Promise<DatasetResponse> {
  const res = await fetch(`${API_URL}/eval/dataset`)
  if (!res.ok) throw new Error('Dataset not available')
  return res.json()
}

async function addEvalItems(payload: {
  section_id: string
  section_label?: string
  items: Array<{
    id?: string
    query: string
    review_type?: string
    expected_contains?: string[] | null
    expected_collections?: string[] | null
    must_not_contain?: string[] | null
    human_rubric?: string | null
  }>
}) {
  const res = await fetch(`${API_URL}/eval/items`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await res.text().catch(() => res.statusText))
  return res.json()
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [debug, setDebug] = useState(true)
  const [baseModel, setBaseModel] = useState(false)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [saveQuery, setSaveQuery] = useState('')
  const [sections, setSections] = useState<Section[]>([])
  const [sectionId, setSectionId] = useState('')
  const [sectionLabel, setSectionLabel] = useState('')
  const [newCategory, setNewCategory] = useState(false)
  const [newSectionId, setNewSectionId] = useState('')
  const [newSectionLabel, setNewSectionLabel] = useState('')
  const [reviewType, setReviewType] = useState<'auto' | 'human'>('human')
  const [itemId, setItemId] = useState('')
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'ok' | 'err'>('idle')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  useEffect(() => {
    fetchDataset()
      .then((d) => setSections(d.sections || []))
      .catch(() => setSections([]))
  }, [])

  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    setError(null)
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)
    try {
      const out = await askApi(text, { debug, base_model: baseModel })
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: out.response || '',
        debug: debug
          ? {
              context: out.context,
              retrieval: out.retrieval,
              base_model: out.base_model,
            }
          : undefined,
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [input, loading, debug, baseModel])

  const openSaveDialog = useCallback((query: string) => {
    setSaveQuery(query)
    setSaveDialogOpen(true)
    const hasSections = sections.length > 0
    setNewCategory(!hasSections)
    setSectionId(sections[0]?.id ?? '')
    setSectionLabel(sections[0]?.label ?? '')
    setNewSectionId('')
    setNewSectionLabel('')
    setReviewType('human')
    setItemId('')
    setSaveStatus('idle')
  }, [sections])

  const submitSave = useCallback(async () => {
    const sid = newCategory ? newSectionId.trim() : sectionId
    const label = newCategory ? newSectionLabel.trim() : sectionLabel
    if (!sid) {
      setSaveStatus('err')
      return
    }
    setSaveStatus('saving')
    try {
      await addEvalItems({
        section_id: sid,
        section_label: label || sid,
        items: [
          {
            id: itemId.trim() || undefined,
            query: saveQuery,
            review_type: reviewType,
            expected_contains: null,
            expected_collections: null,
            human_rubric: null,
          },
        ],
      })
      setSaveStatus('ok')
      setTimeout(() => setSaveDialogOpen(false), 800)
    } catch {
      setSaveStatus('err')
    }
  }, [
    newCategory,
    newSectionId,
    newSectionLabel,
    sectionId,
    sectionLabel,
    saveQuery,
    reviewType,
    itemId,
  ])

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-4 p-4">
      <div className="flex flex-wrap items-center gap-4 rounded-lg border border-border bg-muted/30 p-3">
        <span className="text-sm font-medium text-muted-foreground">Settings</span>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={debug}
            onChange={(e) => setDebug(e.target.checked)}
            className="rounded border-border"
          />
          Debug (include context & retrieval)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={baseModel}
            onChange={(e) => setBaseModel(e.target.checked)}
            className="rounded border-border"
          />
          Base model only (no RAG)
        </label>
      </div>

      <div className="flex flex-1 flex-col gap-3 rounded-lg border border-border bg-card p-4 min-h-[360px]">
        <div className="flex-1 space-y-4 overflow-y-auto">
          {messages.length === 0 && (
            <p className="text-muted-foreground text-sm">Send a message to chat with the Basher LLM. Use settings above to toggle debug and base-model-only.</p>
          )}
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex flex-col gap-1 ${m.role === 'user' ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${
                  m.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-foreground'
                }`}
              >
                <div className="whitespace-pre-wrap">{m.content}</div>
                {m.role === 'user' && (
                  <button
                    type="button"
                    onClick={() => openSaveDialog(m.content)}
                    className="mt-2 text-xs underline opacity-80 hover:opacity-100"
                  >
                    Save as eval task
                  </button>
                )}
              </div>
              {m.role === 'assistant' && m.debug && (
                <details className="w-full max-w-[85%] text-xs text-muted-foreground">
                  <summary className="cursor-pointer rounded border border-border bg-muted/50 px-2 py-1">
                    Debug: context & retrieval
                  </summary>
                  <pre className="mt-1 overflow-x-auto rounded border border-border bg-muted/30 p-2 whitespace-pre-wrap wrap-break-word">
                    {m.debug.context ?? '(none)'}
                  </pre>
                  {m.debug.retrieval && (
                    <pre className="mt-1 overflow-x-auto rounded border border-border bg-muted/30 p-2">
                      {JSON.stringify(m.debug.retrieval, null, 2)}
                    </pre>
                  )}
                </details>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex items-start gap-1">
              <div className="rounded-lg bg-muted px-3 py-2 text-sm text-muted-foreground">
                Thinking…
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {error && (
          <p className="rounded bg-destructive/10 px-2 py-1 text-sm text-destructive">{error}</p>
        )}

        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about Fedora, Git, Bash, Docker…"
            className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm outline-none ring-ring focus:ring-2"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </div>

      <SaveTaskDialog
        open={saveDialogOpen}
        onOpenChange={setSaveDialogOpen}
        sections={sections}
        query={saveQuery}
        reviewType={reviewType}
        setReviewType={setReviewType}
        itemId={itemId}
        setItemId={setItemId}
        newCategory={newCategory}
        setNewCategory={setNewCategory}
        sectionId={sectionId}
        setSectionId={setSectionId}
        sectionLabel={sectionLabel}
        setSectionLabel={setSectionLabel}
        newSectionId={newSectionId}
        setNewSectionId={setNewSectionId}
        newSectionLabel={newSectionLabel}
        setNewSectionLabel={setNewSectionLabel}
        saveStatus={saveStatus}
        onSave={submitSave}
      />
    </div>
  )
}

interface SaveTaskDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sections: Section[]
  query: string
  reviewType: 'auto' | 'human'
  setReviewType: (v: 'auto' | 'human') => void
  itemId: string
  setItemId: (v: string) => void
  newCategory: boolean
  setNewCategory: (v: boolean) => void
  sectionId: string
  setSectionId: (v: string) => void
  sectionLabel: string
  setSectionLabel: (v: string) => void
  newSectionId: string
  setNewSectionId: (v: string) => void
  newSectionLabel: string
  setNewSectionLabel: (v: string) => void
  saveStatus: 'idle' | 'saving' | 'ok' | 'err'
  onSave: () => void
}

function SaveTaskDialog({
  open,
  onOpenChange,
  sections,
  query,
  reviewType,
  setReviewType,
  itemId,
  setItemId,
  newCategory,
  setNewCategory,
  sectionId,
  setSectionId,
  sectionLabel,
  setSectionLabel,
  newSectionId,
  setNewSectionId,
  newSectionLabel,
  setNewSectionLabel,
  saveStatus,
  onSave,
}: SaveTaskDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogHeader>
        <DialogTitle>Save as evaluation task</DialogTitle>
      </DialogHeader>
      <DialogContent className="flex flex-col gap-4">
        <p className="text-sm text-muted-foreground">Query: “{query}”</p>

        <div>
          <label className="mb-1 block text-sm font-medium">Category</label>
          <select
            value={newCategory ? '__new__' : (sectionId || (sections[0]?.id ?? ''))}
            onChange={(e) => {
              const v = e.target.value
              if (v === '__new__') {
                setNewCategory(true)
              } else {
                setNewCategory(false)
                setSectionId(v)
                setSectionLabel(sections.find((s) => s.id === v)?.label ?? v)
              }
            }}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            {sections.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label} ({s.id})
              </option>
            ))}
            <option value="__new__">+ New category</option>
          </select>
        </div>

        {newCategory && (
          <>
            <div>
              <label className="mb-1 block text-sm font-medium">New section ID (slug)</label>
              <input
                type="text"
                value={newSectionId}
                onChange={(e) => setNewSectionId(e.target.value)}
                placeholder="e.g. my_topic"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">New section label</label>
              <input
                type="text"
                value={newSectionLabel}
                onChange={(e) => setNewSectionLabel(e.target.value)}
                placeholder="e.g. My Topic"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
          </>
        )}

        <div>
          <label className="mb-1 block text-sm font-medium">Review type</label>
          <select
            value={reviewType}
            onChange={(e) => setReviewType(e.target.value as 'auto' | 'human')}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            <option value="human">Human</option>
            <option value="auto">Auto</option>
          </select>
        </div>

        <div>
          <label className="mb-1 block text-sm font-medium">Item ID (optional)</label>
          <input
            type="text"
            value={itemId}
            onChange={(e) => setItemId(e.target.value)}
            placeholder="Leave empty to auto-generate"
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          />
        </div>

        <div className="flex justify-end gap-2">
          <DialogClose className="rounded-md border border-border px-3 py-1.5 text-sm font-medium hover:bg-muted">
            Cancel
          </DialogClose>
          <button
            type="button"
            onClick={onSave}
            disabled={saveStatus === 'saving' || (newCategory && !newSectionId.trim())}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {saveStatus === 'saving' ? 'Saving…' : saveStatus === 'ok' ? 'Saved' : 'Save'}
          </button>
        </div>
        {saveStatus === 'err' && (
          <p className="text-sm text-destructive">Save failed. Is the API configured with EVAL_DATASET_PATH?</p>
        )}
      </DialogContent>
    </Dialog>
  )
}
