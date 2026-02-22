"use client"

import { Dialog, DialogHeader, DialogTitle, DialogContent, DialogClose } from "@/components/ui/dialog"
import type { CategoricalScores, EvaluationResult } from "./columns"

const SCORE_OPTIONS_1_10 = Array.from({ length: 10 }, (_, i) => i + 1)

type Props = {
  result: EvaluationResult | null
  open: boolean
  onOpenChange: (open: boolean) => void
  scores: CategoricalScores | null
  onScoreChange: (id: string, field: keyof CategoricalScores, value: number | null) => void
}

function PreBlock({ text, label }: { text: string; label: string }) {
  return (
    <div className="m-4">
      <div className="text-xs font-medium text-muted-foreground mb-1">{label}</div>
      <pre className="whitespace-pre-wrap break-words rounded border bg-muted/50 p-3 text-sm overflow-x-auto max-h-64 overflow-y-auto">
        {text || "(empty)"}
      </pre>
    </div>
  )
}

export function ResultDetailDialog({ result, open, onOpenChange, scores, onScoreChange }: Props) {
  if (!result) return null

  const collections = result.retrieval?.collections
  const collectionList = Array.isArray(collections)
    ? collections.map((c: unknown) => (typeof c === "object" && c && "collection" in c ? (c as { collection: string }).collection : String(c)))
    : []

  const scoreSelect = (field: keyof CategoricalScores, label: string) => {
    const value = scores?.[field] ?? ""
    return (
      <label key={field} className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground whitespace-nowrap">{label}:</span>
        <select
          value={value === null || value === undefined ? "" : value}
          onChange={(e) =>
            onScoreChange(result.id, field, e.target.value === "" ? null : Number(e.target.value))
          }
          className="rounded border bg-background px-2 py-1 text-sm w-14"
        >
          <option value="">—</option>
          {SCORE_OPTIONS_1_10.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </label>
    )
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange} className="max-w-[100vw]">
      <DialogHeader className="flex flex-row items-center justify-between border-b w-full">
        <DialogTitle className="truncate pr-4">{result.id}</DialogTitle>
        <div className="flex items-center gap-4 flex-wrap">
          {scoreSelect("score_relevance", "Relevance (1–10)")}
          {scoreSelect("score_context_usage", "Context (1–10)")}
          {scoreSelect("score_correctness", "Correctness (1–10)")}
          <DialogClose />
        </div>
      </DialogHeader>
      <DialogContent>
        <div className="space-y-4">
          <PreBlock text={result.query} label="Query" />
          <PreBlock text={result.response} label="Response" />
          <PreBlock text={result.context} label="Context (RAG)" />
          {result.human_rubric && (
            <div className="text-sm">
              <span className="font-medium text-muted-foreground">Rubric: </span>
              {result.human_rubric}
            </div>
          )}
          {collectionList.length > 0 && (
            <div className="text-sm">
              <span className="font-medium text-muted-foreground">Collections: </span>
              {collectionList.join(", ")}
              {typeof result.retrieval?.chunks_used === "number" && (
                <span className="ml-2 text-muted-foreground">({result.retrieval.chunks_used} chunks)</span>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
