"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { DataTable } from "./data-table"
import { ResultDetailDialog } from "./result-detail-dialog"
import {
  createColumns,
  type CategoricalScores,
  type EvaluationResult,
} from "./columns"

function scoresFromResults(results: EvaluationResult[]): Record<string, CategoricalScores> {
  const out: Record<string, CategoricalScores> = {}
  for (const r of results) {
    out[r.id] = {
      human_score: r.human_score ?? null,
      score_relevance: r.score_relevance ?? null,
      score_context_usage: r.score_context_usage ?? null,
      score_correctness: r.score_correctness ?? null,
    }
  }
  return out
}

const SAVE_DEBOUNCE_MS = 500

type Props = {
  initialData: EvaluationResult[]
  runKey?: string
}

export function EvaluationClient({ initialData, runKey = "default" }: Props) {
  const [scores, setScores] = useState<Record<string, CategoricalScores>>(() =>
    scoresFromResults(initialData)
  )
  useEffect(() => {
    Object.values(saveTimeoutsRef.current).forEach(clearTimeout)
    saveTimeoutsRef.current = {}
    setScores(scoresFromResults(initialData))
  }, [runKey, initialData])
  const saveTimeoutsRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  const scoresRef = useRef(scores)
  scoresRef.current = scores

  const [selectedResult, setSelectedResult] = useState<EvaluationResult | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)

  const persistScores = useCallback(
    (resultId: string) => {
      const payload = scoresRef.current[resultId]
      if (!payload || !runKey) return
      fetch("/evaluation/data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          runKey,
          resultId,
          scores: {
            human_score: payload.human_score ?? null,
            score_relevance: payload.score_relevance ?? null,
            score_context_usage: payload.score_context_usage ?? null,
            score_correctness: payload.score_correctness ?? null,
          },
        }),
      }).catch((err) => console.error("Failed to save scores:", err))
    },
    [runKey]
  )

  const onScoreChange = useCallback(
    (id: string, field: keyof CategoricalScores, value: number | null) => {
      setScores((prev) => {
        const next = {
          ...prev,
          [id]: { ...prev[id], [field]: value },
        }
        const t = saveTimeoutsRef.current[id]
        if (t) clearTimeout(t)
        saveTimeoutsRef.current[id] = setTimeout(() => {
          delete saveTimeoutsRef.current[id]
          persistScores(id)
        }, SAVE_DEBOUNCE_MS)
        return next
      })
    },
    [persistScores]
  )

  const onView = useCallback((result: EvaluationResult) => {
    setSelectedResult(result)
    setDialogOpen(true)
  }, [])

  const columns = useMemo(
    () => createColumns({ onView, scores, onScoreChange }),
    [onView, scores, onScoreChange]
  )

  const selectedScores = selectedResult
    ? scores[selectedResult.id] ?? {
        human_score: selectedResult.human_score ?? null,
        score_relevance: selectedResult.score_relevance ?? null,
        score_context_usage: selectedResult.score_context_usage ?? null,
        score_correctness: selectedResult.score_correctness ?? null,
      }
    : null

  return (
    <>
      <DataTable columns={columns} data={initialData} />
      <ResultDetailDialog
        result={selectedResult}
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        scores={selectedScores}
        onScoreChange={(id, field, value) => {
          onScoreChange(id, field, value)
          if (selectedResult?.id === id)
            setSelectedResult((r) =>
              r ? { ...r, [field]: value } : null
            )
        }}
      />
    </>
  )
}
