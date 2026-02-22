"use client"

import { ColumnDef } from "@tanstack/react-table"

export type EvaluationResult = {
  id: string
  section_id: string
  section_label: string
  query: string
  response: string
  context: string
  retrieval: {
    chunks_used: number
    collections: Array<string | { collection: string; hits?: number; used?: number; scores?: number[] }>
  }
  review_type: "auto" | "human"
  auto_pass: boolean | null
  auto_checks?: {
    expected_contains: boolean
    must_not_contain: boolean
  } | null
  human_score: number | null
  score_relevance?: number | null
  score_context_usage?: number | null
  score_correctness?: number | null
  human_rubric: string | null
}

const TRUNCATE_LEN = 60

function truncate(s: string, len = TRUNCATE_LEN) {
  if (!s) return ""
  return s.length <= len ? s : s.slice(0, len) + "…"
}

export type CategoricalScores = {
  human_score?: number | null
  score_relevance?: number | null
  score_context_usage?: number | null
  score_correctness?: number | null
}

export type ColumnActions = {
  onView: (result: EvaluationResult) => void
  scores: Record<string, CategoricalScores>
  onScoreChange: (id: string, field: keyof CategoricalScores, value: number | null) => void
}

export function createColumns(actions: ColumnActions): ColumnDef<EvaluationResult>[] {
  const { onView, scores, onScoreChange } = actions
  return [
    {
      id: "view",
      header: "",
      cell: ({ row }) => (
        <button
          type="button"
          onClick={() => onView(row.original)}
          className="rounded bg-primary px-2 py-1 text-xs font-medium text-primary-foreground hover:opacity-90"
        >
          View
        </button>
      ),
    },
    {
      accessorKey: "query",
      header: "Query",
      cell: ({ row }) => (
        <div className="max-w-[220px] text-sm text-muted-foreground" title={row.original.query}>
          {truncate(row.original.query, 50)}
        </div>
      ),
    },
    {
      accessorKey: "section_label",
      header: "Section",
      cell: ({ row }) => <span className="text-sm">{row.original.section_label}</span>,
    },
    {
      accessorKey: "auto_pass",
      header: "Auto",
      cell: ({ row }) => {
        const pass = row.original.auto_pass
        if (pass == null) return <span className="text-muted-foreground">—</span>
        return (
          <span className={pass ? "text-green-600" : "text-destructive"}>
            {pass ? "✓" : "✗"}
          </span>
        )
      },
    },
    ...(["score_relevance", "score_context_usage", "score_correctness"] as const).map((field) => ({
      accessorKey: field,
      header: field === "score_relevance" ? "Relevance" : field === "score_context_usage" ? "Context" : "Correctness",
      cell: ({ row }: { row: { original: EvaluationResult } }) => {
        const id = row.original.id
        const rowScores = scores[id]
        const value = rowScores?.[field] ?? row.original[field] ?? ""
        return (
          <select
            value={value === null || value === undefined ? "" : value}
            onChange={(e) =>
              onScoreChange(id, field, e.target.value === "" ? null : Number(e.target.value))
            }
            className="rounded border bg-background px-2 py-0.5 text-sm w-12"
          >
            <option value="">—</option>
            {Array.from({ length: 10 }, (_, i) => i + 1).map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        )
      },
    })),
    {
      accessorKey: "human_rubric",
      header: "Rubric",
      cell: ({ row }) => (
        <div className="max-w-[160px] text-xs text-muted-foreground" title={row.original.human_rubric ?? ""}>
          {truncate(row.original.human_rubric ?? "", 40)}
        </div>
      ),
    },
    {
      accessorKey: "retrieval",
      header: "Retrieval",
      cell: ({ row }) => {
        const r = row.original.retrieval
        if (!r) return null
        const cols = r.collections ?? []
        const names = cols.map((c) => (typeof c === "string" ? c : c?.collection ?? "")).filter(Boolean)
        return (
          <div className="text-xs text-muted-foreground">
            {r.chunks_used ?? 0} chunks · {names.slice(0, 3).join(", ")}{names.length > 3 ? "…" : ""}
          </div>
        )
      },
    },
  ]
}
