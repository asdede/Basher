"use client"

import { EvaluationClient } from "./evaluation-client"
import { useState, useEffect } from "react"
import type { EvaluationResult } from "./columns"

async function fetchRunKeys(): Promise<string[]> {
  const res = await fetch("/evaluation/data")
  if (!res.ok) throw new Error("Failed to list evaluation runs")
  const data = await res.json()
  return data.runKeys ?? []
}

async function fetchRunData(runKey: string): Promise<{ results: EvaluationResult[]; runKey: string }> {
  const res = await fetch(`/evaluation/data?runKey=${encodeURIComponent(runKey)}`)
  if (!res.ok) throw new Error("Failed to load evaluation run")
  return res.json()
}

export default function EvaluationPage() {
  const [runKeys, setRunKeys] = useState<string[]>([])
  const [selectedRunKey, setSelectedRunKey] = useState<string>("")
  const [results, setResults] = useState<EvaluationResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchRunKeys()
      .then((keys) => {
        setRunKeys(keys)
        if (keys.length > 0 && !selectedRunKey) setSelectedRunKey(keys[0])
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load runs"))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!selectedRunKey) return
    setLoading(true)
    setError(null)
    fetchRunData(selectedRunKey)
      .then(({ results: r, runKey }) => {
        setResults(r)
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load run"))
      .finally(() => setLoading(false))
  }, [selectedRunKey])

  const displayLabel = (key: string) => {
    const m = key.match(/^results_(.+)$/)
    const ts = m ? m[1] : key
    const dateObject = ts.split("_")
    const [year,month,day] = [dateObject[0].substring(0,4), dateObject[0].substring(4,6), dateObject[0].substring(6,8)]
    const [hour,minute,second] = [dateObject[1].substring(0,2), dateObject[1].substring(2,4), dateObject[1].substring(4,6)]
    const date = `${year}-${month}-${day}`
    const time = `${hour}:${minute}:${second}`
    return ` Run: ${date} ${time}`
  }

  return (
    <div className="container mx-auto py-10">
      <h1 className="text-2xl font-bold mb-4">Evaluation Results</h1>
      {error && (
        <p className="text-sm text-destructive mb-4">{error}</p>
      )}
      <select
        value={selectedRunKey}
        onChange={(e) => setSelectedRunKey(e.target.value)}
        disabled={loading || runKeys.length === 0}
        className="rounded-md border border-input bg-background px-3 py-2 text-sm"
      >
        {runKeys.length === 0 && <option value="">No runs found</option>}
        {runKeys.map((key) => (
          <option key={key} value={key}>
            {displayLabel(key)}
          </option>
        ))}
      </select>
      <p className="text-sm text-muted-foreground mb-4 mt-4">
        Click <strong>View</strong> to open response and context. Set <strong>Relevance</strong>, <strong>Context</strong>, and <strong>Correctness</strong> (1–10); scores are saved to the result file.
      </p>
      <EvaluationClient initialData={results} runKey={selectedRunKey} />
    </div>
  )
}
