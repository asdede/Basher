"use client"

import { useState } from "react"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

type Progress = {
  current: number
  total: number
  percent: number
  message: string
} | null

export default function RunEvaluationPage() {
  const [progress, setProgress] = useState<Progress>(null)
  const [complete, setComplete] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [useBaseModel, setUseBaseModel] = useState(false)

  async function runEvaluation() {
    setRunning(true)
    setProgress(null)
    setComplete(false)
    setError(null)
    try {
      const url = `${API_URL}/run-evaluation?use_base_model=${useBaseModel}&stream=true`
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) throw new Error("No response body")
      let buffer = ""
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n\n")
        buffer = lines.pop() ?? ""
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6))
            if (data.status === "complete") {
              setComplete(true)
            } else {
              setProgress({
                current: data.current,
                total: data.total,
                percent: data.percent ?? 0,
                message: data.message ?? "",
              })
            }
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to run evaluation")
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="container max-w-2xl py-8 space-y-6 mx-auto">
      <h1 className="text-2xl font-semibold">Run Evaluation</h1>
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={useBaseModel}
            onChange={(e) => setUseBaseModel(e.target.checked)}
            disabled={running}
          />
          Use base model only
        </label>
        <button
          onClick={runEvaluation}
          disabled={running}
          className="rounded bg-primary px-4 py-2 text-primary-foreground disabled:opacity-50"
        >
          {running ? "Running…" : "Start evaluation"}
        </button>
      </div>
      {error && (
        <p className="text-destructive">{error}</p>
      )}

      {(running || progress !== null) && (
        <div className="rounded-lg border bg-card p-4 space-y-3">
          <div className="flex justify-between text-sm font-medium">
            <span>
              {progress ? `${progress.current} / ${progress.total} items` : "Starting…"}
            </span>
            <span aria-hidden="true">
              {progress ? `${progress.percent}%` : "0%"}
            </span>
          </div>
          <div
            className="h-4 w-full rounded-full bg-muted overflow-hidden"
            role="progressbar"
            aria-valuenow={progress?.percent ?? 0}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Evaluation progress"
          >
            <div
              className="h-full bg-primary rounded-full transition-[width] duration-300 ease-out"
              style={{ width: `${progress?.percent ?? 0}%` }}
            />
          </div>
          {progress?.message && (
            <p className="text-muted-foreground text-sm truncate" title={progress.message}>
              {progress.message}
            </p>
          )}
        </div>
      )}

      {complete && !running && (
        <p className="text-green-600 dark:text-green-400">Evaluation complete.</p>
      )}
    </div>
  )
}
