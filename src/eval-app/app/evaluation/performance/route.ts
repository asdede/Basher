import { NextResponse } from "next/server"
import fs from "fs/promises"
import path from "path"

const RESULTS_DIR = path.join(process.cwd(), "..", "..", "eval", "results")

type ResultRow = {
  score_relevance?: number | null
  score_context_usage?: number | null
  score_correctness?: number | null
}

function avg(values: number[]): number | null {
  if (values.length === 0) return null
  return values.reduce((a, b) => a + b, 0) / values.length
}

function parseRunKey(runKey: string): Date | null {
  const m = runKey.match(/^results_(\d{8})_(\d{6})$/)
  if (!m) return null
  const [, ymd, hms] = m
  const y = parseInt(ymd.slice(0, 4), 10)
  const mo = parseInt(ymd.slice(4, 6), 10) - 1
  const d = parseInt(ymd.slice(6, 8), 10)
  const h = parseInt(hms.slice(0, 2), 10)
  const min = parseInt(hms.slice(2, 4), 10)
  const s = parseInt(hms.slice(4, 6), 10)
  return new Date(y, mo, d, h, min, s)
}

export async function GET() {
  try {
    const files = await fs.readdir(RESULTS_DIR)
    const runKeys = files
      .filter((f) => f.startsWith("results_") && f.endsWith(".json"))
      .map((f) => path.basename(f, ".json"))
      .sort()
      .reverse()

    const series: Array<{
      runKey: string
      date: string
      dateTime: number
      relevance: number | null
      contextUsage: number | null
      correctness: number | null
      scoredCount: number
    }> = []

    for (const runKey of runKeys) {
      const filePath = path.join(RESULTS_DIR, `${runKey}.json`)
      const raw = await fs.readFile(filePath, "utf-8")
      const data = JSON.parse(raw) as { results?: ResultRow[] }
      const results = data.results ?? []

      const rel: number[] = []
      const ctx: number[] = []
      const cor: number[] = []
      for (const r of results as ResultRow[]) {
        if (typeof r.score_relevance === "number") rel.push(r.score_relevance)
        if (typeof r.score_context_usage === "number") ctx.push(r.score_context_usage)
        if (typeof r.score_correctness === "number") cor.push(r.score_correctness)
      }

      const date = parseRunKey(runKey)
      const dateStr = date
        ? date.toISOString().slice(0, 19).replace("T", " ")
        : runKey

      series.push({
        runKey,
        date: dateStr,
        dateTime: date ? date.getTime() : 0,
        relevance: avg(rel),
        contextUsage: avg(ctx),
        correctness: avg(cor),
        scoredCount: Math.max(rel.length, ctx.length, cor.length),
      })
    }

    series.sort((a, b) => a.dateTime - b.dateTime)

    const latestRunKey = runKeys[0] ?? null
    const latest =
      latestRunKey && series.length > 0
        ? series.find((s) => s.runKey === latestRunKey) ?? series[series.length - 1]
        : null

    return NextResponse.json({
      series,
      latestRunKey,
      latest: latest
        ? {
            runKey: latest.runKey,
            date: latest.date,
            relevance: latest.relevance,
            contextUsage: latest.contextUsage,
            correctness: latest.correctness,
            scoredCount: latest.scoredCount,
          }
        : null,
    })
  } catch (err) {
    console.error("Evaluation performance route error:", err)
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to load performance data" },
      { status: 500 }
    )
  }
}
