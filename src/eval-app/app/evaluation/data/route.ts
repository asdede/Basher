import { NextRequest, NextResponse } from "next/server"
import fs from "fs/promises"
import path from "path"

const RESULTS_DIR = path.join(process.cwd(), "..", "..", "eval", "results")

export async function GET(request: NextRequest) {
  const runKey = request.nextUrl.searchParams.get("runKey")

  try {
    if (runKey) {
      const filePath = path.join(RESULTS_DIR, `${runKey}.json`)
      const raw = await fs.readFile(filePath, "utf-8")
      const data = JSON.parse(raw)
      return NextResponse.json({ results: data.results ?? [], runKey })
    }

    const files = await fs.readdir(RESULTS_DIR)
    const runKeys = files
      .filter((f) => f.startsWith("results_") && f.endsWith(".json"))
      .map((f) => path.basename(f, ".json"))
    return NextResponse.json({ runKeys })
  } catch (err) {
    console.error("Evaluation data route error:", err)
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to read evaluation data" },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { runKey, resultId, scores: scoreUpdates } = body as {
      runKey?: string
      resultId?: string
      scores?: {
        human_score?: number | null
        score_relevance?: number | null
        score_context_usage?: number | null
        score_correctness?: number | null
      }
    }
    if (!runKey || !resultId || !scoreUpdates || typeof scoreUpdates !== "object") {
      return NextResponse.json(
        { error: "runKey, resultId, and scores are required" },
        { status: 400 }
      )
    }
    const filePath = path.join(RESULTS_DIR, `${runKey}.json`)
    const raw = await fs.readFile(filePath, "utf-8")
    const data = JSON.parse(raw) as { results?: Array<Record<string, unknown>> }
    const results = data.results ?? []
    const index = results.findIndex((r) => r.id === resultId)
    if (index === -1) {
      return NextResponse.json({ error: "Result not found" }, { status: 404 })
    }
    const result = results[index] as Record<string, unknown>
    if (scoreUpdates.human_score !== undefined) result.human_score = scoreUpdates.human_score
    if (scoreUpdates.score_relevance !== undefined) result.score_relevance = scoreUpdates.score_relevance
    if (scoreUpdates.score_context_usage !== undefined)
      result.score_context_usage = scoreUpdates.score_context_usage
    if (scoreUpdates.score_correctness !== undefined) result.score_correctness = scoreUpdates.score_correctness
    await fs.writeFile(filePath, JSON.stringify(data, null, 2), "utf-8")
    return NextResponse.json({ ok: true })
  } catch (err) {
    console.error("Evaluation data POST error:", err)
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to save scores" },
      { status: 500 }
    )
  }
}
