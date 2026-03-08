"use client"

import { useState, useEffect } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts"

type SeriesPoint = {
  runKey: string
  date: string
  dateTime: number
  relevance: number | null
  contextUsage: number | null
  correctness: number | null
  scoredCount: number
}

type PerformanceData = {
  series: SeriesPoint[]
  latestRunKey: string | null
  latest: {
    runKey: string
    date: string
    relevance: number | null
    contextUsage: number | null
    correctness: number | null
    scoredCount: number
  } | null
}

const RELEVANCE_COLOR = "#22c55e"
const CONTEXT_COLOR = "#3b82f6"
const CORRECTNESS_COLOR = "#a855f7"

function formatScore(v: number | null): string {
  if (v == null) return "—"
  return (v / 10).toFixed(0) + "%"
}

export default function PerformanceMonitorPage() {
  const [data, setData] = useState<PerformanceData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch("/evaluation/performance")
      .then((res) => {
        if (!res.ok) throw new Error(res.statusText)
        return res.json()
      })
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="container max-w-6xl py-10 mx-auto">
        <h1 className="text-2xl font-bold mb-6">Performance Monitor</h1>
        <p className="text-muted-foreground">Loading…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container max-w-6xl py-10 mx-auto">
        <h1 className="text-2xl font-bold mb-6">Performance Monitor</h1>
        <p className="text-destructive">{error}</p>
      </div>
    )
  }

  const series = data?.series ?? []
  const latest = data?.latest ?? null
  const hasTimeSeries = series.length > 0

  const chartData = series.map((s: SeriesPoint) => ({
    ...s,
    name: s.date.slice(0, 16),
    relevance: s.relevance ?? 0,
    contextUsage: s.contextUsage ?? 0,
    correctness: s.correctness ?? 0,
  }))

  const barData = latest
    ? [
        {
          metric: "Relevance",
          value: latest.relevance ?? 0,
          fill: RELEVANCE_COLOR,
          label: formatScore(latest.relevance),
        },
        {
          metric: "Context usage",
          value: latest.contextUsage ?? 0,
          fill: CONTEXT_COLOR,
          label: formatScore(latest.contextUsage),
        },
        {
          metric: "Correctness",
          value: latest.correctness ?? 0,
          fill: CORRECTNESS_COLOR,
          label: formatScore(latest.correctness),
        },
      ]
    : []

  return (
    <div className="container max-w-6xl py-10 space-y-10 mx-auto">
      <h1 className="text-2xl font-bold">Performance Monitor</h1>
      <p className="text-muted-foreground text-sm">
        Scores are 1–10; divide by 10 for percentage. Only runs with at least one scored item are included.
      </p>

      {hasTimeSeries ? (
        <div className="rounded-lg border bg-card p-4">
          <h2 className="text-lg font-semibold mb-4">Performance over time</h2>
          <div className="h-[360px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                />
                <YAxis
                  domain={[0, 10]}
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number) => [value.toFixed(1), ""]}
                  labelFormatter={(label: string) => `Run: ${label}`}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="relevance"
                  name="Relevance"
                  stroke={RELEVANCE_COLOR}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="contextUsage"
                  name="Context usage"
                  stroke={CONTEXT_COLOR}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="correctness"
                  name="Correctness"
                  stroke={CORRECTNESS_COLOR}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="rounded-lg border bg-muted/30 p-6 text-center text-muted-foreground">
          No evaluation runs with scores yet. Run an evaluation and set scores in Evaluation Results.
        </div>
      )}

      {latest && (
        <div className="rounded-lg border bg-card p-4 mx-auto">
          <h2 className="text-lg font-semibold mb-2">Latest run</h2>
          <p className="text-sm text-muted-foreground mb-4">
            {latest.date} · {latest.scoredCount} item(s) scored
          </p>
          <div className="h-[280px] w-full max-w-md mx-auto">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={barData}
                layout="vertical"
                margin={{ top: 8, right: 24, left: 100, bottom: 8 }}
              >
                <XAxis
                  type="number"
                  domain={[0, 10]}
                  tick={{ fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                />
                <YAxis
                  type="category"
                  dataKey="metric"
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={{ stroke: "hsl(var(--border))" }}
                  width={90}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                  formatter={(value: number, _name: string, props: { payload?: { label?: string } }) =>
                    [`${value.toFixed(1)} (${props.payload?.label ?? ""})`, ""]
                  }
                />
                <Bar dataKey="value" name="Score (1–10)" radius={[0, 4, 4, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
