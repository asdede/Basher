#!/usr/bin/env python3
"""
LLM-as-judge for Basher eval result JSON files.

Re-scores each item with a stronger (or equal) Ollama model. Writes a new
JSON with judge fields and a short markdown summary.

Usage:
  python eval/llm_judge.py eval/rag_qwen35_4b.json -o eval/rag_qwen35_4b.judged.json
  BASHER_JUDGE_MODEL=qwen3.5:4b python eval/llm_judge.py eval/ablation_qwen25_1.5b/rag_qwen25_1.5b.json

Default judge: qwen3.5:4b with think=false (fits 8GB; stronger than 1.5b answers).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError:
    print("needs requests", file=sys.stderr)
    sys.exit(1)

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
if OLLAMA.endswith("/api/generate"):
    OLLAMA = OLLAMA[: -len("/api/generate")]
JUDGE_MODEL = os.environ.get("BASHER_JUDGE_MODEL", "qwen3.5:4b")

JUDGE_PROMPT = """You are a strict evaluator for a Fedora/Linux assistant.
Score the ASSISTANT answer for the USER question. Use any DOCUMENTATION only as optional grounding evidence (it may be empty for baseline runs).

Return ONLY a JSON object with these keys:
{{
  "correctness": <int 1-5>,
  "grounding": <int 1-5>,
  "helpfulness": <int 1-5>,
  "pass": <true|false>,
  "reason": "<one short sentence>"
}}

Rubric:
- correctness: factual accuracy for Fedora/bash/git/docker/linux; punish invented flags/commands.
- grounding: if documentation is provided, prefer answers that align with it; if docs empty, score based on general correctness (3=neutral).
- helpfulness: concise, actionable, right language.
- pass: true only if correctness >= 4 AND the answer does not recommend clearly wrong package managers/commands for the context (e.g. apt as the way to manage Fedora packages).
- Mentioning that apt is WRONG for Fedora is OK (do not fail for that).

USER QUESTION:
{query}

DOCUMENTATION (may be empty):
{context}

ASSISTANT ANSWER:
{answer}

HUMAN RUBRIC (if any): {rubric}
"""


def call_judge(query: str, answer: str, context: str, rubric: str) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        query=query or "",
        context=(context or "")[:3500] or "(none)",
        answer=(answer or "")[:4000] or "(empty)",
        rubric=(rubric or "none"),
    )
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 256,
                "num_ctx": 8192,
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    raw = (r.json().get("response") or "").strip()
    return parse_judge(raw)


def parse_judge(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {
                "correctness": None,
                "grounding": None,
                "helpfulness": None,
                "pass": None,
                "reason": f"judge parse failed: {raw[:120]}",
                "raw": raw,
            }
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {
                "correctness": None,
                "grounding": None,
                "helpfulness": None,
                "pass": None,
                "reason": f"judge parse failed: {raw[:120]}",
                "raw": raw,
            }

    def clamp(v: Any) -> Optional[int]:
        try:
            i = int(v)
            return max(1, min(5, i))
        except (TypeError, ValueError):
            return None

    passed = data.get("pass")
    if isinstance(passed, str):
        passed = passed.strip().lower() in ("true", "yes", "1", "pass")
    elif passed is not None:
        passed = bool(passed)

    return {
        "correctness": clamp(data.get("correctness")),
        "grounding": clamp(data.get("grounding")),
        "helpfulness": clamp(data.get("helpfulness")),
        "pass": passed,
        "reason": str(data.get("reason") or "")[:300],
    }


def judge_file(path: Path, out: Path, limit: Optional[int] = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = list(data.get("results") or [])
    if limit is not None:
        results = results[:limit]

    judged = []
    for i, item in enumerate(results, 1):
        q = item.get("query") or ""
        ans = item.get("response") or ""
        ctx = item.get("context") or ""
        if item.get("base_model") or (data.get("run_meta") or {}).get("base_model"):
            # baseline runs store context as a marker string sometimes
            if ctx.startswith("(base model"):
                ctx = ""
        rubric = item.get("human_rubric") or ""
        print(f"[{i}/{len(results)}] judge {item.get('id')} ...", flush=True)
        try:
            j = call_judge(q, ans, ctx, rubric)
        except requests.RequestException as e:
            j = {
                "correctness": None,
                "grounding": None,
                "helpfulness": None,
                "pass": None,
                "reason": f"judge error: {e}",
            }
        row = dict(item)
        row["llm_judge"] = j
        row["human_score"] = j.get("correctness")
        row["score_correctness"] = j.get("correctness")
        row["score_context_usage"] = j.get("grounding")
        row["score_relevance"] = j.get("helpfulness")
        # Corrected grade: prefer LLM judge pass over brittle auto_pass
        if j.get("pass") is not None:
            row["judge_pass"] = j["pass"]
        judged.append(row)
        time.sleep(0.05)

    payload = {
        "run_meta": {
            **(data.get("run_meta") or {}),
            "judge_model": JUDGE_MODEL,
            "judge_source": str(path.resolve()),
            "judged_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": judged,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summarize_pair(rag: dict, base: dict | None) -> str:
    def stats(results: list) -> dict:
        scored = [r for r in results if (r.get("llm_judge") or {}).get("pass") is not None]
        n = len(scored)
        p = sum(1 for r in scored if r["llm_judge"]["pass"])
        corr = [r["llm_judge"]["correctness"] for r in scored if r["llm_judge"].get("correctness")]
        by = {}
        for r in scored:
            sec = r.get("section_id") or "?"
            by.setdefault(sec, {"n": 0, "p": 0})
            by[sec]["n"] += 1
            if r["llm_judge"]["pass"]:
                by[sec]["p"] += 1
        return {
            "n": n,
            "p": p,
            "rate": (p / n) if n else None,
            "avg_correctness": (sum(corr) / len(corr)) if corr else None,
            "by": by,
            "fail_ids": [r["id"] for r in scored if not r["llm_judge"]["pass"]],
        }

    rs = stats(rag.get("results") or [])
    lines = [
        "# LLM-as-judge summary",
        "",
        f"- Judge model: `{JUDGE_MODEL}`",
        f"- Answer run: `{(rag.get('run_meta') or {}).get('mode')}` / model `{(rag.get('run_meta') or {}).get('model')}`",
        "",
        "## Pass rates (judge_pass)",
        "",
        "| Mode | Judge pass | Avg correctness (1–5) |",
        "|------|------------|------------------------|",
        f"| RAG | {rs['p']}/{rs['n']} ({100*rs['rate']:.0f}%) | {rs['avg_correctness']:.2f} |"
        if rs["rate"] is not None and rs["avg_correctness"] is not None
        else f"| RAG | {rs['p']}/{rs['n']} | n/a |",
    ]
    bs = None
    if base:
        bs = stats(base.get("results") or [])
        if bs["rate"] is not None and bs["avg_correctness"] is not None:
            lines.append(
                f"| Baseline | {bs['p']}/{bs['n']} ({100*bs['rate']:.0f}%) | {bs['avg_correctness']:.2f} |"
            )

    lines += ["", "## By section (RAG)", "", "| Section | Judge pass |", "|---------|------------|"]
    for sec, v in sorted(rs["by"].items()):
        lines.append(f"| {sec} | {v['p']}/{v['n']} |")

    if bs:
        lines += ["", "## By section (Baseline)", "", "| Section | Judge pass |", "|---------|------------|"]
        for sec, v in sorted(bs["by"].items()):
            lines.append(f"| {sec} | {v['p']}/{v['n']} |")

        rag_ids = {r["id"]: r for r in rag["results"] if (r.get("llm_judge") or {}).get("pass") is not None}
        base_ids = {r["id"]: r for r in base["results"] if (r.get("llm_judge") or {}).get("pass") is not None}
        helped = sorted(
            i for i in rag_ids if i in base_ids and rag_ids[i]["llm_judge"]["pass"] and not base_ids[i]["llm_judge"]["pass"]
        )
        hurt = sorted(
            i for i in rag_ids if i in base_ids and (not rag_ids[i]["llm_judge"]["pass"]) and base_ids[i]["llm_judge"]["pass"]
        )
        lines += ["", f"- RAG helped (judge): **{len(helped)}** — " + (", ".join(f'`{x}`' for x in helped) or "—")]
        lines += [f"- RAG hurt (judge): **{len(hurt)}** — " + (", ".join(f'`{x}`' for x in hurt) or "—")]

    lines += ["", "## RAG judge failures", ""]
    for fid in rs["fail_ids"][:25]:
        lines.append(f"- `{fid}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge for Basher eval JSON")
    ap.add_argument("results", type=Path, nargs="+", help="One or more results JSON (RAG first, optional baseline second)")
    ap.add_argument("-o", "--output-dir", type=Path, help="Directory for *.judged.json and JUDGE_SUMMARY.md")
    ap.add_argument("--limit", type=int, default=None, help="Judge only first N items (smoke)")
    args = ap.parse_args()

    judged_paths = []
    payloads = []
    for src in args.results:
        if not src.exists():
            print(f"missing {src}", file=sys.stderr)
            return 1
        out_dir = args.output_dir or src.parent
        out = out_dir / (src.stem + ".judged.json")
        print(f"=== judging {src} with {JUDGE_MODEL} -> {out} ===", flush=True)
        payload = judge_file(src, out, limit=args.limit)
        judged_paths.append(out)
        payloads.append(payload)

    rag_p = payloads[0]
    base_p = payloads[1] if len(payloads) > 1 else None
    md = summarize_pair(rag_p, base_p)
    summary_path = (args.output_dir or args.results[0].parent) / "JUDGE_SUMMARY.md"
    summary_path.write_text(md, encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
