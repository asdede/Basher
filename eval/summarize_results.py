#!/usr/bin/env python3
"""
Summarize one or two eval result JSON files (RAG vs baseline).

Usage:
  python eval/summarize_results.py eval/results/rag_....json eval/results/base_....json
  python eval/summarize_results.py eval/results/rag_....json eval/results/base_....json -o eval/results/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_stats(results: list[dict]) -> dict[str, Any]:
    by_section: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0})
    n = passed = errors = 0
    fails: list[str] = []
    for r in results:
        if r.get("review_type") != "auto":
            continue
        if r.get("auto_pass") is None and r.get("error"):
            errors += 1
            fails.append(f"{r.get('id')}: ERROR {r.get('error')}")
            continue
        if r.get("auto_pass") is None:
            continue
        n += 1
        sec = r.get("section_id") or "other"
        by_section[sec]["n"] += 1
        if r.get("auto_pass"):
            passed += 1
            by_section[sec]["pass"] += 1
        else:
            fails.append(r.get("id") or "?")
    return {
        "n": n,
        "pass": passed,
        "rate": (passed / n) if n else None,
        "errors": errors,
        "by_section": dict(by_section),
        "fails": fails,
    }


def fmt_rate(s: dict[str, Any]) -> str:
    if not s["n"]:
        return "n/a"
    return f"{s['pass']}/{s['n']} ({100 * s['pass'] / s['n']:.0f}%)"


def render(rag_path: Path, base_path: Path | None, rag: dict, base: dict | None) -> str:
    rm = rag.get("run_meta") or {}
    lines = [
        "# Basher eval summary — baseline vs RAG",
        "",
        "## Run meta",
        "",
        f"- RAG file: `{rag_path}`",
        f"- Model: `{rm.get('model')}`",
        f"- num_ctx: `{rm.get('num_ctx')}`",
        f"- API: `{rm.get('api_url')}`",
        f"- Dataset: `{rm.get('dataset')}`",
    ]
    if base_path:
        bm = (base or {}).get("run_meta") or {}
        lines += [
            f"- Baseline file: `{base_path}`",
            f"- Baseline base_model flag: `{bm.get('base_model')}`",
        ]
    lines += ["", "## Auto-review (overall)", ""]

    rs = auto_stats(rag.get("results") or [])
    lines.append(f"| Mode | Auto pass |")
    lines.append(f"|------|-----------|")
    lines.append(f"| RAG | {fmt_rate(rs)} |")
    bs = None
    if base is not None:
        bs = auto_stats(base.get("results") or [])
        lines.append(f"| Baseline (no RAG) | {fmt_rate(bs)} |")

    lines += ["", "## Auto-review by section", "", "| Section | RAG | Baseline |", "|---------|-----|----------|"]
    sections = sorted(set(rs["by_section"]) | (set(bs["by_section"]) if bs else set()))
    for sec in sections:
        rsec = rs["by_section"].get(sec, {"n": 0, "pass": 0})
        rtxt = f"{rsec['pass']}/{rsec['n']}" if rsec["n"] else "—"
        if bs:
            bsec = bs["by_section"].get(sec, {"n": 0, "pass": 0})
            btxt = f"{bsec['pass']}/{bsec['n']}" if bsec["n"] else "—"
        else:
            btxt = "—"
        lines.append(f"| {sec} | {rtxt} | {btxt} |")

    lines += ["", "## RAG auto failures", ""]
    if rs["fails"]:
        for fid in rs["fails"][:20]:
            lines.append(f"- `{fid}`")
        if len(rs["fails"]) > 20:
            lines.append(f"- … and {len(rs['fails']) - 20} more")
    else:
        lines.append("- (none)")

    if bs is not None:
        lines += ["", "## Baseline-only notes", ""]
        rag_fail = set(rs["fails"])
        base_fail = set(bs["fails"])
        rag_helps = sorted(base_fail - rag_fail)
        both_fail = sorted(rag_fail & base_fail)
        lines.append(f"- Passed with RAG but failed baseline: **{len(rag_helps)}**")
        for fid in rag_helps[:15]:
            lines.append(f"  - `{fid}`")
        lines.append(f"- Failed in both: **{len(both_fail)}**")
        for fid in both_fail[:15]:
            lines.append(f"  - `{fid}`")

    lines += [
        "",
        "## Notes for report",
        "",
        "- Auto scores use `expected_contains` / `must_not_contain` only; human items need the eval app.",
        "- Compare retrieval (`expected_collections` vs `retrieval`) when discussing RAG quality.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize Basher eval JSON (RAG vs baseline)")
    p.add_argument("rag", type=Path, help="RAG results JSON")
    p.add_argument("base", type=Path, nargs="?", help="Optional baseline results JSON")
    p.add_argument("-o", "--output", type=Path, help="Write markdown here (default: stdout)")
    args = p.parse_args()

    if not args.rag.exists():
        print(f"Not found: {args.rag}", file=sys.stderr)
        return 1
    rag = load(args.rag)
    base = None
    if args.base:
        if not args.base.exists():
            print(f"Not found: {args.base}", file=sys.stderr)
            return 1
        base = load(args.base)

    md = render(args.rag, args.base, rag, base)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
