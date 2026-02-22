#!/usr/bin/env python3
"""
Run Basher evaluation dataset against the API.

Usage:
  API_URL=http://localhost:8000 python run_evals.py [--dataset path] [--output path] [--no-call]

Output JSON is suitable for a React scoring app: each item has response, context,
retrieval info, auto_pass (for auto-review items), and placeholder for human score.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

try:
    import requests
except ImportError:
    print("run_evals.py requires 'requests'. pip install requests", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "dataset.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "results"
API_URL = os.environ.get("BASHER_API_URL", "http://localhost:8000")


def load_dataset(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_items_with_sections(data: dict) -> List[tuple]:
    """Return list of (item, section_id, section_label) from dataset. Supports both 'sections' and legacy 'items'."""
    sections = data.get("sections")
    if sections:
        out = []
        for sec in sections:
            sec_id = sec.get("id", "")
            label = sec.get("label", sec_id)
            for item in sec.get("items", []):
                out.append((item, sec_id, label))
        return out
    # Legacy flat items
    return [(item, item.get("category", "other"), item.get("category", "other")) for item in data.get("items", [])]


def ask_api(query: str, debug: bool = True, use_base_model: bool = False) -> dict:
    resp = requests.post(
        f"{API_URL}/ask",
        json={"query": query, "debug": debug, "base_model": use_base_model},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def normalize_text(s: str) -> str:
    return (s or "").lower().strip()


def check_expected_contains(response_text: str, expected_contains: Optional[List[str]]) -> bool:
    if not expected_contains:
        return True
    text = normalize_text(response_text)
    return any(normalize_text(phrase) in text for phrase in expected_contains)


def check_must_not_contain(response_text: str, must_not_contain: Optional[List[str]]) -> bool:
    if not must_not_contain:
        return True
    text = normalize_text(response_text)
    return not any(normalize_text(phrase) in text for phrase in must_not_contain)


def run_auto_review(item: dict, response_text: str) -> dict:
    """Returns { "auto_pass": bool, "checks": { "expected_contains": bool, "must_not_contain": bool } }."""
    expected = item.get("expected_contains") or []
    must_not = item.get("must_not_contain") or []
    ok_expected = check_expected_contains(response_text, expected) if expected else True
    ok_must_not = check_must_not_contain(response_text, must_not)
    return {
        "auto_pass": ok_expected and ok_must_not,
        "checks": {
            "expected_contains": ok_expected,
            "must_not_contain": ok_must_not,
        },
    }


def run_evals(
    dataset_path: Path,
    output_path: Path,
    dry_run: bool,
    use_base_model: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[dict]:
    data = load_dataset(dataset_path)
    items_with_sections = get_items_with_sections(data)
    if not items_with_sections:
        print("No items in dataset.", file=sys.stderr)
        return []

    total = len(items_with_sections)
    results = []
    for i, (item, section_id, section_label) in enumerate(items_with_sections):
        current = i + 1
        q = item.get("query", "")
        item_id = item.get("id", f"item-{i}")
        msg = f"[{section_id}] {item_id}"
        if progress_callback:
            progress_callback(current, total, msg)
        print(f"[{current}/{total}] {msg} ...", flush=True)

        base_result = {
            "id": item_id,
            "section_id": section_id,
            "section_label": section_label,
            "query": q,
            "review_type": item.get("review_type", "human"),
            "expected_collections": item.get("expected_collections"),
            "human_rubric": item.get("human_rubric"),
        }

        if dry_run:
            results.append({
                **base_result,
                "response": "",
                "context": "",
                "retrieval": {},
                "auto_pass": None,
                "human_score": None,
                "score_relevance": None,
                "score_context_usage": None,
                "score_correctness": None,
                "dry_run": True,
            })
            continue

        try:
            # Always request debug=True so we get context and retrieval for RAG review
            out = ask_api(q, debug=True, use_base_model=use_base_model)
        except requests.RequestException as e:
            print(f"  Error: {e}", file=sys.stderr)
            results.append({
                **base_result,
                "response": "",
                "context": "",
                "retrieval": {},
                "error": str(e),
                "auto_pass": None,
                "human_score": None,
                "score_relevance": None,
                "score_context_usage": None,
                "score_correctness": None,
            })
            continue

        response_text = out.get("response", "")
        context = out.get("context", "")
        retrieval = out.get("retrieval", {})
        if context == "" and not out.get("retrieval"):
            print(f"  Warning: API returned no context/retrieval (ensure API uses debug from request body).", file=sys.stderr)

        auto_result = {}
        if item.get("review_type") == "auto":
            auto_result = run_auto_review(item, response_text)

        results.append({
            **base_result,
            "response": response_text,
            "context": context,
            "retrieval": retrieval,
            "auto_pass": auto_result.get("auto_pass"),
            "auto_checks": auto_result.get("checks"),
            "human_score": None,
            "score_relevance": None,
            "score_context_usage": None,
            "score_correctness": None,
        })

    return results


def main(
    use_base_model: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    argv: Optional[List[str]] = None,
):
    """Run evaluation. When invoked from the API, pass argv=['run_evals.py'] so
    parse_args() does not consume the process argv (e.g. uvicorn's)."""
    parser = argparse.ArgumentParser(description="Run Basher evaluation dataset")
    parser.add_argument("--dataset", "-d", type=Path, default=DEFAULT_DATASET, help="Path to dataset.json")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="Output dir or file; if dir, writes results_<timestamp>.json")
    parser.add_argument("--no-call", action="store_true", help="Dry run: do not call API, only structure output")
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        if argv is None:
            sys.exit(1)
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    results = run_evals(
        args.dataset,
        args.output,
        dry_run=args.no_call,
        use_base_model=use_base_model,
        progress_callback=progress_callback,
    )

    out_path = args.output
    if out_path.suffix != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        out_path = out_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    payload = {
        "run_meta": {
            "dataset": str(args.dataset),
            "api_url": API_URL,
            "dry_run": args.no_call,
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} results to {out_path}")
    auto_items = [r for r in results if r.get("review_type") == "auto" and r.get("auto_pass") is not None]
    if auto_items:
        passed = sum(1 for r in auto_items if r.get("auto_pass"))
        print(f"Auto-review: {passed}/{len(auto_items)} passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
