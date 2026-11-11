# Basher evaluation

Evaluation dataset and runner for the RAG-trained Basher assistant (Fedora, Bash, Linux, Docker, git).

## Dataset (`dataset.json`)

The dataset is organized in **sections**:

| Section id   | Label            |
|-------------|------------------|
| `fedora`    | Fedora-based     |
| `git`       | Git              |
| `bash`      | Bash             |
| `linux`     | Linux commands   |
| `docker`    | Docker           |

- **Items**: Each has `id`, `query`, `review_type`, and optional RAG/data fields. Section is implied by parent.
- **review_type**
  - **`auto`**: Runner checks whether the response contains expected phrases and avoids forbidden ones. Use for factual, command-oriented questions.
  - **`human`**: No automatic pass/fail; score via the React app. Use for nuance, helpfulness, and RAG behaviour (e.g. “did it use the docs?”).
- **RAG-oriented fields** (for analysis and human review):
  - `expected_collections`: Collections that *should* contribute (e.g. `fedora_quick_docs`, `bash_docs`). Runner includes this in results; your app can compare to `retrieval.collections`.
  - `human_rubric`: Short instructions for human reviewers (e.g. “Check that the answer uses documentation and does not hallucinate”).
  - `expected_contains` / `must_not_contain`: Used only for `review_type: "auto"`.

## Running evals

1. Start the API (e.g. `./Scripts/basher.sh --start` or `docker compose up -d`).
2. Confirm freeze: `curl -s localhost:8000/health` → expected model + `num_ctx`.
3. From project root:
   ```bash
   # RAG (retrieval on)
   python eval/run_evals.py -o eval/rag_qwen35_4b.json

   # Baseline (no RAG)
   python eval/run_evals.py --base-model -o eval/base_qwen35_4b.json

   # Dry run (no API calls; useful to check output shape)
   python eval/run_evals.py --no-call

   # Markdown summary for the report
   python eval/summarize_results.py \
     eval/rag_qwen35_4b.json \
     eval/base_qwen35_4b.json \
     -o eval/SUMMARY.md
   ```
4. Optional env: `BASHER_API_URL=http://localhost:8000` (default).

**Requires:** `requests` (e.g. `pip install requests` or use the project’s API venv).

`run_meta` in each results file records `mode`, `base_model`, `model`, and `num_ctx` from `/health`.

Latest frozen run (2026-11-11): auto **39/46 (85%)** for both RAG and baseline — see `eval/SUMMARY.md`.


## Output format (for React scoring app)

Output is a single JSON file per run, e.g. `eval/results/results_20250222_120000.json`:

```json
{
  "run_meta": {
    "dataset": "/path/to/dataset.json",
    "api_url": "http://localhost:8000",
    "dry_run": false
  },
  "results": [
    {
      "id": "bash-loop-files",
      "section_id": "bash",
      "section_label": "Bash",
      "query": "How do I loop over all .txt files...",
      "response": "...",
      "context": "...",
      "retrieval": { "chunks_used": 5, "collections": [...] },
      "review_type": "auto",
      "auto_pass": true,
      "auto_checks": { "expected_contains": true, "must_not_contain": true },
      "human_score": null,
      "expected_collections": ["bash_docs"],
      "human_rubric": null
    },
    {
      "id": "explain-pipe",
      "section_id": "bash",
      "section_label": "Bash",
      "query": "Explain what a pipe does...",
      "response": "...",
      "context": "...",
      "retrieval": { ... },
      "review_type": "human",
      "auto_pass": null,
      "human_score": null,
      "human_rubric": "Clarity, correctness..."
    }
  ]
}
```

- **Auto items**: Set `human_score` in your app if you want to add a human override or note; `auto_pass` is already set by the runner.
- **Human items**: Your React app only needs to set `human_score` (e.g. 1–5 or pass/fail).
- **RAG**: Use `context` and `retrieval.collections` to show what was retrieved; compare with `expected_collections` for retrieval-quality review.

## What to evaluate (RAG-trained model)

- **Retrieval**: Right collections and chunks used? (see `retrieval` and `expected_collections`).
- **Grounding**: Does the answer stay close to the retrieved context vs hallucinating?
- **Completeness**: For commands, are options and caveats from the docs mentioned when relevant?
- **Scope**: For out-of-scope questions, does the model decline or redirect appropriately?
- **Human-only**: Helpfulness, tone, conciseness, and “did it use the docs?” are best scored by humans using `human_rubric`.
