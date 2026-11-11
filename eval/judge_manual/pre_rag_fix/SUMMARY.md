# Basher eval summary — baseline vs RAG

## Run meta

- RAG file: `.cursor/eval_results/rag_qwen35_4b.json`
- Model: `qwen3.5:4b`
- num_ctx: `12288`
- API: `http://localhost:8000`
- Dataset: `/home/masterbates/code/school/Basher/eval/dataset.json`
- Baseline file: `.cursor/eval_results/base_qwen35_4b.json`
- Baseline base_model flag: `True`

## Auto-review (overall)

| Mode | Auto pass |
|------|-----------|
| RAG | 39/46 (85%) |
| Baseline (no RAG) | 39/46 (85%) |

## Auto-review by section

| Section | RAG | Baseline |
|---------|-----|----------|
| bash | 7/8 | 7/8 |
| docker | 12/13 | 12/13 |
| fedora | 5/8 | 6/8 |
| git | 8/10 | 8/10 |
| linux | 7/7 | 6/7 |

## RAG auto failures

- `fedora-dnf-update`
- `no-hallucination-command`
- `fedora-trap-ubuntu`
- `git-undo-last-commit`
- `git-hallucination-fake-cmd`
- `bash-trap-hallucination`
- `docker-hallucination-fake-cmd`

## Baseline-only notes

- Passed with RAG but failed baseline: **1**
  - `linux-fake-command`
- Failed in both: **6**
  - `bash-trap-hallucination`
  - `docker-hallucination-fake-cmd`
  - `fedora-trap-ubuntu`
  - `git-hallucination-fake-cmd`
  - `git-undo-last-commit`
  - `no-hallucination-command`

## Notes for report

- Auto scores use `expected_contains` / `must_not_contain` only — **superseded for grading by Composer judge** in `eval/judge_manual/` (see `JUDGE_SUMMARY.md` / `REPORT.md`).
- Composer judge (60 items): baseline **4b 57/60 (95%)** vs RAG **4b 47/60 (78%)** — RAG often over-refused or followed noisy docs. Ablation **1.5b**: RAG **52/60** vs base **49/60** (RAG helps).
- Stack freeze: `qwen3.5:4b`, `num_ctx=12288`, host Ollama (GPU) + Compose Qdrant/API (`docker-compose.host-ollama.yml`).
- `/ask` uses Ollama `think: false` — without it, qwen3.5 spent `num_predict` on thinking and returned empty answers.
- Artifacts: `eval/rag_qwen35_4b.json`, `eval/base_qwen35_4b.json`, `eval/judge_manual/*.judged.json`.
