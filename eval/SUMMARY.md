# Basher eval summary — baseline vs RAG

## Run meta

- RAG file: `eval/rag_qwen35_4b.json`
- Model: `qwen3.5:4b`
- num_ctx: `12288`
- API: `http://localhost:8000`
- Dataset: `/home/masterbates/code/school/Basher/eval/dataset.json`
- Baseline file: `eval/base_qwen35_4b.json`
- Baseline base_model flag: `True`

## Auto-review (overall)

| Mode | Auto pass |
|------|-----------|
| RAG | 39/46 (85%) |
| Baseline (no RAG) | 42/46 (91%) |

## Auto-review by section

| Section | RAG | Baseline |
|---------|-----|----------|
| bash | 7/8 | 7/8 |
| docker | 12/13 | 13/13 |
| fedora | 5/8 | 6/8 |
| git | 8/10 | 9/10 |
| linux | 7/7 | 7/7 |

## RAG auto failures

- `fedora-dnf-update`
- `no-hallucination-command`
- `fedora-trap-ubuntu`
- `git-undo-last-commit`
- `git-hallucination-fake-cmd`
- `bash-trap-hallucination`
- `docker-hallucination-fake-cmd`

## Baseline-only notes

- Passed with RAG but failed baseline: **0**
- Failed in both: **4**
  - `bash-trap-hallucination`
  - `fedora-trap-ubuntu`
  - `git-undo-last-commit`
  - `no-hallucination-command`

## Notes for report

- Auto scores use `expected_contains` / `must_not_contain` only (brittle; e.g. “don’t use apt” fails).
- **Composer judge (preferred):** both RAG and baseline **57/60 (95%)** after RAG fix — see `eval/judge_manual/JUDGE_SUMMARY.md`.
- RAG fix (softer ask prompt, `RAG_MIN_SCORE=0.40`, top-4, skip docs if best &lt; `0.45`) raised judge RAG from **47/60 → 57/60**.
- Size ablation `qwen2.5:1.5b` (post-fix): judge RAG **51/60**, base **55/60** — see `eval/judge_manual/ablation_1.5b/`. Official freeze remains `qwen3.5:4b`.
- Pre-fix judged artifacts archived under `eval/judge_manual/pre_rag_fix/`.
