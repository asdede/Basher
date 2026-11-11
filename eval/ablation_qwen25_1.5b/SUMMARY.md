# Basher eval summary — baseline vs RAG

## Run meta

- RAG file: `.cursor/eval_results/ablation_qwen25_1.5b/rag_qwen25_1.5b.json`
- Model: `qwen2.5:1.5b`
- num_ctx: `12288`
- API: `http://localhost:8000`
- Dataset: `/home/masterbates/code/school/Basher/eval/dataset.json`
- Baseline file: `.cursor/eval_results/ablation_qwen25_1.5b/base_qwen25_1.5b.json`
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
| docker | 13/13 | 12/13 |
| fedora | 5/8 | 7/8 |
| git | 7/10 | 9/10 |
| linux | 7/7 | 7/7 |

## RAG auto failures

- `fedora-dnf-update`
- `no-hallucination-command`
- `fedora-trap-ubuntu`
- `git-undo-last-commit`
- `git-cherry-pick`
- `git-stash`
- `bash-trap-hallucination`

## Baseline-only notes

- Passed with RAG but failed baseline: **1**
  - `docker-hallucination-fake-cmd`
- Failed in both: **3**
  - `bash-trap-hallucination`
  - `git-undo-last-commit`
  - `no-hallucination-command`

## Notes for report

- Auto scores use `expected_contains` / `must_not_contain` only; human items need the eval app.
- Compare retrieval (`expected_collections` vs `retrieval`) when discussing RAG quality.

## Composer judge (post RAG-fix)

| Mode | Judge pass | Avg correctness |
|------|------------|-----------------|
| RAG | 51/60 (85%) | 4.02 |
| Baseline | 55/60 (92%) | 4.35 |

See `eval/judge_manual/ablation_1.5b/JUDGE_SUMMARY.md`. Net RAG vs base: -4.
