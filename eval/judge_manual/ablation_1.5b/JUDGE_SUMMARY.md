# Composer LLM-as-judge — qwen2.5:1.5b ablation

Judge: Composer over post-RAG-fix `qwen2.5:1.5b` RAG/baseline runs (same API gates as 4b).
Official freeze remains `qwen3.5:4b`; this is ablation only.

## Judge_pass (60 items)

| Mode | Judge pass | Avg correctness |
|------|------------|-----------------|
| RAG qwen2.5:1.5b | 51/60 (85%) | 4.02 |
| Baseline qwen2.5:1.5b | 55/60 (92%) | 4.35 |

Auto (for reference): RAG 39/46 (85%), Baseline 42/46 (91%).

## RAG vs baseline (judge)

- RAG helped: **3** — `bash-vs-man`, `docker-hallucination-fake-cmd`, `docker-run-port`
- RAG hurt: **7** — `bash-quoting`, `git-cherry-pick`, `git-diff-staged`, `git-reflog`, `git-stash`, `git-undo-last-commit`, `linux-xargs-basic`
- Net: **-4**

## Fail IDs

- RAG fails (9): `out-of-scope-polite`, `git-diff-staged`, `git-undo-last-commit`, `git-reflog`, `git-cherry-pick`, `git-stash`, `bash-quoting`, `bash-subshell`, `linux-xargs-basic`
- Base fails (5): `out-of-scope-polite`, `bash-vs-man`, `bash-subshell`, `docker-run-port`, `docker-hallucination-fake-cmd`

## vs official qwen3.5:4b (Composer judge, post RAG-fix)

| | 1.5b | 4b |
|--|------|-----|
| RAG judge | 51/60 | 57/60 |
| Base judge | 55/60 | 57/60 |

## Verdict for project

- **Do not switch official model to 1.5b.** Absolute quality is clearly worse (more wrong primary commands: stash/cherry-pick/subshell/quoting under RAG).
- On 1.5b, **baseline beats RAG** on judge_pass (net -4): weak model + noisy chunks still produces confabulation even after retrieval gates.
- Useful ablation evidence only: smaller models need RAG more in theory, but this ask-set shows RAG still hurts net until retrieval/chunk quality improves further.
- Keep `qwen3.5:4b` as freeze; cite 1.5b as negative control / size ablation in docs.

Artifacts: `eval/judge_manual/ablation_1.5b/rag_1.5b.judged.json`, `base_1.5b.judged.json`.
