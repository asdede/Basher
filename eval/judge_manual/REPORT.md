# Composer LLM-as-judge (after RAG fix)

Judge: Composer over **new** `qwen3.5:4b` RAG/baseline runs (post softer prompt + stricter retrieval + low-score skip).
1.5b ablation removed.

## Judge_pass (60 items)

| Mode | Judge pass | Avg correctness |
|------|------------|-----------------|
| RAG qwen3.5:4b | 57/60 (95%) | 4.55 |
| Baseline qwen3.5:4b | 57/60 (95%) | 4.52 |

Auto (for reference): RAG 39/46 (85%), Baseline 42/46 (91%).

## RAG vs baseline (judge)

- RAG helped: **2** — `bash-subshell`, `git-remote-add`
- RAG hurt: **2** — `fedora-clean-cache`, `linux-find-name`
- Net: **+0**

## vs previous Composer judge (before RAG fix)

| | Before | After |
|--|--------|-------|
| RAG judge | 47/60 (78%) | 57/60 |
| Base judge | 57/60 (95%) | 57/60 |

## Corrected Tulokset grade

Still approximately **3–4 / 5**: methodology improved and RAG refusal bugs largely fixed; baseline remains slightly stronger on judge_pass. Honest framing: retrieval tuning reduced over-refusal; RAG is not yet a clear win vs a strong 4b base on this ask set.

Artifacts: `eval/judge_manual/rag_4b.judged.json`, `base_4b.judged.json`.
