# Composer LLM-as-judge summary

Judge: **Composer** (manual review of saved answers). No new model generations.
Auto phrase-matching false-negatives (e.g. mentioning `apt` while forbidding it) were corrected.

## Overall judge_pass

| Run | Judge pass | Avg correctness |
|-----|------------|-----------------|
| RAG qwen3.5:4b | 47/60 (78%) | 3.68 |
| Baseline qwen3.5:4b | 57/60 (95%) | 4.12 |
| RAG qwen2.5:1.5b | 52/60 (87%) | 3.92 |
| Baseline qwen2.5:1.5b | 49/60 (82%) | 3.65 |

## RAG lift (judge)

| Model | RAG helped | RAG hurt | Net |
|-------|------------|----------|-----|
| qwen3.5:4b | 3 (`out-of-scope-polite`, `git-cherry-pick`, `bash-subshell`) | 13 (`fedora-search-package`, `no-hallucination-command`, `fedora-repo-list`, `git-reflog`, `git-remote-add`, `bash-set-u`, `man-page-exists`, `linux-find-name`, `linux-xargs-basic`, `docker-list-containers`, `docker-stop-container`, `docker-run-detached`, `docker-image-vs-container`) | -10 |
| qwen2.5:1.5b | 6 (`git-undo-last-commit`, `git-stash`, `git-hallucination-fake-cmd`, `multi-source`, `linux-redirect-stderr`, `linux-fake-command`) | 3 (`fedora-repo-list`, `git-diff-staged`, `linux-find-name`) | +3 |

## Corrected grade note (Tulokset)

Under auto matching, 4b RAG and baseline **tied at 85%**. Under Composer judge, rates and item-level errors are more honest (e.g. `bash-set-u` wrong on RAG-4b; `fedora-trap-ubuntu` is a true pass).  
**1.5b still shows a clearer RAG benefit** on hallucination/trap items.  
Official product model remains **qwen3.5:4b** (tools); 1.5b is ablation only.

Artifacts: `eval/judge_manual/*.judged.json`
