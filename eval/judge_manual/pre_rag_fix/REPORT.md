# Corrected grades — Composer as LLM judge

No new Ollama runs. Judgments are over the **saved** answer JSONs.

Files: `eval/judge_manual/*.{judged.json}` + `JUDGE_SUMMARY.md`.

## Headline (judge_pass, all 60 items)

| Run | Judge pass | Avg correctness (1–5) |
|-----|------------|------------------------|
| **Baseline qwen3.5:4b** | **57/60 (95%)** | 4.12 |
| RAG qwen3.5:4b | 47/60 (78%) | 3.68 |
| RAG qwen2.5:1.5b | 52/60 (87%) | 3.92 |
| Baseline qwen2.5:1.5b | 49/60 (82%) | 3.65 |

### vs old auto metric (46 auto items only)

| Run | Auto | Composer judge (60) |
|-----|------|---------------------|
| RAG 4b | 39/46 (85%) | 47/60 (78%) |
| Base 4b | 39/46 (85%) | 57/60 (95%) |
| RAG 1.5b | 42/46 (91%) | 52/60 (87%) |
| Base 1.5b | 40/46 (87%) | 49/60 (82%) |

## What changed after real judging

1. **Auto was too kind to phrase hits and too harsh on “don’t use apt”.**  
   Examples fixed: `fedora-trap-ubuntu`, `fedora-dnf-update` → real passes.
2. **Auto missed real errors.**  
   Examples: RAG-4b `bash-set-u` (“untrap”), `man -H`, missing `docker ps`; base-4b wrong `bash-subshell` (`&`), wrong cherry-pick.
3. **On 4b, RAG often hurt:** model over-refused (“docs don’t say…”) or followed noisy chunks instead of known-correct commands. Net judge lift **−10** vs baseline.
4. **On 1.5b, RAG helped:** net **+3** (traps / stash / fake commands). Matches the ablation story.

## Corrected grade for Tulokset

| Lens | Grade | Comment |
|------|-------|---------|
| Auto-only table | 3/5 | Tied 85%; weak evidence of RAG value |
| **Composer judge (this)** | **3/5** | Stronger methodology; shows RAG **hurts** 4b ask-mode and **helps** 1.5b — honest, not flashy |
| If overclaiming “RAG wins on 4b” | 1–2/5 | Contradicted by judge |

**Project overall (unchanged ballpark): ~3, up to 3–4** if the report states this clearly: official agent model is 4b for tools; ask-RAG on 4b needs better grounding prompts; 1.5b ablation shows when retrieval helps.

## Use in the written report

- Lead with **judge table**, mention auto as a weak baseline metric.
- Do **not** hide that baseline-4b scored higher under judge.
- Sell RAG as: retrieval pipeline works; value depends on model strength; agent+tools is the other half of the system.
