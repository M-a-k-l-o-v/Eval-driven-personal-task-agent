# Returning to the Recorded Training Prompt

[Experiment index](../README.md)

## Question and Context

Were changes in inference instructions making checkpoint comparisons misleading?

## Change

Evaluate base Qwen, aligned and both boundary-diverse candidates with the same recorded training prompt. Its SHA256 is:

`a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468`

This did not retrain the adapters. It restored a matched inference prompt for the comparison.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Bare Qwen / matched training prompt](../../../results/phase2/qwen-base/qwen-base-combined-150-training-prompt-a314.json) | 150 | 32/150 (0.213) | 0.493 | 0.791 | 0.025 | 0.034 |
| [Format-aligned / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-training-prompt-a314.json) | 150 | 40/150 (0.267) | 0.453 | 0.771 | 0.025 | 0.017 |
| [Boundary pool / 1,000 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-training-prompt-a314.json) | 150 | 37/150 (0.247) | 0.493 | 0.738 | 0.050 | 0.086 |
| [Boundary pool / 1,500 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-training-prompt-a314.json) | 150 | 34/150 (0.227) | 0.473 | 0.708 | 0.013 | 0.103 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Bare Qwen / matched training prompt](../../../results/phase2/qwen-base/qwen-base-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.
- [Format-aligned / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.
- [Boundary pool / 1,000 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.
- [Boundary pool / 1,500 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.

## Interpretation

Aligned recovered to 40/150 successes, compared with 26/150 under policy prompt v1 and 6/150 under v2. Prompt compatibility was a real measurement concern.

Neither boundary-diverse candidate dominated aligned: additional boundary exposure improved some abstention metrics while reducing other task outcomes.

## Limits

This comparison does not prove that the old training prompt is the best possible policy. It demonstrates why prompt changes must be tracked independently from training changes.

The historic suite policy and parser later changed. Keep these artifacts as historical evidence, not directly interchangeable with today's strict-parser results.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
