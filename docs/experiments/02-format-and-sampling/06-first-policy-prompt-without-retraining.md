# First Policy Prompt: Evaluation Without Retraining

[Experiment index](../README.md)

## Question and Context

Would explicitly describing ACT, CLARIFY and REFUSE improve the existing models at inference time?

## Change

Evaluate bare Qwen and the existing aligned adapter with the first decision-policy prompt. The aligned checkpoint was not retrained with this new prompt.

The intended policy gives refusal precedence for unsafe/unavailable capabilities, clarification for recoverable missing details or required confirmation, and ACT for supported complete requests.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Bare Qwen / policy prompt v1](../../../results/phase2/qwen-base/qwen-base-combined-150-policy-prompt-v1.json) | 150 | 37/150 (0.247) | 0.537 | 0.827 | 0.013 | 0.034 |
| [Format-aligned / policy prompt v1](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-policy-prompt-v1.json) | 150 | 26/150 (0.173) | 0.433 | 0.653 | 0.100 | 0.069 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Bare Qwen / policy prompt v1](../../../results/phase2/qwen-base/qwen-base-combined-150-policy-prompt-v1.json): 2026-09-26 UTC, 150 cases.
- [Format-aligned / policy prompt v1](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-policy-prompt-v1.json): 2026-09-26 UTC, 150 cases.

## Interpretation

The new prompt measured 37/150 successes for base Qwen and 26/150 for aligned. That contrast raised a prompt-distribution-shift concern: a checkpoint trained with one prompt may behave differently under a new policy prompt.

## Limits

This is a prompt-only experiment, not evidence of policy-aligned fine-tuning. Historical policy-era fixtures/scoring and previous prompt evaluations also differ, so not every historical comparison is a controlled prompt ablation.

The saved failure summaries diagnose the outputs of these runs. They are not labels to use for fixture-specific training.

## Supporting Files

- [qwen-base-policy-prompt-v1-failure-summary.txt](../../../results/phase2/analysis/qwen-base-policy-prompt-v1-failure-summary.txt)
- [format-aligned-policy-prompt-v1-failure-summary.txt](../../../results/phase2/analysis/format-aligned-policy-prompt-v1-failure-summary.txt)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
