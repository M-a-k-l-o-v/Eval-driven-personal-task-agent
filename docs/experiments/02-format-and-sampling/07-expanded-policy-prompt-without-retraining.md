# Expanded Policy Prompt: Evaluation Without Retraining

[Experiment index](../README.md)

## Question and Context

Could a more explicit policy explanation correct the shortcomings of the first inference prompt?

## Change

Use an expanded policy prompt on the same base and aligned model candidates. No newly trained policy-prompt adapter is established by these artifacts.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Bare Qwen / policy prompt v2](../../../results/phase2/qwen-base/qwen-base-combined-150-policy-prompt-v2.json) | 150 | 5/150 (0.033) | 0.480 | 0.521 | 0.113 | 0.052 |
| [Format-aligned / policy prompt v2](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-policy-prompt-v2.json) | 150 | 6/150 (0.040) | 0.447 | 0.492 | 0.237 | 0.190 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Bare Qwen / policy prompt v2](../../../results/phase2/qwen-base/qwen-base-combined-150-policy-prompt-v2.json): 2026-09-26 UTC, 150 cases.
- [Format-aligned / policy prompt v2](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-policy-prompt-v2.json): 2026-09-26 UTC, 150 cases.

## Interpretation

Task success dropped to 5/150 for base Qwen and 6/150 for aligned, while clarification/refusal wording scores rose. More boundary-language matches did not imply better overall behavior.

A longer or clearer human-readable instruction is not automatically a better prompt for a small checkpoint trained under a different format.

## Limits

These are single inference evaluations. They do not prove prompt length itself caused the regression, nor that Qwen cannot learn the policy. Do not relax evaluation targets to turn the wording gains into successes.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
