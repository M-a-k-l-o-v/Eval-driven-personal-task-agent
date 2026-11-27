# Control Data With Contrast Pairs

[Experiment index](../README.md)

## Question and Context

Could related ACT and non-ACT examples teach the model the difference between a complete request and one that should not be acted on?

This was an early contrast-pair trial, before the later anchored matched-group experiment.

## Setup

The training source remained BAYMAX synthetic control data. Contrast pairs were added to make behavioral alternatives more explicit.

The historical result does not pin the selected pairs, effective token distribution, checkpoint hash or exact prompt. Do not substitute today's generator behavior for the missing historical recipe.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Control contrast pairs](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-contrast-pairs-v1-50.json) | 50 | 12/50 (0.240) | 0.530 | 0.767 | 0.111 | 0.071 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Control contrast pairs](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-contrast-pairs-v1-50.json): 2026-08-22 UTC, 50 cases.

## Interpretation

The run completed 12/50 tasks. Clarification and refusal keyword scores were still low. This supported further investigation of boundary training, but did not demonstrate that contrast pairs generalize reliably.

## Limits

Comparisons with the previous control evaluation are historical, not a verified single-variable ablation. The later matched experiment changes group selection and coverage much more explicitly; it is not a repeat of this trial.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
