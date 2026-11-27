# Weighted Control Examples: Early Oversampling Trial

[Experiment index](../README.md)

## Question and Context

Would increasing the exposure of clarification and refusal examples make the model use those decisions?

## Setup

A weighted/oversampled version of the synthetic-v7 mixture was trained. Weighting changes how often an existing example appears; it does not add semantic diversity.

The precise repeat settings and effective training prefix are not pinned in the saved evaluation. Later artifact audits were added because intended source weights did not reliably describe the data surviving a cap.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Weighted control mix](../../../results/phase2/qwen-lora/baymax-mix/synthetic-v7-weighted-v1-50.json) | 50 | 11/50 (0.220) | 0.540 | 0.809 | 0.028 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Weighted control mix](../../../results/phase2/qwen-lora/baymax-mix/synthetic-v7-weighted-v1-50.json): 2026-09-15 UTC, 50 cases.

## Interpretation

The run completed 11/50 tasks, versus 15/50 in the early unweighted v7 artifact. Clarification remained near zero and refusal remained zero.

This particular weighting attempt did not establish improvement. It does not prove all forms of weighting are ineffective.

## Limits

The comparison is historical and lacks sufficient provenance for a clean causal attribution. Repetition can crowd out unique data, but these metrics alone cannot demonstrate memorization.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
