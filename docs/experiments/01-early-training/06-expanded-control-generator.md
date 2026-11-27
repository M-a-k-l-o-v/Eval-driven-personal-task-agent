# Expanded Control Generator: Early Mixed Trial

[Experiment index](../README.md)

## Question and Context

Would broader authored clarification, refusal and confirmation examples improve the mixed model?

## Setup

This is the synthetic-v7 mixed-data trial. The artifact is named "smoke", but its evaluation contains all 50 v1 scenarios. Evaluation size and training size are different facts.

The user discussed a 5,000-row trial during this period. The surviving result does not independently pin that exposure, so the note does not treat the filename as a training manifest.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Expanded control mix](../../../results/phase2/qwen-lora/baymax-mix/synthetic-v7-smoke-v1-50.json) | 50 | 15/50 (0.300) | 0.500 | 0.865 | 0.000 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Expanded control mix](../../../results/phase2/qwen-lora/baymax-mix/synthetic-v7-smoke-v1-50.json): 2026-09-15 UTC, 50 cases.

## Interpretation

Task success reached 15/50, but clarification and refusal keyword metrics were both zero. Better productive task completion did not establish better boundary judgment.

## Limits

No frozen effective class/token counts or exact recipe are stored in the result. Do not compare this 50-case rate directly with later 150-case rates. The generator version is a historical label, not an immutable checkpoint identity.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
