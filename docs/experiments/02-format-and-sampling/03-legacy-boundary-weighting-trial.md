# Legacy Boundary-Weighting Trial

[Experiment index](../README.md)

## Question and Context

This is a separate early weighting result that could otherwise be confused with the extra-six/extra-twelve run.

## Recoverable Setup

The stored filename identifies a boundary-weighting trial historically called "legacy13". The evaluation contains 50 scenarios. Its exact repeat factors and training recipe are not established by that label.

Two token-audit snapshots exist. They should be inspected as separate snapshots; neither should be silently assumed to describe all historical weighting trials.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Legacy weighting](../../../results/phase2/qwen-lora/baymax-mix/boundary-weighting-legacy13-v1-50.json) | 50 | 12/50 (0.240) | 0.560 | 0.783 | 0.111 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Legacy weighting](../../../results/phase2/qwen-lora/baymax-mix/boundary-weighting-legacy13-v1-50.json): 2026-09-16 UTC, 50 cases.

## Interpretation

Success was 12/50, clarification wording accuracy 0.111 and refusal zero. It did not solve refusal behavior.

## Limits

The recipe is not sufficiently pinned for exact reproduction or a one-variable comparison with the other weighted runs. This note deliberately preserves that uncertainty rather than assigning today's configuration to an old result.

## Supporting Files

- [boundary-weighting-legacy13-token-audit-5000-snapshot1.json](../../../results/phase2/analysis/boundary-weighting-legacy13-token-audit-5000-snapshot1.json)
- [boundary-weighting-legacy13-token-audit-5000-snapshot2.json](../../../results/phase2/analysis/boundary-weighting-legacy13-token-audit-5000-snapshot2.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
