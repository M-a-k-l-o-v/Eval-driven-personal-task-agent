# Productive Midpoint: One Hundred Fifty BAYMAX ACT Examples

[Experiment index](../README.md)

## Question and Context

Could a midpoint between 100 and 200 authored ACT examples retain productive accuracy without the same premature-action pressure?

## Change

The historical ACT-midpoint candidate reduced the BAYMAX productive quota to 150 while retaining the boundary-training approach. It was evaluated with decision metadata and later strict replay.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Historical ACT150 / decision metrics](../../../results/phase2/qwen-lora/baymax-mix/act-midpoint150-train1k-combined-150-decision-metrics.json) | 150 | 46/150 (0.307) | 0.520 | 0.804 | 0.025 | 0.052 |
| [Historical ACT150 / decision metrics](../../../results/phase2/template-shift/act-midpoint150-train1k-template-shift-30-decision-metrics.json) | 30 | 6/30 (0.200) | 0.533 | 0.828 | 0.200 | 0.050 |
| [Historical ACT150 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-midpoint150-train1k-combined-150-strict-parser-rescore.json) | 150 | 46/150 (0.307) | 0.480 | 0.764 | 0.025 | 0.052 |
| [Historical ACT150 / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/act-midpoint150-train1k-template-shift-30-strict-parser-rescore.json) | 30 | 6/30 (0.200) | 0.467 | 0.761 | 0.200 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Historical ACT150 / decision metrics (150 cases) | 0.330 | 0.000 |
| Historical ACT150 / decision metrics (30 cases) | 0.343 | 0.000 |
| Historical ACT150 / strict replay (150 cases) | 0.308 | 0.107 |
| Historical ACT150 / strict replay (30 cases) | 0.321 | 0.100 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [Historical ACT150 / decision metrics](../../../results/phase2/qwen-lora/baymax-mix/act-midpoint150-train1k-combined-150-decision-metrics.json): 2026-09-27 UTC, 150 cases.
- [Historical ACT150 / decision metrics](../../../results/phase2/template-shift/act-midpoint150-train1k-template-shift-30-decision-metrics.json): 2026-09-27 UTC, 30 cases.
- [Historical ACT150 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-midpoint150-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.
- [Historical ACT150 / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/act-midpoint150-train1k-template-shift-30-strict-parser-rescore.json): 2026-09-28 UTC, 30 cases.

## Interpretation

Full-suite success was 46/150 before and after replay; shift success was 6/30. Other metrics changed under strict parsing even though task-success totals stayed the same.

The result did not establish a clean win over ACT-balanced.

## Limits

Matching success totals does not imply matching outputs or error types. The surviving historical result is not a complete frozen training manifest. Earlier seed/split defects still apply.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
