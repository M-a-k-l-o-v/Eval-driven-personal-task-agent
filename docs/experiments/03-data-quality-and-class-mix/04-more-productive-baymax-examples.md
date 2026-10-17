# Increasing Productive BAYMAX Examples to Two Hundred

[Experiment index](../README.md)

## Question and Context

Could more domain-specific productive examples recover task completion without adding more clarification/refusal rows?

## Change

The historical ACT-balanced mixture selected 400 BAYMAX rows: 200 ACT, 100 CLARIFY and 100 REFUSE. The remaining 600 raw slots came from xLAM.

This is the old ACT-balanced model, not generator14 ACT200 or the cleaned anchored checkpoint. The similar class counts do not make the data revisions identical.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Historical ACT200 / legacy metrics](../../../results/phase2/qwen-lora/baymax-mix/act-balanced-train1k-combined-150-legacy-metrics.json) | 150 | 47/150 (0.313) | 0.553 | 0.805 | 0.025 | 0.086 |
| [Historical ACT200 / decision metrics](../../../results/phase2/qwen-lora/baymax-mix/act-balanced-train1k-combined-150-decision-metrics.json) | 150 | 47/150 (0.313) | 0.553 | 0.805 | 0.025 | 0.086 |
| [Historical ACT200 / legacy metrics](../../../results/phase2/template-shift/act-balanced-train1k-template-shift-30-legacy-metrics.json) | 30 | 6/30 (0.200) | 0.533 | 0.828 | 0.200 | 0.100 |
| [Historical ACT200 / decision metrics](../../../results/phase2/template-shift/act-balanced-train1k-template-shift-30-decision-metrics.json) | 30 | 6/30 (0.200) | 0.533 | 0.828 | 0.200 | 0.100 |
| [Historical ACT200 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-balanced-train1k-combined-150-strict-parser-rescore.json) | 150 | 46/150 (0.307) | 0.493 | 0.738 | 0.000 | 0.086 |
| [Historical ACT200 / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/act-balanced-train1k-template-shift-30-strict-parser-rescore.json) | 30 | 6/30 (0.200) | 0.500 | 0.794 | 0.200 | 0.100 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Historical ACT200 / decision metrics (150 cases) | 0.387 | 0.000 |
| Historical ACT200 / decision metrics (30 cases) | 0.397 | 0.000 |
| Historical ACT200 / strict replay (150 cases) | 0.349 | 0.120 |
| Historical ACT200 / strict replay (30 cases) | 0.368 | 0.067 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [Historical ACT200 / legacy metrics](../../../results/phase2/qwen-lora/baymax-mix/act-balanced-train1k-combined-150-legacy-metrics.json): 2026-09-27 UTC, 150 cases.
- [Historical ACT200 / decision metrics](../../../results/phase2/qwen-lora/baymax-mix/act-balanced-train1k-combined-150-decision-metrics.json): 2026-09-27 UTC, 150 cases.
- [Historical ACT200 / legacy metrics](../../../results/phase2/template-shift/act-balanced-train1k-template-shift-30-legacy-metrics.json): 2026-09-27 UTC, 30 cases.
- [Historical ACT200 / decision metrics](../../../results/phase2/template-shift/act-balanced-train1k-template-shift-30-decision-metrics.json): 2026-09-27 UTC, 30 cases.
- [Historical ACT200 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-balanced-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.
- [Historical ACT200 / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/act-balanced-train1k-template-shift-30-strict-parser-rescore.json): 2026-09-28 UTC, 30 cases.

## Interpretation

The historical evaluation recorded 47/150 successes; strict replay recorded 46/150. Template-shift success remained 6/30.

The increase in productive exposure improved some established-suite outcomes but also increased premature actions. It was not a complete improvement in decision policy.

## Limits

The "decision-metrics" evaluation added explicit decision reporting but did not create a new trained model. The "legacy-metrics" file remains preserved.

Strict replay fixes measurement only, not the old training partitions or seed initialization. Later ACT200 runs are not single-variable comparisons with this checkpoint.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
