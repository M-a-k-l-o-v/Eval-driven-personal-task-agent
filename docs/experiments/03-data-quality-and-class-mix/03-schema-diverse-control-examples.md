# Schema-Diverse Control Examples

[Experiment index](../README.md)

## Question and Context

Was the model recognizing authoring templates without learning the tools' real argument schemas?

## Change

Diversify synthetic productive examples and schema usage while keeping the broad 100 ACT / 100 CLARIFY / 100 REFUSE BAYMAX contribution.

The goal was transfer across wording and arguments, not rewriting evaluation cases or accepting malformed outputs.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Schema-diverse](../../../results/phase2/qwen-lora/baymax-mix/schema-diverse-train1k-combined-150.json) | 150 | 42/150 (0.280) | 0.560 | 0.752 | 0.075 | 0.190 |
| [Schema-diverse](../../../results/phase2/template-shift/schema-diverse-train1k-template-shift-30.json) | 30 | 5/30 (0.167) | 0.533 | 0.733 | 0.300 | 0.200 |
| [Schema-diverse / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/schema-diverse-train1k-combined-150-strict-parser-rescore.json) | 150 | 40/150 (0.267) | 0.480 | 0.672 | 0.025 | 0.172 |
| [Schema-diverse / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/schema-diverse-train1k-template-shift-30-strict-parser-rescore.json) | 30 | 5/30 (0.167) | 0.500 | 0.700 | 0.300 | 0.150 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Schema-diverse / strict replay (150 cases) | 0.395 | 0.133 |
| Schema-diverse / strict replay (30 cases) | 0.323 | 0.100 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [Schema-diverse](../../../results/phase2/qwen-lora/baymax-mix/schema-diverse-train1k-combined-150.json): 2026-09-27 UTC, 150 cases.
- [Schema-diverse](../../../results/phase2/template-shift/schema-diverse-train1k-template-shift-30.json): 2026-09-27 UTC, 30 cases.
- [Schema-diverse / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/schema-diverse-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.
- [Schema-diverse / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/schema-diverse-train1k-template-shift-30-strict-parser-rescore.json): 2026-09-28 UTC, 30 cases.

## Interpretation

Historical full-suite success was 42/150; strict replay measured 40/150. Shift success was 5/30.

Higher boundary wording scores than some ACT-heavy variants did not make this checkpoint a universally better model. Correct routing, valid output and correct arguments remain separate capabilities.

## Limits

A template-shift argument mismatch can be a genuine schema error, not a formatting false negative. Scorer changes require independent justification.

The old split/initialization caveats remain. Today's generator implementation cannot by itself recover the exact historical selection.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
