# One Hundred ACT, CLARIFY and REFUSE Examples

[Experiment index](../README.md)

## Question and Context

Could a larger unique BAYMAX contribution improve on the fixed 150-example pool without oversampling exact duplicates?

## Change

The 1,000-row mixture targeted 300 unique BAYMAX examples: 100 productive ACT, 100 CLARIFY and 100 REFUSE, plus 700 generic xLAM productive examples.

"Balanced BAYMAX" is not "balanced full dataset": the xLAM portion still adds ACT targets, including multi-tool sequences.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [100/100/100 BAYMAX](../../../results/phase2/qwen-lora/baymax-mix/act-clarify-refuse-train1k-combined-150.json) | 150 | 43/150 (0.287) | 0.567 | 0.761 | 0.050 | 0.172 |
| [100/100/100 BAYMAX](../../../results/phase2/template-shift/act-clarify-refuse-train1k-template-shift-30.json) | 30 | 6/30 (0.200) | 0.600 | 0.800 | 0.200 | 0.150 |
| [100/100/100 BAYMAX / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-clarify-refuse-train1k-combined-150-strict-parser-rescore.json) | 150 | 42/150 (0.280) | 0.500 | 0.694 | 0.013 | 0.172 |
| [100/100/100 BAYMAX / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/act-clarify-refuse-train1k-template-shift-30-strict-parser-rescore.json) | 30 | 6/30 (0.200) | 0.533 | 0.733 | 0.200 | 0.100 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| 100/100/100 BAYMAX / strict replay (150 cases) | 0.367 | 0.113 |
| 100/100/100 BAYMAX / strict replay (30 cases) | 0.435 | 0.100 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [100/100/100 BAYMAX](../../../results/phase2/qwen-lora/baymax-mix/act-clarify-refuse-train1k-combined-150.json): 2026-09-27 UTC, 150 cases.
- [100/100/100 BAYMAX](../../../results/phase2/template-shift/act-clarify-refuse-train1k-template-shift-30.json): 2026-09-27 UTC, 30 cases.
- [100/100/100 BAYMAX / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-clarify-refuse-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.
- [100/100/100 BAYMAX / strict replay](../../../results/phase2/parser-integrity-v1/template-shift/act-clarify-refuse-train1k-template-shift-30-strict-parser-rescore.json): 2026-09-28 UTC, 30 cases.

## Interpretation

The historical full evaluation recorded 43/150 successes and the shift suite 6/30. Strict-parser replay reduced the full count to 42/150 and some argument/wording metrics.

This was a useful improvement over some early mixes, but the shift suite still exposed poor generalization and output-contract errors.

## Limits

The historical and strict-replay rows are not different trained models. Use strict replay when comparing with the other replayed candidates. Old synthetic split and LoRA seed issues were not fixed by rescoring.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
