# Strict Parser Replay: Correcting Measurement, Not Training

[Experiment index](../README.md)

## Question and Context

Could permissive parsing execute a valid-looking part of an otherwise invalid model response, overstating benchmark performance?

## Change and Method

Reparse saved raw generations with strict-complete-response-v1, replay fake tool execution, and write separate result files. No model inference or training was rerun.

Malformed calls, invalid meta-tool fields, conflicting decisions and trailing output reject the whole response. There is no accepted valid subset or automatic repair.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [100/100/100 BAYMAX / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-clarify-refuse-train1k-combined-150-strict-parser-rescore.json) | 150 | 42/150 (0.280) | 0.500 | 0.694 | 0.013 | 0.172 |
| [Schema-diverse / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/schema-diverse-train1k-combined-150-strict-parser-rescore.json) | 150 | 40/150 (0.267) | 0.480 | 0.672 | 0.025 | 0.172 |
| [Historical ACT200 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-balanced-train1k-combined-150-strict-parser-rescore.json) | 150 | 46/150 (0.307) | 0.493 | 0.738 | 0.000 | 0.086 |
| [Historical ACT150 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-midpoint150-train1k-combined-150-strict-parser-rescore.json) | 150 | 46/150 (0.307) | 0.480 | 0.764 | 0.025 | 0.052 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| 100/100/100 BAYMAX / strict replay (150 cases) | 0.367 | 0.113 |
| Schema-diverse / strict replay (150 cases) | 0.395 | 0.133 |
| Historical ACT200 / strict replay (150 cases) | 0.349 | 0.120 |
| Historical ACT150 / strict replay (150 cases) | 0.308 | 0.107 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [100/100/100 BAYMAX / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-clarify-refuse-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.
- [Schema-diverse / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/schema-diverse-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.
- [Historical ACT200 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-balanced-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.
- [Historical ACT150 / strict replay](../../../results/phase2/parser-integrity-v1/combined-150/act-midpoint150-train1k-combined-150-strict-parser-rescore.json): 2026-09-28 UTC, 150 cases.

## Interpretation

Full-suite success counts changed from 43 to 42 for ACR, 42 to 40 for schema-diverse, 47 to 46 for ACT-balanced and stayed 46 for ACT-midpoint.

These differences correct the measuring instrument. They must not be counted as model regressions caused by new training.

## Provenance and Limits

Replay artifacts retain source/scenario hashes and inherit the original generation cost/latency. That latency cannot measure replay performance or a new model-speed gain.

Old split leakage and unseeded LoRA initialization were not repaired by replay. Source filenames embedded in JSON can predate renaming; use FILE_RENAMES.json to locate their current paths.

The linked parser README documents rejected-output counts and the corrected synthetic split. Later routing metrics may be derived from saved decisions without changing these historical JSON files.

## Supporting Files

- [README.md](../../../results/phase2/parser-integrity-v1/README.md)
- [FILE_RENAMES.json](../../../results/phase2/FILE_RENAMES.json)
- [SCORING.md](../../../docs/SCORING.md)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
