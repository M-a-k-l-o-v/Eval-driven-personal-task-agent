# Two-to-One Productive-to-Boundary SFT Ratio

[Experiment index](../README.md)

## Question and Context

Would a more boundary-heavy mixture help, following the owner's discussion of a 2:1 productive/non-productive ratio?

This is an empirical BAYMAX ratio test. It is not evidence that a paper's ratio transfers unchanged to this model and dataset.

## Change

Keep the original anchored two-epoch BAYMAX pool, add 64 unique clarifications and 70 unique refusals, and replace 134 capped xLAM positions. Preserve validation/test bytes and other training settings.

Raw training cap: 666 ACT / 164 CLARIFY / 170 REFUSE, rather than the reference's 800 ACT / 100 CLARIFY / 100 REFUSE. ACT includes both xLAM and productive BAYMAX examples.

## Actual Artifact

| Class | Usable rows | Trainable tokens |
| --- | ---: | ---: |
| Direct productive call | 372 | 11,109 |
| Multi-tool sequence | 272 | 14,062 |
| CLARIFY | 164 | 4,593 |
| REFUSE | 170 | 5,022 |

978 usable rows: BAYMAX 534, xLAM 444; no exact duplicates. The effective ACT:boundary row ratio is 644:334, approximately 1.93:1. Boundary tokens are 27.6% of target tokens, not one-third.

The initially proposed 167/167 allocation failed unique-input coverage. The corrected quota preserved the boundary total without repeating examples or borrowing held-out data.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [2:1 SFT / two epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-2to1-anchored-train1k-val200-epochs2-combined-150.json) | 150 | 29/150 (0.193) | 0.487 | 0.623 | 0.125 | 0.259 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| 2:1 SFT / two epochs (150 cases) | 0.477 | 0.053 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: 2:1 SFT / two epochs (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 31 | 38 | 6 | 6 |
| CLARIFY | 14 | 17 | 8 | 1 |
| REFUSE | 8 | 1 | 19 | 1 |

### Result Provenance

- [2:1 SFT / two epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-2to1-anchored-train1k-val200-epochs2-combined-150.json): 2026-10-09 UTC, 150 cases.

## Interpretation

Success fell from 44/150 to 29/150 and macro-F1 from 0.514 to 0.477. Premature actions fell from 31 to 22, but correct ACT routing fell from 57/81 to 31/81; ACT-to-CLARIFY errors rose from 21 to 38.

The mixture improved some boundary recall and wording scores at a large productive-task cost. It was not a balanced win.

## Limits

Changing the ratio also changes examples, source proportions, token exposure and optimizer steps. This does not isolate class weights alone. The checkpoint remains useful as an alternative starting point for the next DPO comparison.

## Supporting Files

- [qwen2_5_1_5b_lora_target_compatible_v14_2to1_anchored_val200_epochs2.json](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_2to1_anchored_val200_epochs2.json)
- [prepare_sft_2to1.py](../../../scripts/prepare_sft_2to1.py)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
