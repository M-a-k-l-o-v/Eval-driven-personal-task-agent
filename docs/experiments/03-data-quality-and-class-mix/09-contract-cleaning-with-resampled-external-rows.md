# Target-Contract Cleaning With Resampled External Rows

[Experiment index](../README.md)

## Question and Context

The new strict training gate found target/schema failures in xLAM. Would excluding incompatible examples help?

## Change

Build a target-compatible external source: 57,611 accepted rows after 618 credential exclusions and 1,771 additional target/credential exclusions, for 2,389 total skipped rows.

Retain the generator14 ACT200 recipe, but rebuild the mixed sampler. Removing rows changes which xLAM examples survive the cap; this is the cleaned-resampled candidate.

## Actual Artifact

Raw cap 1,000; usable train 977: direct 447, multi 330, refusal 100, clarification 100. BAYMAX 400, xLAM 577, zero exact duplicates.

Usable validation 196: direct 72, multi 64, refusal 30, clarification 30.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Cleaned / resampled](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-train1k-val200-combined-150.json) | 150 | 47/150 (0.313) | 0.493 | 0.761 | 0.013 | 0.086 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Cleaned / resampled (150 cases) | 0.354 | 0.073 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: Cleaned / resampled (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 59 | 15 | 2 | 5 |
| CLARIFY | 28 | 3 | 5 | 4 |
| REFUSE | 17 | 2 | 7 | 3 |

### Result Provenance

- [Cleaned / resampled](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-train1k-val200-combined-150.json): 2026-10-06 UTC, 150 cases.

## Interpretation

Success reached 47/150, versus 38/150 for original ACT200. Invalid outputs fell from 20 to 11, but routing macro-F1 fell from 0.396 to 0.354.

The model completed more tasks while handling abstention worse. "Cleaning improved reasoning" would overstate the evidence.

## Limits

A recorded content-hash comparison found only 7/600 shared xLAM rows and 301/400 shared BAYMAX rows versus the original raw cap. Even the unchanged BAYMAX source yielded different selected examples. This does not isolate removal of faulty examples. A full-pool exclusion rate of about three percent can still change a small capped artifact substantially. The anchored comparison was created to address that confound.

## Supporting Files

- [qwen2_5_1_5b_lora_target_compatible_v14_act200_val200.json](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_val200.json)
- [TRAINING_DATA_AUDIT.md](../../../docs/TRAINING_DATA_AUDIT.md)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
