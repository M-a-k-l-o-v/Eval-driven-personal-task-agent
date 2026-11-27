# Target-Contract Cleaning With Anchored External Rows

[Experiment index](../README.md)

## Question and Context

Did the earlier cleaned model change because of better targets, or because rebuilding the mixture selected different external examples?

## Change

Anchor selection to original ACT200. Preserve all 400 capped BAYMAX rows and retain 582 of the original 600 xLAM rows; replace only the 18 excluded xLAM rows with compatible same-split rows.

Validation retained 90 BAYMAX and 107 xLAM rows, replacing three xLAM examples. Train/validation/test membership remains protected. The raw recipe stays 200 BAYMAX ACT / 100 CLARIFY / 100 REFUSE plus 600 xLAM.

## Actual Artifact and Recipe

One fresh SFT epoch, base Qwen2.5-1.5B-Instruct, seed 1, LoRA rank 8/alpha16 on q_proj and v_proj, dropout 0.05. Learning rate 2e-4, cosine scheduler, warmup fraction 0.03.

Owner low-VRAM overrides: context 1,024, batch one, accumulation 16, raw train/eval caps 1,000/200.

Usable train: 974 rows, direct 435, multi 339, refusal 100, clarification 100; BAYMAX 400, xLAM 574, no exact duplicates. Validation retained 195.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Anchored / one epoch](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-combined-150.json) | 150 | 39/150 (0.260) | 0.500 | 0.708 | 0.050 | 0.155 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Anchored / one epoch (150 cases) | 0.429 | 0.080 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: Anchored / one epoch (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 50 | 22 | 3 | 6 |
| CLARIFY | 24 | 7 | 5 | 4 |
| REFUSE | 12 | 2 | 12 | 3 |

### Result Provenance

- [Anchored / one epoch](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-combined-150.json): 2026-10-06 UTC, 150 cases.

## Interpretation

Success was 39/150, macro-F1 0.429 and invalid outputs 12/150. Compared with resampling's 47 successes but 0.354 macro-F1, anchoring recovered more balanced routing but not its task total.

This became the data reference for the epoch-length trials.

## Limits

Replacing 18 examples also changes token lengths and usable counts. This is a much tighter comparison, not perfectly identical exposure. Single-seed measurements do not establish a statistically reliable causal effect.

## Supporting Files

- [qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200.json](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
