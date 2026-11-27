# Anchored Data at Four Epochs

[Experiment index](../README.md)

## Question and Context

Would substantially longer training improve the two-epoch reference, or simply strengthen the wrong decision patterns?

## Controlled Change

Train a fresh adapter for four total epochs on the same anchored data. This was not continuing from the saved two-epoch adapter.

- Fresh base Qwen/Qwen2.5-1.5B-Instruct plus fresh LoRA; not a resumed adapter.
- LoRA rank 8, alpha 16, q_proj/v_proj, dropout 0.05, bias none.
- Learning rate 2e-4, cosine schedule, warmup fraction 0.03, seed 1.
- Owner overrides: context 1,024, train/eval batch one, accumulation 16, raw caps 1,000/200.
- Assistant-only SFT token cross-entropy; prompt/padding tokens do not provide target loss.
- Recorded training/inference system prompt hash a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468.

The usable training/validation counts stayed 974/195, with the same broad class counts and no exact duplicates. The learning-rate schedule now spans four epochs.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Anchored / four epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-epochs4-combined-150.json) | 150 | 37/150 (0.247) | 0.440 | 0.628 | 0.263 | 0.103 |
| [Anchored / four epochs](../../../results/phase2/template-shift/target-compatible-anchored-epochs4-template-shift-30.json) | 30 | 4/30 (0.133) | 0.400 | 0.528 | 0.300 | 0.050 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Anchored / four epochs (150 cases) | 0.372 | 0.087 |
| Anchored / four epochs (30 cases) | 0.348 | 0.200 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: Anchored / four epochs (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 39 | 36 | 1 | 5 |
| CLARIFY | 14 | 19 | 4 | 3 |
| REFUSE | 17 | 3 | 4 | 5 |

### Decision Matrix: Anchored / four epochs (30 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 2 | 7 | 1 | 0 |
| CLARIFY | 1 | 8 | 1 | 0 |
| REFUSE | 3 | 0 | 1 | 6 |

### Result Provenance

- [Anchored / four epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-epochs4-combined-150.json): 2026-10-07 UTC, 150 cases.
- [Anchored / four epochs](../../../results/phase2/template-shift/target-compatible-anchored-epochs4-template-shift-30.json): 2026-10-07 UTC, 30 cases.

## Interpretation

Compared with two epochs, combined task successes fell from 44 to 37, macro-F1 from 0.514 to 0.372, and invalid outputs increased from two to thirteen.

Correct CLARIFY routing rose from 11/40 to 19/40, but correct ACT fell from 57/81 to 39/81 and correct REFUSE from 16/29 to 4/29. Higher clarification wording scores therefore concealed a harmful routing tradeoff.

Shift success also fell, from five to four, and invalid outputs rose from two to six. Its macro-F1 was nearly unchanged, but the class-specific behavior changed substantially.

## Why Lower Loss Was Not Enough

Validation token loss fell to 0.2364 from the one-epoch run's 0.3107. That objective rewards the supplied answer tokens; it does not directly optimize routing macro-F1, execution success or semantic explanations.

The loss reduction does not prove useful generalization, nor does the regression prove memorization. It does show that more epochs are not a reliable fix under this recipe.

## Decision and Limits

Keep two epochs as the reference. The saved audit separates wrong decisions, wording-score mismatch, argument errors and invalid output without repairing results.

This was one seed, and changing epochs changes the schedule horizon. Do not generalize it to every learning rate, dataset or local model.

## Supporting Files

- [qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs4.json](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs4.json)
- [anchored-epochs2-vs4-failure-audit.md](../../../results/phase2/analysis/anchored-epochs2-vs4-failure-audit.md)
- [anchored-epochs2-vs4-failure-audit.json](../../../results/phase2/analysis/anchored-epochs2-vs4-failure-audit.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
