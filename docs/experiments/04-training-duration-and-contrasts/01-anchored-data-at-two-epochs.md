# Anchored Data at Two Epochs: Selected Reference

[Experiment index](../README.md)

## Question and Context

Was one pass over the cleaned anchored mixture insufficient to learn its targets?

## Controlled Change

Increase fresh SFT from one to two total epochs. Keep the selected data, prompt, seed, raw caps and LoRA settings fixed. A longer run also stretches the cosine schedule and warmup in steps, so this is not an extra pass at the exact previous per-step learning rates.

## Actual Training Recipe

- Fresh base Qwen/Qwen2.5-1.5B-Instruct plus fresh LoRA; not a resumed adapter.
- LoRA rank 8, alpha 16, q_proj/v_proj, dropout 0.05, bias none.
- Learning rate 2e-4, cosine schedule, warmup fraction 0.03, seed 1.
- Owner overrides: context 1,024, train/eval batch one, accumulation 16, raw caps 1,000/200.
- Assistant-only SFT token cross-entropy; prompt/padding tokens do not provide target loss.
- Recorded training/inference system prompt hash a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468.

The same usable artifact retained 974 train rows: 435 direct, 339 multi-tool, 100 clarification and 100 refusal. BAYMAX contributed 400 and xLAM 574. Validation retained 195. There were zero exact duplicates.

The trainer completed 122 optimizer steps.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Anchored / two epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-epochs2-combined-150.json) | 150 | 44/150 (0.293) | 0.600 | 0.771 | 0.075 | 0.172 |
| [Anchored / two epochs](../../../results/phase2/template-shift/target-compatible-anchored-epochs2-template-shift-30.json) | 30 | 5/30 (0.167) | 0.433 | 0.694 | 0.250 | 0.250 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Anchored / two epochs (150 cases) | 0.514 | 0.013 |
| Anchored / two epochs (30 cases) | 0.345 | 0.067 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: Anchored / two epochs (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 57 | 21 | 3 | 0 |
| CLARIFY | 20 | 11 | 9 | 0 |
| REFUSE | 11 | 0 | 16 | 2 |

### Decision Matrix: Anchored / two epochs (30 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 3 | 6 | 1 | 0 |
| CLARIFY | 5 | 2 | 3 | 0 |
| REFUSE | 3 | 0 | 5 | 2 |

### Result Provenance

- [Anchored / two epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-epochs2-combined-150.json): 2026-10-06 UTC, 150 cases.
- [Anchored / two epochs](../../../results/phase2/template-shift/target-compatible-anchored-epochs2-template-shift-30.json): 2026-10-07 UTC, 30 cases.

## Interpretation

Combined success increased from 39 to 44/150, macro-F1 from 0.429 to 0.514, and invalid outputs fell from 12 to two. Premature productive decisions on boundary requests fell from 36 to 31.

This is the selected development reference because it balances routing and validity better than the later candidates, not because it is a solved assistant. Shift success was only 5/30, with eight premature ACT decisions on the twenty boundary cases.

## Reproducibility

The linked frozen reference records data revision, config/prompt hashes, capped raw-input hashes, adapter identity and both evaluations. "Frozen" means identities recorded for drift detection, not permission-locked files.

Dataset revision: `329fab4d8f1604a4085d3ec50e05881e7c069717bb1428db7e592743bb07f03c`.

Canonical training config hash: `ce9efa2db24a065dbdb4eb482c07a29a56bfbcbdce32495418eb54b92d985c40`.

## Limits

Exact base-model revision was not recorded. Some inference settings come from owner commands rather than embedded result metadata. Results are single-seed, repeatedly inspected development evidence. An earlier bounded audit found near-duplicate training/evaluation wording; the benchmark is not an untouched final test.

## Supporting Files

- [qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json)
- [anchored-epochs2-reference.json](../../../results/phase2/references/anchored-epochs2-reference.json)
- [anchored-epochs2-vs4-failure-audit.md](../../../results/phase2/analysis/anchored-epochs2-vs4-failure-audit.md)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
