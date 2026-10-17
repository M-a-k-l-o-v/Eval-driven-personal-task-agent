# Corrected Policy With Two Hundred Productive Examples

[Experiment index](../README.md)

## Question and Context

Would the productive-heavy 200/100/100 BAYMAX recipe work better with the corrected policy and credential-filtered external source?

## Change

Use the same source family as the previous boundary-heavy run, but cap BAYMAX at 400: 200 ACT, 100 CLARIFY and 100 REFUSE. Fill 600 raw slots with unique xLAM ACT examples.

Keep a representative 200-row validation cap, the base model, one epoch, seed, prompt and LoRA settings. This is the "original ACT200" control used by subsequent cleaning comparisons.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Corrected ACT200](../../../results/phase2/qwen-lora/baymax-mix/auth-filtered-generator14-act200-train1k-val200-combined-150.json) | 150 | 38/150 (0.253) | 0.473 | 0.675 | 0.062 | 0.121 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Corrected ACT200 (150 cases) | 0.396 | 0.133 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [Corrected ACT200](../../../results/phase2/qwen-lora/baymax-mix/auth-filtered-generator14-act200-train1k-val200-combined-150.json): 2026-10-05 UTC, 150 cases.

## Interpretation

Success increased from 25/150 to 38/150; macro-F1 was 0.396. That recovered productive performance but reduced some boundary wording metrics.

The comparison supports a mixture tradeoff, not a claim that more ACT always helps.

## Limits

It is not the same source as historical ACT-balanced, even though the broad quotas match. The measured 20 invalid outputs also motivated validating training targets separately from changing decision proportions.

## Supporting Files

- [qwen2_5_1_5b_lora_auth_filtered_v14_act200_val200.json](../../../configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_act200_val200.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
