# DPO From the Two-to-One SFT Model

[Experiment index](../README.md)

## Question and Context

Could the more cautious 2:1 SFT checkpoint resist the ACT bias seen after DPO on the original reference?

## Controlled Change

Change the initial policy and matching frozen reference to the 2:1 SFT adapter. Keep the exact original 705 preference pairs, prompt, DPO settings and evaluator unchanged.

- Start policy and frozen reference from the specified SFT adapter, not bare Qwen.
- One DPO epoch, sigmoid preference loss, beta 0.1, learning rate 5e-6, constant scheduler.
- Batch one, accumulation 16, context 1,024, seed 1; inherited LoRA rank/targets.
- TRL 1.14.0 ordinary PyTorch full-logits path, with reference log probabilities precomputed.
- Chosen/rejected answers share the same prompt; no JSON repair or relaxed evaluator.
- Rule-derived training preferences plus owner review, not LLM-judge labels.
- No DPO validation preference split was constructed; evaluation used the existing development suite.

"2:1 DPO" means DPO started from a 2:1-trained SFT model. The preference dataset itself was not rebalanced to 2:1.

## Provenance

The config separates preference provenance (reference_config) from policy initialization (initial_sft_config). Starting config and adapter hashes are pinned. Actual artifact retained 705 pairs / 400 prompts, longest sequence 633, zero truncation.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [DPO / 2:1 start](../../../results/phase2/qwen-lora/dpo/ratio-2to1-sft2-dpo1-combined-150.json) | 150 | 31/150 (0.207) | 0.473 | 0.676 | 0.125 | 0.172 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| DPO / 2:1 start (150 cases) | 0.453 | 0.047 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: DPO / 2:1 start (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 40 | 32 | 3 | 6 |
| CLARIFY | 21 | 18 | 1 | 0 |
| REFUSE | 13 | 5 | 10 | 1 |

### Result Provenance

- [DPO / 2:1 start](../../../results/phase2/qwen-lora/dpo/ratio-2to1-sft2-dpo1-combined-150.json): 2026-10-09 UTC, 150 cases.

## Interpretation

Against 2:1 SFT, task successes rose from 29 to 31/150, while macro-F1 fell from 0.477 to 0.453. Premature ACT decisions increased from 22 to 34, and correct refusal routing fell from 19 to ten.

The starting checkpoint affected the outcome, but DPO again shifted toward action. This was not an overall improvement over the original two-epoch SFT reference.

## Limits

Changing initialization also changes the frozen DPO reference distribution, necessarily. It is not a test of changing only the policy weights while holding reference probabilities fixed.

The same preference-coverage gaps remained. No additional preference examples were introduced in this experiment.

## Supporting Files

- [qwen2_5_1_5b_2to1_epochs2_v1.json](../../../configs/dpo/qwen2_5_1_5b_2to1_epochs2_v1.json)
- [dpo-run.json](../../../models/qwen2_5_1_5b_2to1_epochs2_dpo_v1/dpo-run.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
