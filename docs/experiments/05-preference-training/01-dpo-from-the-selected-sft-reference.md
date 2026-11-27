# DPO From the Selected Two-Epoch SFT Reference

[Experiment index](../README.md)

## Question and Context

Could preferences between correct and incorrect decisions sharpen routing after SFT, without changing the model or evaluator?

DPO trains relative preference for a chosen answer over a rejected answer to the same prompt. It is not an LLM judge reviewing benchmark results.

## Training Recipe

- Start policy and frozen reference from the specified SFT adapter, not bare Qwen.
- One DPO epoch, sigmoid preference loss, beta 0.1, learning rate 5e-6, constant scheduler.
- Batch one, accumulation 16, context 1,024, seed 1; inherited LoRA rank/targets.
- TRL 1.14.0 ordinary PyTorch full-logits path, with reference log probabilities precomputed.
- Chosen/rejected answers share the same prompt; no JSON repair or relaxed evaluator.
- Rule-derived training preferences plus owner review, not LLM-judge labels.
- No DPO validation preference split was constructed; evaluation used the existing development suite.

## Preference Artifact

400 unique BAYMAX training anchors: 200 ACT, 100 CLARIFY, 100 REFUSE. xLAM rows were not used as preference anchors. There were 705 pairs because an anchor can have multiple distinct rejections.

| Chosen over rejected | Pairs |
| --- | ---: |
| ACT > CLARIFY | 200 |
| ACT > REFUSE | 200 |
| CLARIFY > ACT | 50 |
| CLARIFY > REFUSE | 100 |
| REFUSE > ACT | 55 |
| REFUSE > CLARIFY | 100 |

50 clarification and 45 refusal anchors lacked a suitable productive rejection. Missing directions were reported rather than filled with arbitrary tool calls.

Actual artifact: 23,008 chosen completion tokens, 20,981 rejected tokens, longest sequence 633, zero truncated pairs. The Windows smoke established runnable training, not improved capability.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [DPO / original reference](../../../results/phase2/qwen-lora/dpo/anchored-sft2-dpo1-combined-150.json) | 150 | 53/150 (0.353) | 0.533 | 0.856 | 0.013 | 0.052 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| DPO / original reference (150 cases) | 0.376 | 0.033 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: DPO / original reference (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 72 | 5 | 1 | 3 |
| CLARIFY | 35 | 4 | 1 | 0 |
| REFUSE | 21 | 1 | 5 | 2 |

### Result Provenance

- [DPO / original reference](../../../results/phase2/qwen-lora/dpo/anchored-sft2-dpo1-combined-150.json): 2026-10-09 UTC, 150 cases.

## Interpretation

Task successes increased from SFT's 44 to 53/150, but macro-F1 fell from 0.514 to 0.376. Premature ACT decisions on boundary cases rose from 31 to 56. Correct refusal routing fell from 16 to five, and correct clarification from eleven to four.

This run favored productive action and argument accuracy. It was a task-completion gain paired with a substantial boundary regression, not a general improvement.

## Limits

Preference directions were incomplete and asymmetric. That is a plausible contributor, not proven causation. Pair counts are not unique-prompt counts. Results are single-seed and repeatedly evaluated.

The initial Windows failure was a TRL/Triton compatibility problem, resolved with the ordinary PyTorch path; it did not change preference labels or the DPO objective.

## Supporting Files

- [qwen2_5_1_5b_anchored_epochs2_v1.json](../../../configs/dpo/qwen2_5_1_5b_anchored_epochs2_v1.json)
- [dpo-run.json](../../../models/qwen2_5_1_5b_anchored_epochs2_dpo_v1/dpo-run.json)
- [DPO_EXPERIMENT.md](../../../docs/DPO_EXPERIMENT.md)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
