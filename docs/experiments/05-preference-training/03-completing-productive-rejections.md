# Completing Missing Productive Rejections for Boundary Decisions

[Experiment index](../README.md)

## Question and Context

Did the first preference set underteach "clarify instead of act" and "refuse instead of act" because those rejected alternatives were missing for many anchors?

## Change

Start fresh from the original anchored two-epoch SFT adapter, not the previous DPO adapter or the 2:1 SFT model.

Add 95 explicit incorrect ACT answers to existing training prompts: 50 clarification and 45 refusal anchors. Negatives deliberately guess missing details, skip confirmation, call a known unavailable operation, or substitute an inappropriate productive action.

Training-only counterparts take priority. Authoring explanations stay in the review sidecar, not the model input. No evaluation fixture supplied a target.

## Preference Artifact

| Chosen over rejected | Before | After |
| --- | ---: | ---: |
| ACT > CLARIFY | 200 | 200 |
| ACT > REFUSE | 200 | 200 |
| CLARIFY > ACT | 50 | 100 |
| CLARIFY > REFUSE | 100 | 100 |
| REFUSE > ACT | 55 | 100 |
| REFUSE > CLARIFY | 100 | 100 |

800 pairs, still 400 unique prompts and the same 200/100/100 unique anchor counts. Actual artifact: 25,854 chosen and 24,541 rejected completion tokens, longest sequence 633, no truncated pairs.

- Start policy and frozen reference from the specified SFT adapter, not bare Qwen.
- One DPO epoch, sigmoid preference loss, beta 0.1, learning rate 5e-6, constant scheduler.
- Batch one, accumulation 16, context 1,024, seed 1; inherited LoRA rank/targets.
- TRL 1.14.0 ordinary PyTorch full-logits path, with reference log probabilities precomputed.
- Chosen/rejected answers share the same prompt; no JSON repair or relaxed evaluator.
- Rule-derived training preferences plus owner review, not LLM-judge labels.
- No DPO validation preference split was constructed; evaluation used the existing development suite.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [DPO / completed ACT rejections](../../../results/phase2/qwen-lora/dpo/anchored-sft2-dpo1-act-rejections-v2-combined-150.json) | 150 | 46/150 (0.307) | 0.533 | 0.793 | 0.050 | 0.155 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| DPO / completed ACT rejections (150 cases) | 0.482 | 0.047 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: DPO / completed ACT rejections (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 60 | 14 | 1 | 6 |
| CLARIFY | 30 | 9 | 1 | 0 |
| REFUSE | 15 | 2 | 11 | 1 |

### Result Provenance

- [DPO / completed ACT rejections](../../../results/phase2/qwen-lora/dpo/anchored-sft2-dpo1-act-rejections-v2-combined-150.json): 2026-10-09 UTC, 150 cases.

## Interpretation

Task successes were 46/150, macro-F1 0.482 and invalid outputs seven. Compared with first DPO's 53 successes / 0.376 macro-F1, this recovered some routing balance at a task-completion cost.

Premature ACT decisions fell from 56 to 45, but the original SFT reference had only 31. Original SFT still had higher macro-F1 (0.514) and fewer invalid outputs (two).

The gap repair helped relative to the first DPO candidate, but did not beat the original SFT on balanced routing.

## Limits

Adding 95 pairs increases optimizer exposure at the same epoch count: this is not a step-matched ablation. Coverage is complete but directions are not equally weighted: ACT-over-boundary has 400 pairs, boundary-over-ACT 200.

No claim follows that DPO cannot work, that Qwen is inherently unsuitable, or that longer DPO would fix it. Those remain untested hypotheses.

## Supporting Files

- [qwen2_5_1_5b_anchored_epochs2_act_rejections_v2.json](../../../configs/dpo/qwen2_5_1_5b_anchored_epochs2_act_rejections_v2.json)
- [dpo-run.json](../../../models/qwen2_5_1_5b_anchored_epochs2_dpo_act_rejections_v2/dpo-run.json)
- [dpo_act_rejections.py](../../../src/baymax/models/dpo_act_rejections.py)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
