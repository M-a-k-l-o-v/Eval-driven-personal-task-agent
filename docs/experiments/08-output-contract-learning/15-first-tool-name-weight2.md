# First Tool-Name Loss Weight: Two Instead of Three

**Status:** Completed; unsuccessful against the unweighted reference.
Date: 2026-10-15.

## Question

Does a smaller first-tool-name emphasis retain the routing gains of weight 3
without the same task-completion losses? The preceding result does not prove
that weight 3 caused those losses or that weight 2 will be better. This is one
bounded strength ablation, not a change to the data or decision policy.

See [the weight-3 experiment](13-first-tool-name-weight3.md) and its
[paired failure review](14-tool-weight3-paired-failure-review.md).

## Controlled Change

Use the same frozen version-14 name-first dataset, Qwen2.5-1.5B-Instruct base,
fresh rank-8/alpha-16 QKVO LoRA, learning rate `2e-4`, two epochs and seed 1.
Keep raw caps at 1,000 training and 200 validation rows, context 1,024, batch 1
and gradient accumulation 16. Do not resume either existing adapter.

Only `training.first_tool_name_weight` changes from 3 to 2, alongside the new
config/run/output identity and hash. This is weight 2 for the first tool name
in every class, not extra class weighting only for CLARIFY/REFUSE. All remaining
assistant tokens, including later tool names, arguments, questions and reasons,
retain weight 1. Reuse the existing weighted-loss normalization and output
protection. Validation loss remains unweighted.

Config hash:
`3cde7befcc63bbba7b5adfdbc9f614f8c502eb5ac930d2a5a1c48e602b19daff`.
Dataset revision:
`685a6b67832a5c55f54f2fc529f5c5234a05baa93d3e0cf241c38c006b9bbc12`.

## Training Command

From `project-baymax`:

```powershell
$config = "configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first_tool_weight2.json"

.venv\Scripts\python scripts\train_sft_lora.py `
  --config $config `
  --local-low-vram `
  --max-train-rows 1000 `
  --max-eval-rows 200 `
  --max-sequence-length 1024
```

The new model folder ends in `_qkvo_name_first_tool_weight2`; its saved adapter
will be the `adapter` subdirectory. The trainer refuses a populated output
directory. Historical configs, checkpoints, datasets and results stay untouched.

## Evaluation Setup

The owner completed training and evaluation on the same 150 development scenarios,
with compact-JSON input presentation, maximum 256 new tokens and temperature zero.
The saved inference configs differ only in adapter path. The repeatedly inspected
suite is not an untouched final test. Scorer and benchmark targets were not changed.

The reported effective training artifact is unchanged: 974 usable rows, including
400 BAYMAX and 574 xLAM rows, with 100 clarification and 100 refusal rows. Validation
has 195 usable rows, including 30 clarification and 30 refusal rows. Both artifacts
have zero exact duplicates. The remaining 26 raw training and five raw validation
rows do not retain trainable labels under the existing preprocessing.

## Saved Result Sources

| Weight | Result | SHA256 |
| --- | --- | --- |
| 1 | [Unweighted name-first compact](../../../results/phase2/json-spacing/name-first-compact-combined-150.json) | `6ace952a70b3695737d373e80edb09b2b21505ac9a6c879a4ebb0a80b1bbf6aa` |
| 2 | [First-name weight 2 compact](../../../results/phase2/json-spacing/name-first-tool-weight2-compact-combined-150.json) | `de4f9cb571a36c6a8ac6b4ba20112479135a24507ccd79b6c5c16a7cb8c0a99a` |
| 3 | [First-name weight 3 compact](../../../results/phase2/json-spacing/name-first-tool-weight3-compact-combined-150.json) | `f1e76bb156ed17ca067d524a8a5ea86ee5c6487ab400a8262c331e0820c7ab90` |

## Measured Results

Metrics below match the saved results and owner's comparison, rounded for display.
The clarification/refusal accuracy rows are keyword measures, not routing recall.

| Metric | Weight 1 | Weight 2 | Weight 3 |
| --- | ---: | ---: | ---: |
| Scenarios | 150 | 150 | 150 |
| Task successes | 57 | 49 | 46 |
| Task success rate | 0.380 | 0.327 | 0.307 |
| Routing macro-F1 | 0.612 | 0.604 | 0.626 |
| Invalid outputs | 1 | 2 | 3 |
| Invalid-output rate | 0.007 | 0.013 | 0.020 |
| Average tool-call accuracy | 0.673 | 0.677 | 0.683 |
| Average argument accuracy | 0.835 | 0.800 | 0.787 |
| Average clarification accuracy | 0.175 | 0.237 | 0.237 |
| Average refusal accuracy | 0.431 | 0.328 | 0.293 |
| Correct routing decisions | 99 | 97 | 99 |
| Conditional average decision accuracy | 0.664 | 0.655 | 0.673 |
| Decision coverage | 0.993 | 0.987 | 0.980 |
| Average tool-name hallucination rate | 0.013 | 0.013 | 0.013 |
| Average latency, ms | 1863 | 1933 | 1844 |
| Reported API cost, USD | 0.0000 | 0.0000 | 0.0000 |

## Routing Matrices

Rows are expected decisions; columns are model decisions. Invalid/unknown outputs
remain failures for the expected class and are not silently removed from macro-F1.

### Weight 1: Reference

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 66 | 10 | 4 | 1 |
| CLARIFY | 26 | 10 | 4 | 0 |
| REFUSE | 5 | 1 | 23 | 0 |

### Weight 2: This Experiment

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 64 | 11 | 4 | 2 |
| CLARIFY | 21 | 13 | 6 | 0 |
| REFUSE | 7 | 2 | 20 | 0 |

### Weight 3: Previous Experiment

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 64 | 10 | 4 | 3 |
| CLARIFY | 20 | 15 | 5 | 0 |
| REFUSE | 6 | 3 | 20 | 0 |

## Conclusion

Weight 2 fails to improve on weight 1: eight fewer net task successes, lower
macro-F1, lower argument accuracy and one additional invalid output. Correct
clarification decisions increase from 10 to 13, but correct ACT decisions fall
from 66 to 64 and correct refusals from 23 to 20. Total correct routing falls
from 99 to 97. The higher clarification keyword score is not an overall win.

Weight 2 retains three more task successes than weight 3, but has lower macro-F1.
It therefore does not establish the hoped-for better balance. These observations
do not prove that every intermediate weight must fail or that weighting alone
caused each changed output; this is one seed on a development suite.

**Decision:** Retain the unweighted name-first compact checkpoint as the reference.
Do not promote either weighted candidate. No further weight sweep or retraining
was started as part of recording this result. Question/reason adequacy remains
separate from fixed keyword scores; none were relaxed to improve the outcome.

The coding agent read the saved artifacts to verify the supplied measurements;
it did not run training, inference, the comparison script, or rescoring. No raw
results were changed. Local experiment Markdown is not staged or committed.
