# First Tool-Name Loss Weight: Three Instead of One

**Status:** Owner training and evaluation completed; paired result review completed.
Date: 2026-10-15.

## Hypothesis

The frozen name-first model still makes wrong routing choices on selected seen
inputs. Test whether directly emphasizing its first call name improves routing,
without changing the data mixture, prompt, serialization, or epoch count.

The preceding component diagnostic does not prove this intervention is best.
Question/reason loss is larger, but exact-target loss is not semantic adequacy.
This experiment deliberately targets routing rather than text reproduction.

## Fixed Reference

Use the fresh two-epoch QKVO name-first recipe, not A's older Q/V-only recipe:

- base: `Qwen/Qwen2.5-1.5B-Instruct`, initialized from pretrained weights;
- fresh LoRA: rank 8, alpha 16, Q/K/V/O, dropout 0.05;
- learning rate `2e-4`, two epochs, seed 1, same scheduler and warmup;
- unchanged version-14 name-first split files and order;
- raw caps: 1,000 train and 200 validation;
- laptop overrides: context 1,024, batch 1, gradient accumulation 16.

Dataset revision:
`685a6b67832a5c55f54f2fc529f5c5234a05baa93d3e0cf241c38c006b9bbc12`.
Reference config hash:
`684a76cd39d845b6177bdaee495bb77bcb436406d384baf912f80b8164fb8b8b`.
Weighted config hash:
`7a5c074667c6041c6adc47e890197c5ce07f8a1175e90f4416f65592c027f619`.

## One Objective Change

`training.first_tool_name_weight = 3.0` applies to pure string-content tokens
of the first tool name in every class: ACT, CLARIFY and REFUSE. Other assistant
tokens, including questions, reasons, arguments and later call names, keep weight 1.
Mixed text/syntax tokens also stay at 1, matching the diagnostic's region policy.
Prompt and padding labels stay ignored. No examples are repeated or rewritten.

Use the same shared character-offset mapping as the diagnostic. The training
token IDs, labels, retained rows and truncation behavior are unchanged. Validation
has no extra weights and retains ordinary assistant-token cross-entropy.

Normalize training loss as `sum(weight * token_nll) / sum(weights)` over the
actual accumulation window, including a shorter final window. Three is a relative
weight, not a blanket threefold increase in loss or learning rate. Gradient tests
check equivalence to the corresponding combined weighted batch. The custom path
supports one training device and fails explicitly on incompatible Trainer APIs.

Historical configs default to weight 1 and use the original Trainer. Their config
hashes are unchanged. Nondefault weighting is included in the new config hash.

## Command

From `project-baymax`:

```powershell
$config = "configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first_tool_weight3.json"

.venv\Scripts\python scripts\train_sft_lora.py `
  --config $config `
  --local-low-vram `
  --max-train-rows 1000 `
  --max-eval-rows 200 `
  --max-sequence-length 1024
```

This does not resume the existing adapter. The new config names its own model
folder, ending in `_qkvo_name_first_tool_weight3`, and refuses a populated output
directory before loading ML dependencies. Existing checkpoints remain untouched.

The new model directory receives `weighted-loss-run.json` before training, with
the full config, concrete plan, retained row count, weighted first-name-token
count, effective token mass, and normalization/validation definitions.

## Evaluation Plan

Use exactly the same 150 established development scenarios and compact JSON
inference presentation as the unweighted name-first reference. Preserve the
parser, scorer, decoding settings and all old results. Compare routing macro-F1,
the full decision matrix, task success and invalid-output rate; inspect wording
adequacy separately. The repeatedly inspected suite is not an untouched final test.

Training loss is now weighted while validation loss is not. Do not compare the
two values as if they use the same objective, or compare the weighted training
loss directly against the historical unweighted training loss.

## Measured Outcome

The owner completed training and the compact-JSON evaluation. The reported usable
artifact contains 974 training rows and 195 validation rows, with no exact
duplicates. Training retains 400 BAYMAX rows, including 100 clarification and 100
refusal rows. The saved `weighted-loss-run.json` records 3,275 weighted first-name
tokens and effective training token mass 43,164.

| Metric | Unweighted name-first | First-name weight 3 |
| --- | ---: | ---: |
| Task successes / 150 | 57 | 46 |
| Routing macro-F1 | 0.612 | 0.626 |
| Correct routing decisions / 150 | 99 | 99 |
| Invalid outputs / 150 | 1 | 3 |
| Average argument accuracy | 0.835 | 0.787 |

Weighting increased correct clarification decisions from 10 to 15, but correct
ACT decisions fell from 66 to 64 and correct refusals from 23 to 20. It is not
an overall task-performance win. The conditional decision-accuracy increase
excludes more unknown decisions; it is not an increase in total correct routes.

The [paired failure review](14-tool-weight3-paired-failure-review.md) documents
all 15 lost successes and four gained successes, separating real execution/plan
regressions from wording-score changes. Keep unweighted name-first as this
experiment's reference; do not change evaluation rules to promote either model.

The coding agent inspected saved results and run metadata only. It did not run
training, inference, a benchmark comparison script, or rescoring. Local experiment
Markdown remains uncommitted under the owner's documentation restriction.
