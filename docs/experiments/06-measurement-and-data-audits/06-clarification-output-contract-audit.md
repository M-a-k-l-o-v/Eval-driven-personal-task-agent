# Clarification Output Contract: Target Audit and Failure Diagnostics

[Experiment index](../README.md)

## Question

Experiment C sometimes calls `request_clarification` with `reason`, although the
shared contract requires `question`. Is this incorrect training supervision or
an invalid model output? Can we distinguish it without repairing benchmark answers?

## Training Targets Checked

The audit read the same capped raw prefixes selected by C, using the existing
`load_sft_jsonl` and `validate_training_targets` functions. No model or tokenizer
was loaded and no training data was changed.

| Split | Raw targets checked | Clarification targets | Distinct question strings | Target-contract errors |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,000 | 100 | 65 | 0 |
| Validation | 200 | 30 | 22 | 0 |

All selected clarification targets use a nonempty `question` field, not `reason`.
Supplied meta-tool schemas match the shared definitions. Distinct target wording
was inspected: it requests a time, duration, recipient, text, task/event identity,
conflict resolution or confirmation. Examples include "How many minutes should
the consultation last?" and "Please confirm you want me to send this resignation
message." A confirmation request need not end with a question mark.

Repeated question strings do not imply duplicate complete training rows. This
check covers raw capped targets, not newly measured post-tokenization artifacts,
and does not prove semantic correctness against every example's entire world state.
No automatic semantic question-quality grader was added.

## Recorded C Output Failures

| Raw output diagnostic | Count |
| --- | ---: |
| Attempted clarification with invalid arguments | 10 |
| Other malformed calls | 5 |
| Unknown decision from an empty array | 1 |

The ten clarification calls supplied `reason` instead of `question`. Six were
on expected CLARIFY cases, two on ACT cases and two on REFUSE cases. Thus the
diagnostic identifies an attempted clarification, not necessarily correct routing.
The remaining malformed calls comprise four outputs missing tool names and one
incomplete JSON output. The empty array is valid JSON but expresses no decision.
Official invalid-output count remains 15 and the matrix's invalid/unknown count
remains 16. Original scores, expected behavior, parser rules and result JSON are unchanged.

## Diagnostic Change

The behavior analyzer previously classified a matching meta-tool name as
`correct_decision_wrong_wording`, even if its required arguments were invalid.
It now consults the existing strict parser first. Invalid clarification arguments
are labeled `attempted_clarification_invalid_arguments`; the analogous refusal
label is `attempted_refusal_invalid_arguments`. Mixed meta/productive calls remain
malformed, not accepted boundary decisions.

The quick failure-summary command can print a separate diagnostic table:

```powershell
.venv\Scripts\python scripts\summarize_eval_failures.py `
  results\phase2\qwen-lora\baymax-mix\target-compatible-generator14-act200-anchored-lr1e-4-train1k-val200-epochs2-combined-150.json `
  --output-diagnostics
```

The default failure-reason summary is unchanged. Diagnostics use the current
parser only to inspect saved generations. If a recorded rejection/unknown is
not reproduced by that parser, it is flagged separately rather than rescored.

## Decision and Limits

There is no demonstrated `reason`/`question` defect in the selected targets to fix.
Do not rename fields in evaluation outputs or add fixture-specific training examples.
Any future output-contract learning experiment should be independently authored,
keep class quantities fixed and separate format validity from routing correctness.
No new training or prompt change was made during this audit.

## Sources

- [C result JSON](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-lr1e-4-train1k-val200-epochs2-combined-150.json)
- [C training config and dataset identity](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_lr1e-4.json)
- [Shared meta-tool definitions](../../../src/baymax/core/meta_tools.py)
- [Target validation](../../../src/baymax/models/sft_validation.py)
- [Read-only diagnostic classifier](../../../src/baymax/eval/output_diagnostics.py)
- [Failure-summary command](../../../scripts/summarize_eval_failures.py)

Audit recorded on 12 October 2026. It is a measurement investigation, not a new model run.
