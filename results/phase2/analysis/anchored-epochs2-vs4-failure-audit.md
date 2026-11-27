# Anchored Two- vs Four-Epoch Failure Audit

Captured on 2026-10-07 from four saved benchmark files. No new inference,
training, rescoring, parser changes or fixture changes were performed.

## Selected Reference

Use the **anchored two-epoch** adapter as the development reference.
The [reference record](../references/anchored-epochs2-reference.json) pins SHA256
identities for the config, adapter, checkpoint, dataset metadata, capped raw inputs
and both reference result files. It detects drift; it does not permission-lock
files or copy model weights. Do not overwrite these artifacts.

The original run did not record an exact base-model revision. Tokenized artifacts
were not rebuilt: usable-row counts come from the owner's training reports.
Generation flags are recorded from owner commands, not independently recoverable
from the saved benchmark metadata. These limitations remain explicit.

Both runs started from base Qwen with fresh LoRA adapters. Data revision, prompt,
seed, caps and LoRA settings were unchanged. Longer epochs also extend the cosine
schedule and warmup duration; this is not an isolated extra pass at identical
step-by-step learning rates.

## Measured Outcomes

| Suite | Epochs | Task Success | Routing Macro-F1 | Invalid Outputs |
| --- | ---: | ---: | ---: | ---: |
| Combined 150 | 2 | 44/150 | 0.514 | 2/150 |
| Combined 150 | 4 | 37/150 | 0.372 | 13/150 |
| Template Shift 30 | 2 | 5/30 | 0.345 | 2/30 |
| Template Shift 30 | 4 | 4/30 | 0.348 | 6/30 |

The machine-readable [case audit](anchored-epochs2-vs4-failure-audit.json) includes
all 270 failed records across 360 scenario runs, with source-file hashes,
zero-based result indices, scenario IDs and raw-output hashes. Original result
files remain the source of truth.

## Primary Failure Categories

Each failed record has one primary category. Invalid output takes precedence,
followed by unknown decision, wrong decision, boundary wording-score mismatch,
arguments, tools, execution and other failures. A primary category does not mean
no other defect exists in that output.

| Primary Category | Combined 2 | Combined 4 | Shift 2 | Shift 4 |
| --- | ---: | ---: | ---: | ---: |
| Wrong decision | 64 | 75 | 18 | 13 |
| Correct decision, wording-score mismatch | 27 | 17 | 4 | 6 |
| Invalid output | 2 | 13 | 2 | 6 |
| Argument failure | 10 | 5 | 1 | 1 |
| Tool selection or sequence failure | 3 | 3 | 0 | 0 |
| Total failed | 106 | 113 | 25 | 26 |

Do not interpret fewer primary argument failures as independently improved
arguments: records can move into earlier categories.

## Decision Errors

On Combined 150, ACT cases incorrectly routed to CLARIFY increased from **21 to
36**. Correct CLARIFY decisions increased from **11 to 19**, but correct REFUSE
decisions fell from **16 to 4**. Productive calls on CLARIFY/REFUSE cases remained
**31** in each run.

On Template Shift 30, correct CLARIFY decisions rose from **2 to 8**, but correct
REFUSE decisions fell from **5 to 1** and correct ACT decisions from **3 to 2**.
Four epochs reduced wrong valid decisions there while increasing rejected outputs.
This is a routing tradeoff, not robust policy improvement.

Diagnostic flags in the JSON report are non-exclusive. Mixed calls, execution
failures and routing errors can overlap; do not sum those flags as a failure total.

## Output-Contract Errors

All 23 rejected outputs have identified contract causes. Causes can overlap:

- Meta-tool values outside `arguments`, such as a top-level `reason`.
- Missing or incorrect meta-tool argument keys: `question` vs `reason`.
- Productive calls mixed with a meta-tool, repeated meta-tools, or mixed decisions.
- Calls containing arguments but no tool name.
- An undotted productive tool name, rejected by `AgentToolCall` validation.
- Malformed JSON or trailing content.

The existing strict parser was used. Inspecting an intact JSON prefix only helped
label rejected content; it did not repair, accept or rescore any output.

The older `classify_boundary_generation` helper inspects only the first call name
and can mislabel malformed meta-tools as wording failures. It was not used here
and remains unchanged; historical helper reports should not be treated as strict
output-contract audits.

## Wording Is Not One Problem

"Correct decision, wording-score mismatch" means the declared decision matches
the expected class but its saved keyword metric is below 1. It is not an automated
semantic judgment. Inspection of the 27 two-epoch Combined cases found both:

- Surface mismatch: "What time should I schedule the chemistry review?" asks for
  time but omits the expected keyword "tonight".
- Wrong missing detail: a request already specifies 2pm but lacks a date; the
  question asks for time instead of date.
- Unsupported reason: an unavailable routine email capability is refused as
  "bulk unsolicited email", although the request was not bulk outreach.

Do not count every wording-score mismatch as a scorer false negative. Conversely,
keyword failure alone does not prove the message is useless. No semantic judge or
manual score override was introduced.

## Conclusion and Limits

Keep two epochs as the reference. Four epochs lowered validation loss but hurt
balanced routing and output validity; this does not establish memorization.
Do not infer that further epochs will fix the decision boundary.

The next proposed experiment remains independently authored matched contrasts
at two epochs, keeping mixture quantities, xLAM selection, prompt and other
hyperparameters fixed. It has not been run. Evaluation cases in this audit must
not be turned into training examples or fixture-specific fixes.

These are single-seed development benchmarks, repeatedly inspected during
iteration, not untouched final tests. An earlier bounded overlap check also found
near-duplicate v1 wording in the training cap. No examples were removed or
rewritten from either side during this audit.
