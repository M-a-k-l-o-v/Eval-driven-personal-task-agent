# Tool Names Versus Clarification and Refusal Text

**Status:** Owner-run component measurement completed.
Date: 2026-10-15.

## Question

Whole-answer loss is higher for CLARIFY and REFUSE than ACT. Locate that difference
within the answer: choosing the tool name, composing boundary text, or other tokens.
Do not change the checkpoint, prompt, dataset, labels, caps, or benchmark rules.

## Command

```powershell
.venv\Scripts\python scripts\report_sft_loss.py `
  --split train `
  --max-rows 1000 `
  --max-sequence-length 1024 `
  --by-component `
  --output tmp\name-first-training-loss-components.json
```

The default remains the same frozen two-epoch QKVO name-first checkpoint. The
output is a new artifact; the earlier whole-answer report remains untouched.
Without `--by-component`, report behavior and schema remain unchanged.

## Components

| Component | Scored content |
| --- | --- |
| `first_tool_name` | The first call's `name` string contents. |
| `later_tool_name` | Other call-name string contents in multi-tool targets. |
| `question_text` | The `arguments.question` string contents of `request_clarification`. |
| `reason_text` | The `arguments.reason` string contents of `refuse_request`. |
| `other` | JSON syntax, keys, productive-tool arguments, and visible chat endings. |
| `mixed_boundary` | Tokens overlapping both a named string region and its surrounding syntax. |

Use the standard JSON decoder to locate field values without reserializing any
target. Fast-tokenizer character offsets map those regions onto the exact training
tokens. Quotes are not part of the string-content regions. Tokens crossing a region
boundary are not silently attributed to pure names or wording. Escaped text remains
in its original serialized representation.

Verify offset-tokenized IDs match training IDs and that the prompt token sequence
is a prefix of the full input. Prompt labels and position zero remain excluded.
Preserve skipped and partially truncated rows exactly as the original diagnostic.

One forward pass supplies the model loss and logits. Compute unreduced cross-entropy
only on supervised positions, using each token's preceding logit. Select at most
64 supervised positions at a time to limit temporary FP32 memory. Reject non-finite
losses and discrepancies between component totals and the model's reported loss.
Every scored token belongs to exactly one component.

The console adds a class/component table with contributing rows, tokens and loss.
The JSON includes per-row components, class summaries and all-class component
summaries, alongside the existing identity hashes and whole-answer losses.

## Interpretation

Large first-name loss supports investigating tool-choice learning. Low name loss
alongside higher question/reason loss supports investigating supplied-answer text.
Neither finding by itself proves a cause or chooses the next training intervention.

These are teacher-forced losses, not free-generation routing probabilities or
semantic adequacy scores. Later tool-name pieces see earlier gold name pieces;
question and reason tokens see the correct gold tool name. In argument-first
formats, even the first name sees preceding gold arguments, so format must remain
fixed when interpreting comparisons. Inspect mixed-boundary coverage before
interpreting pure-string statistics.

## Owner-Run Result

Source: `tmp/name-first-training-loss-components.json` and the owner's terminal output.
Whole-answer losses and retained row counts matched the earlier class-loss report.

| Class | Component | Rows | Tokens | Loss/token |
| --- | --- | ---: | ---: | ---: |
| ACT | First tool name | 774 | 2,575 | 0.016015 |
| ACT | Later tool names | 338 | 1,410 | 0.001974 |
| ACT | Other | 774 | 26,879 | 0.043888 |
| CLARIFY | First tool name | 100 | 400 | 0.276975 |
| CLARIFY | Question text | 100 | 1,045 | 0.817933 |
| CLARIFY | Other | 100 | 1,300 | 0.000393 |
| CLARIFY | Mixed boundary | 100 | 100 | 0.193296 |
| REFUSE | First tool name | 100 | 300 | 0.164804 |
| REFUSE | Reason text | 100 | 1,205 | 0.651727 |
| REFUSE | Other | 100 | 1,300 | 0.000294 |
| REFUSE | Mixed boundary | 100 | 100 | 0.351888 |

Boundary JSON syntax is fitted closely; the largest boundary-answer loss is on
supplied question/reason text. Different acceptable phrasing can also incur text
loss, so this is not proof of inadequate questions or reasons. First meta-tool
names have higher loss than productive first names, but name-loss averages still
condition on earlier gold name pieces and are not a routing confusion matrix.

The owner chose a [first-tool-name weighting ablation](13-first-tool-name-weight3.md)
before considering a router/generator architecture change. This is a routing-focused
hypothesis, not a demonstrated remedy for the higher text loss.

No checkpoint was run by the coding agent. Verification uses mocked inputs, tiny
CPU tensor arithmetic when PyTorch is installed, and model-free preflight.
