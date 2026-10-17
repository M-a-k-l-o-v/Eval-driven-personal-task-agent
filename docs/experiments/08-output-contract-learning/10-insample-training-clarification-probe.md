# In-Sample Clarification: Can the Frozen Model Reproduce Seen Behavior?

**Status:** Owner-run inference completed; raw responses reviewed.
Date: 2026-10-15.

## Purpose

The reviewed targets largely address the right blockers. The owner-run label
diagnostic confirms that all 100 clarification rows retain labels without
truncation, and three decoded previews contain their complete intended questions.
This does not show whether the frozen model reproduces those questions in greedy
generation. Test that before choosing more training or another data change.

## Ten Reviewed Inputs

Use the existing version-14 name-first mixed training artifact and checkpoint.
Select one-based training file lines 1, 2, 3, 4, 5, 6, 7, 8, 10 and 12. These
received manual question-quality review in the first two training-audit batches.
Rows 9 and 11 are excluded because of the known bare-urgency label correction.
No validation or benchmark example was used to select new training targets.

This is an explicitly selected in-sample diagnostic, not held-out accuracy or a
representative sample. Review the emitted question's adequacy, not merely its
CLARIFY label. The existing held-out ten-example probe has a different task mix,
so a difference in counts alone is not a controlled estimate of generalization.

## Command

```powershell
.venv\Scripts\python scripts\probe_sft_clarification.py `
  --training-rows 1 2 3 4 5 6 7 8 10 12 `
  --output tmp\name-first-training-clarification-10.json
```

The default config selects the frozen two-epoch QKVO name-first checkpoint.
Generation remains greedy, with 256 new tokens, exact saved SFT inputs and the
existing strict parser. Only system/user messages reach inference. Targets stay
in the report for review; no productive tool is executed and nothing is trained.

Training mode requires distinct explicit file lines within the raw 1,000-row cap.
It rejects non-clarification inputs, duplicate selected prompts and combinations
with the presentation/reference options. Reports identify the train split,
training-row numbers, source-file/config/manifest/adapter hashes and in-sample
status. Existing output files are not overwritten.

Default held-out selection and overlap rejection are preserved. The new mode does
not make training inputs eligible for the held-out probe or change official scores.
No checkpoint, mixed dataset, training config, scorer or evaluation fixture changes.

## Interpretation

Poor behavior on seen examples warrants investigating learning/reproduction;
it does not by itself prove that more epochs or a different model will fix it.
Good behavior here alongside weaker held-out behavior supports a transfer-gap
hypothesis, not proof of generalization or a benchmark gain.

Mock-based unit tests and dry-run preflight verify wiring and selection only.

## Measured Result

Source: `tmp/name-first-training-clarification-10.json`.
The owner ran inference; the coding agent read the saved responses. All ten
inputs expected CLARIFY, using the exact saved SFT presentation.

| Predicted decision | Examples |
| --- | ---: |
| CLARIFY | 4 |
| ACT | 4 |
| REFUSE | 2 |
| Invalid/unknown | 0 |

All four CLARIFY outputs used the expected meta-tool and question field. There
were no parser failures. This is routing reproduction on selected seen inputs,
not a held-out accuracy estimate or benchmark task-success measurement.

## Raw-Response Review

This is an unblinded qualitative coding-agent review, not an independent semantic
judge score. Training row numbers are one-based file lines.

| Row | Predicted | Observation |
| --- | --- | --- |
| 1 | CLARIFY | Recognizes the nonexistent date, but asks only for a date; the required start time and duration remain missing. |
| 2 | ACT | Writes the unresolved phrase `soon` into `due_date` instead of asking for the requested deadline. |
| 3 | REFUSE | Claims `gmail.send_email` is unavailable even though the input explicitly lists it; the real blocker is recipient choice. |
| 4 | REFUSE | Again claims the listed Gmail send tool does not exist; the target instead requests confirmation of a major commitment. |
| 5 | CLARIFY | Asks for the missing unit, but offers only mass units without evidence that the measurement is a mass. It does not actually select a unit. |
| 6 | CLARIFY | Adequately asks which of the two existing meetings to move. |
| 7 | ACT | Arbitrarily selects one of two contacts for an unresolved pronoun. |
| 8 | ACT | Copies the literal word `it` instead of asking what content to copy. |
| 10 | ACT | Invents a 14:00 start and 60-minute duration. |
| 12 | CLARIFY | Adequately asks for the missing required event title. |

## What This Establishes

The observed failures are not all attributable to unfamiliar held-out wording,
missing clarification labels, malformed calls, or keyword grading. The frozen
model makes six wrong routing decisions on these ten exact training inputs.
Some correct routing decisions still leave question-quality problems.

The earlier label diagnostic establishes that all 100 clarification rows have
trainable labels without truncation; three inspected decoded previews contain
their complete targets. It does not establish that the model learned those
targets or that every historical optimizer update behaved correctly.

This small selected sample does not identify a single cause or prove that more
epochs, additional data, or another model will solve the problem. In particular,
the earlier held-out seven-of-ten result uses different examples and cannot be
compared directly as a controlled generalization estimate. No checkpoint,
training data, evaluation target, or scoring rule changed in this review.
