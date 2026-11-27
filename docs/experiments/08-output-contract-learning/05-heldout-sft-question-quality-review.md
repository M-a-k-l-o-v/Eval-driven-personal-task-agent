# Held-Out SFT: Question Quality Versus Supplied Targets

**Status:** First ten-example batch reviewed, 2026-10-14.

## What Was Checked

Used the existing owner-run
[name-first held-out probe](../../../tmp/name-first-heldout-clarification-10.json).
No new model run or comparison command was executed. This report belongs to the
two-epoch QKVO name-first checkpoint, not A-reference.

The saved inputs use their exact compact SFT presentation. All ten expected
CLARIFY. They are the first ten unique clarification examples selected from the
first 200 validation rows; they are not a random sample.

Reviewed each request, state, exposed tool schema, supplied target question and
saved model output. Checked all ten inputs/targets against their current
validation.jsonl rows: all matched. The probe's original preflight checked exact
input overlap against its first 1,000 training rows; that is not a semantic or
paraphrase leakage guarantee.

Probe SHA256: `9601e71f96867f0e5f18c494f275da086bf7439ade051d55e32d2f04ca31928e`.
Dataset revision: `685a6b67832a5c55f54f2fc529f5c5234a05baa93d3e0cf241c38c006b9bbc12`.
Recorded adapter weights SHA256:
`fb41064d104e72da4984526e96e06c8ad5c883aad11b585f47252c73b6b24db9`.

This is an unblinded coding-agent diagnostic, not independent human validation
or a new benchmark score. No evaluation fixture was used to construct training
data, and no source row, target, prompt, checkpoint or scorer was changed.

## Criteria

Used the same qualitative distinctions as the previous question review:
adequate, relevant but incomplete, wrong focus. Productive ACT outputs are
classified as routing failures separately, not counted as bad questions.

Information can be necessary because the tool requires it, because the request
explicitly asks for it, or because state does not identify a unique entity.
For example, due_date is optional in the Notion schema but cannot be invented or
discarded when the user explicitly requests a task with an unspecified deadline.
Conversely, Notion status defaults to open and need not be requested unnecessarily.

## Cases

| Validation row | Blocker and supplied target | Saved model behavior | Assessment |
| --- | --- | --- | --- |
| 6 | Packing-list item text is absent. Target asks for the first item. | "What should I copy?" | Adequate in context. |
| 7 | April 31 is not a valid date. Target explains April has 30 days and asks which due date to use. | ACT creates the task with 2026-04-31. | Routing failure; invalid user value propagated. |
| 12 | Requested checklist contents are absent. Target asks for checklist text. | ACT writes "Marv's psychology problem set checklist" as the contents. | Routing failure; a label is substituted for missing content. |
| 15 | Two open review tasks match. Target asks which review task to mark done. | ACT chooses task_history_review. | Routing failure; ambiguous entity selected without clarification. |
| 20 | Invitation reply needs an accept/decline choice not yet made. Target asks for that choice. | "What should I say? Accept or decline?" | Adequate. |
| 25 | Tonight's event lacks exact time and duration. Target asks what time and how many minutes. | "What time should I schedule the statistics bibliography?" | Incomplete; duration remains unknown. |
| 37 | No event ID or identifying details are provided. Target asks which event, requesting ID or details. | "Which event should I move?" | Adequate as a contextually understandable request for identification. |
| 40 | Two design planning review events match. Target asks which one. | "Which event should I move?" | Adequate; giving the two options would improve usability but is not necessary to ask the right question. |
| 41 | Two meeting events match. Target asks which meeting. | "Which event should I move?" | Adequate in context. |
| 42 | Task title Check stock is supplied, but the agreed due date is absent. Target asks for the exact due date. | "What should I name the task? What status should it have?" | Wrong focus; asks for a known title and unnecessary status, omitting the actual blocker. |

## Findings

| Saved output assessment | Cases |
| --- | ---: |
| Adequate clarification | 5 |
| Incomplete clarification | 1 |
| Wrong-focus clarification | 1 |
| ACT instead of clarification | 3 |

The seven valid CLARIFY decisions contain five adequate questions and two
question-quality problems. These are manual judgments on a small selected
sample, not a semantic accuracy metric for the entire validation split.

All ten supplied target questions appear to address their own request/state
blockers appropriately in this batch. In particular, row 25's target includes
duration and row 42's target asks for the due date. Therefore these observed
generation errors cannot simply be explained by incorrect target questions in
those two held-out examples. This does not establish that the actual training
targets or the complete dataset are sound.

The same broad completeness and field-grounding failures occur even with exact
SFT input presentation and no keyword scorer. Formatting is a demonstrated
sensitivity, but is not the only remaining problem. These ten examples do not
establish whether additional exposure, data diversity or another model change
would resolve it.

## Next Small Batch

Inspect a small deterministic sample of clarification targets from the actual
first 1,000 training rows, using their own requests, state and tool schemas.
Keep this separate from the held-out generated-output review. Do not copy these
validation examples into training or derive new targets from benchmark failures.
No training adjustment is proposed or applied in this batch.
