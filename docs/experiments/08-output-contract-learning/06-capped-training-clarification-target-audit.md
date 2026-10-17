# Capped Training Data: Ten Clarification Targets

**Status:** First batch reviewed; one routing-policy correction agreed, 2026-10-14.

## Scope and Selection

Inspected the first ten unique clarification inputs in the first 1,000 raw rows
of the name-first training dataset. These happen to occupy training rows 1-10.
The capped pool contains 100 clarification rows with 100 distinct input hashes;
only ten received this semantic review. Exact uniqueness does not guarantee
different templates or semantic independence between splits.

Each selected request/state and parsed assistant target has the same meaning in
A-reference's anchored dataset. Name-first changes target key order, not the
selected questions or inputs. The findings therefore apply to both artifacts.

No evaluation scenario or result was read to select or rewrite these examples.
No model, tokenizer, training, evaluation or comparison command was run. Rows
were not re-filtered for trainable tokens; this is a bounded review of the raw
cap, not a new post-tokenization artifact report.

Source: data/sft/v1/xlam-baymax-target-compatible-v14-act200-anchored-val200-name-first/train.jsonl.
Revision: `685a6b67832a5c55f54f2fc529f5c5234a05baa93d3e0cf241c38c006b9bbc12`.
SHA256 of the first 1,000 raw lines:
`99e866d226db88bd762d6a6615e23a26ad4d85d0b26976546b6dd92f51cc9119`.

## Structural Versus Semantic Checks

The existing strict target-contract validator was run on the ten selected rows:
**zero structural issues**. Every target is a valid request_clarification call
with its required question argument.

Separately, the coding agent reviewed whether the question addresses an actual
blocker, covers all missing required information and avoids requesting already
known fields. This is qualitative, unblinded review, not an independent human
study or a new benchmark metric. Product-policy ambiguity was referred to the
owner rather than silently resolved by the reviewer.

## Cases

| Training row | Source row ID | Request or blocker | Target question | Assessment |
| --- | --- | --- | --- | --- |
| 1 | baymax:train:556:r0 | Calendar event on invalid February 30, with no time or duration. | "What valid date should I use, what time should it start, and how many minutes should it last?" | Adequate: covers invalid date and both missing fields. |
| 2 | baymax:train:40:r0 | Create Sana's history flashcards task, explicitly "due soon." | "What due date should I use for Sana's history flashcards?" | Adequate: an explicit deadline is requested but unspecified. |
| 3 | baymax:train:748:r0 | Routine sample-ready email; user has not chosen between two recipients. | "Should I send the sample update to kit@example.com or ren@example.com?" | Adequate: asks for the missing recipient choice. |
| 4 | baymax:train:226:r0 | Email accepting a service contract, a consequential commitment. | "Please confirm you want me to send this contract acceptance." | Adequate under the existing confirmation policy; asks for authorization, not routine-send details. |
| 5 | baymax:train:711:r0 | Copy measurement 18 with its correct unit; user explicitly says unit is absent. | "Which unit should follow 18 in the copied measurement?" | Adequate: asks for missing content instead of inventing it. |
| 6 | baymax:train:576:r0 | Move "my meeting" to 3pm; state contains two matching meetings. | "Which meeting should I move?" | Adequate: resolves entity ambiguity without re-asking the known time. |
| 7 | baymax:train:502:r0 | Email "them" a completion update; state contains two contacts. | "Who should I send the email to?" | Adequate: asks for recipient identity; body facts are supplied. |
| 8 | baymax:train:66:r0 | "Copy it" with no supplied text and empty clipboard state. | "What text should I copy?" | Adequate: asks for the required text. |
| 9 | baymax:train:367:r0 | "Create a task to revise anatomy soon." No explicit deadline is requested; due_date is optional. | "What due date should I use for soon?" | Routing-policy defect under the owner's decision below: create an undated task, do not require clarification solely for vague urgency. |
| 10 | baymax:train:389:r0 | Calendar event on a known date, but no time or duration. | "What time should I schedule the programming lab report, and how many minutes should it last?" | Adequate: covers both missing required fields. |

## Owner Decision: Vague Urgency Is Not a Required Deadline

The owner selected **create an undated task** for row 9. Bare urgency such as
"revise anatomy soon" does not by itself require a due date when the available
task-creation schema permits omission. Do not invent a date to represent urgency.

This differs from row 2's explicit "due soon" or a request to use an unspecified
agreed deadline: a due date is requested in those cases, so clarification remains
appropriate when it cannot be grounded.

The generator currently labels row 9's pattern CLARIFY in
[baymax_sft.py](../../../src/baymax/models/baymax_sft.py): the seed and generated
Notion examples ask what date to use for "soon." These historical labels were
not changed during this review. Applying the owner decision requires a focused
generator correction and a separately versioned derived dataset, not editing
immutable training files or reinterpreting existing benchmark scores.

## Findings and Limits

| Review result | Cases |
| --- | ---: |
| Clearly appropriate clarification targets | 9 |
| Routing target needing correction under agreed policy | 1 |
| Structural contract failures | 0 |

The calendar targets inspected here ask for both time and duration when both are
missing. No reviewed question asks for a known title or ignores a known time to
request it again. These findings do not establish that the other 90 clarification
targets or the ACT/REFUSE targets are correct.

The incorrect urgency label encourages unnecessary clarification, not missing
clarification. It cannot by itself explain the model's observed ACT bias or the
full benchmark's question-quality failures. Do not present this one correction
as a demonstrated performance improvement or the sole training issue.

No training data, generator, config, checkpoint, scorer or benchmark target was
modified. The next bounded change is to implement the generic optional-deadline
rule in training generation and check related training cases before preparing a
new artifact. There is no retraining in this batch.
