# Capped Training Questions: Second Ten-Target Batch

**Status:** Completed manual review, 2026-10-14.

## Selection and Scope

Continued the [first capped-training audit](06-capped-training-clarification-target-audit.md)
on the frozen name-first training artifact used by the existing checkpoint.
This is not an audit of the newly built generator-15 source, which has not been
mixed or trained in this investigation.

Selected the next ten unique CLARIFY inputs after training rows 1-10, in the
unchanged first 1,000 raw training rows. Row 11 was excluded because it repeats
the already-understood bare-urgency routing-policy issue. Selected rows are
12-21. Selection did not read evaluation scenarios or result files.

Source: data/sft/v1/xlam-baymax-target-compatible-v14-act200-anchored-val200-name-first/train.jsonl.
Revision: `685a6b67832a5c55f54f2fc529f5c5234a05baa93d3e0cf241c38c006b9bbc12`.
SHA256 of the first 1,000 raw lines remains
`99e866d226db88bd762d6a6615e23a26ad4d85d0b26976546b6dd92f51cc9119`.

Reviewed each request, time/state, productive tool schema and supplied target
question. All ten passed the existing strict target-contract validator with
zero issues. No tokenizer, model, comparison, dataset build or training was run.
No input, target, checkpoint or benchmark score was changed.

These are unblinded coding-agent judgments on a deterministic selected sample,
not independent human validation or a semantic benchmark metric. Exact input
uniqueness does not establish different templates or semantic split independence.

## Cases

| Training row | Source row ID | Actual blocker | Target question | Assessment |
| --- | --- | --- | --- | --- |
| 12 | baymax:train:628:r0 | Event title undecided. Date, start time and end time are supplied. | "What title should I give the event? A title is required to create it." | Adequate: requests the required title; duration can be derived as 50 minutes. |
| 13 | baymax:train:17:r0 | Externally visible service-contract acceptance requires confirmation. | "Please confirm you want me to send this contract acceptance." | Adequate under the agreed confirmation policy. |
| 14 | baymax:train:359:r0 | Requested due date 2026-13-06 is invalid; task ID is known. | "There is no month 13. What valid due date should I use?" | Adequate: identifies the invalid field and asks for its correction. |
| 15 | baymax:train:131:r0 | Sana's economics project outline is explicitly "due soon," without an exact date. | "What due date should I use for Sana's economics project outline?" | Adequate: clarifies a requested deadline, not bare urgency. |
| 16 | baymax:train:401:r0 | Tonight's history worksheet event has no exact time or duration. | "What time tonight should I schedule it, and how many minutes should it last?" | Adequate: asks for both missing required fields. |
| 17 | baymax:train:700:r0 | Task deadline is June 25 or June 27; user explicitly says not to choose. | "Should Label storage boxes be due June 25 or June 27?" | Adequate: requests the user's choice without asking for known task text. |
| 18 | baymax:train:538:r0 | Sana's anatomy quiz-prep task is explicitly "due soon." | "What due date should I use for Sana's anatomy quiz prep?" | Adequate: asks for the unresolved deadline. |
| 19 | baymax:train:107:r0 | Two literature tasks match "the literature task." | "Which literature task should I update?" | Adequate: requests entity selection; done status is already known. |
| 20 | baymax:train:598:r0 | Statistics meeting is not identifiable from empty calendar state; requested new time is known. | "Which statistics meeting should I move?" | Adequate for identification, but could explain that no matching meeting was found and request identifying details. It does not invent an ID or re-ask the known time. |
| 21 | baymax:train:142:r0 | Two other literature tasks match the request. | "Which literature task should I update?" | Adequate: same ambiguity pattern as row 19, with different state. |

## Findings

| Second-batch finding | Cases |
| --- | ---: |
| Appropriate clarification targets | 10 |
| New incomplete/wrong-focus target defects found | 0 |
| Structural contract failures | 0 |
| Known urgency-pattern rows excluded before selection | 1 |

This batch provides no evidence that the model's incomplete or wrong-focus
questions are simply imitating defective targets in these rows. Correct targets
do not prove that the model learned them, that the full dataset is correct, or
that the supplied target remains fully supervised after tokenization.

There is some template reuse: rows 19 and 21 share request/question wording while
their state differs. That is not an incorrect target or an exact duplicate input
by itself, and this sample does not quantify overall semantic diversity.

Across the two batches, 20 targets received semantic review: 19 appear appropriate
and one belongs to the agreed urgency-policy correction. Another occurrence of
that known pattern was deliberately excluded from this second batch. Do not
present these selected counts as a random-sample defect rate.

## Recommendation

Do not change the data mixture or add more examples based on this batch alone.
The remaining target questions inspected so far are appropriate. A bounded next
check is the actual supervision path: whether complete clarification targets,
including their question text, retain assistant labels after chat-template
rendering and truncation in the frozen experiment. That check has not been run
as part of this manual review and should precede another training adjustment.
