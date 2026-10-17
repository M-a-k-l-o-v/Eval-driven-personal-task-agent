# Training Targets: Missing Fields and State Grounding

**Status:** Completed bounded training-data audit; no training change made.
Date: 2026-10-16.

## Question and Method

Before prescribing more epochs, check whether the actual training cap teaches
asking for all missing required information and transferring supplied state into
tool arguments. Correct targets do not prove that the model learned them; absence
or narrow coverage does not by itself prove the cause of a model error.

Use only the frozen training artifact and existing training diagnostics. No
validation/test/benchmark example or output was used to pick, construct or rewrite
a training target in this audit. No dataset was generated, mixed or changed.

Earlier notes already reviewed 20 clarification targets. This batch extends that
review with the next eight clarification rows, file lines 22-29, without exclusions.
For ACT, select the first four single-call and first four multi-call BAYMAX rows
in file order that contain meaningful state: existing events/tasks/contacts or
nonempty clipboard text. Empty state lists alone do not qualify.

The resulting 16-row semantic sample is deterministic and selected by behavior,
not random. Its judgments are an unblinded coding-agent review, not an independent
human study or an estimated defect rate for the entire dataset. Structural counts
below cover the capped BAYMAX pool; semantic conclusions cover only reviewed rows.

## Source and Supervision

Dataset:
`data/sft/v1/xlam-baymax-target-compatible-v14-act200-anchored-val200-name-first/train.jsonl`.
Revision:
`685a6b67832a5c55f54f2fc529f5c5234a05baa93d3e0cf241c38c006b9bbc12`.
Whole training-file SHA256:
`70e043cb05ef2d3dcb1fc5f9a61b6551ea7efc73050b6773b04ab1911c93fac7`.

The saved owner-run component diagnostic,
`tmp/name-first-training-loss-components.json`, records this same training-file
identity. Its SHA256 is
`ac600c6e44d8eaea3a615b8f037c4320c73ed7045bd9f113103ddba1695b1225`.
All 16 selected rows retain positive scored-token counts and are not truncated in
that report. This is existing evidence, not new tokenizer/model execution.

All 16 selected targets passed the existing strict target-contract validator in a
model-free check. Schema validity and retained labels are not proof of semantic
quality or successful learning. The same frozen data is used by the 1x, 2x and
3x first-name-weight experiments; none uses the separately built v15 source.

## Actual Capped BAYMAX Pool

The first 1,000 raw training rows contain 400 BAYMAX and 600 xLAM examples. The
previous tokenized report retains all 400 BAYMAX and 574 xLAM rows. The following
counts refer only to BAYMAX targets, not the much larger external ACT pool.

| Target behavior | Rows |
| --- | ---: |
| CLARIFY | 100 |
| REFUSE | 100 |
| ACT, one call | 175 |
| ACT, multiple calls | 25 |

Among these, 54 single-call ACT rows and nine multi-call ACT rows contain nonempty
calendar/task/contact state or clipboard text. These structural counts do not
prove that every target actually needs or correctly uses its state.

All 25 multi-call BAYMAX targets contain exactly two productive calls, from six
ordered tool-name patterns:

| Ordered pattern | Training rows |
| --- | ---: |
| `clipboard.read` -> `notion.create_task` | 3 |
| `clipboard.read` -> `gmail.create_draft` | 2 |
| `calendar.create_event` -> `notion.create_task` | 5 |
| `calendar.create_event` -> `gmail.create_draft` | 5 |
| `notion.update_task` -> `gmail.send_email` | 4 |
| `notion.create_task` -> `gmail.create_draft` | 6 |

This pool teaches several two-step compositions. It does not provide a BAYMAX
example that calls the same productive tool twice or a three-step composition.
This is a description of the training pool, not a proposal to copy any benchmark
subtype into training. External xLAM examples can contain other compositions;
they were not semantically audited here.

## Eight Additional Clarification Targets

| Training row | Actual blocker | Target question | Assessment |
| --- | --- | --- | --- |
| 22 | No identifiable statistics meeting in calendar state; requested time is 11:00. | "Which statistics meeting should I move?" | Appropriate entity-identification question. Could explain no match was found, but does not re-ask the known time. |
| 23 | Draft needs the explicitly requested agreed deadline, which is not supplied. | "What is the agreed deadline for the history presentation?" | Appropriate; asks for the missing fact rather than inventing body content. |
| 24 | Event date is February 30; time and duration are absent. | "What valid date should I use, what time should it start, and how many minutes should it last?" | Appropriate; covers the invalid date and both missing required fields. |
| 25 | Recipient pronoun is unresolved between two contacts; message facts are supplied. | "Who should I send the email to?" | Appropriate; does not select an arbitrary contact. |
| 26 | User has not supplied the code snippet to copy. | "Please provide the code snippet to copy." | Appropriate; requests content instead of copying the description of the operation. |
| 27 | User explicitly requests specific wording that is absent. | "Please provide the wording you want sent to sol@example.com." | Appropriate; recipient is known, requested wording is missing. |
| 28 | Date undecided; title, 11:20 start and 40-minute duration are known. | "Which date should I use for volunteer orientation?" | Appropriate; does not ask again for known time or duration. |
| 29 | User has not chosen between two supplied date formats. | "Which date format should I copy: 2026-06-26 or 26/06/2026?" | Appropriate; requests the unresolved choice rather than choosing for the user. |

No new incomplete/wrong-focus target defect was found in these eight rows. This
does not certify the remaining clarification targets. Earlier reviewed examples
also teach asking for time and duration together when both are absent.

## Eight State-Bearing ACT Targets

### Single Call

| Training row | State and requested operation | Target mapping | Assessment |
| --- | --- | --- | --- |
| 206 | One calculus planning review exists; move it to 13:00. | Uses the supplied event ID and `start_time: "13:00"`. | Correct state-bound entity selection; no invented duration or date. |
| 211 | One chemistry planning review exists; move it to 13:00. | Uses that event's supplied ID and requested time. | Correct, but closely repeats row 206's request/operation pattern. |
| 214 | One programming reading-notes task exists; mark it done. | Uses its supplied task ID and `status: "done"`. | Correct state-bound entity selection. |
| 227 | One programming problem-set task exists; mark it done. | Uses its supplied task ID and done status. | Correct, but closely repeats row 214's pattern. |

### Multiple Calls

| Training row | State and requested operation | Target mapping | Assessment |
| --- | --- | --- | --- |
| 224 | Clipboard contains `statistics reading notes`; read it and create a task from it. | Reads clipboard, then sets task title to `statistics reading notes`. | Correct content transfer and complete two-call plan. |
| 242 | Clipboard contains `physics reading notes`; same request wording as row 224. | Reads clipboard, then uses that text as title. | Correct; varies content but not request wording. |
| 246 | Clipboard contains `chemistry notes for Nadia`; read and draft to the supplied recipient. | Reads clipboard, then uses that text as body and the supplied recipient. | Correct content transfer; subject is permissible composition. |
| 248 | One programming-outline task exists; mark it done and email the supplied recipient. | Updates the supplied task ID, then sends the supplied completion fact. | Correct entity mapping and complete sequence under the existing routine-send policy. |

All eight reviewed ACT targets appear consistent with their own requests, state
and tool schemas. This is target review, not a model generation or tool execution.

## Coverage and Consistency Limits

The three `clipboard.read` -> `notion.create_task` examples are training rows
224, 242 and 285. All say exactly "Read my clipboard and create a task from it."
Their clipboard texts differ, and all three targets correctly use the text as the
task title. Thus the mapping is taught, but by only three content variations of
one instruction template. The two clipboard-to-draft examples similarly vary
content/recipient while using the same instruction pattern.

Some clarification target questions also repeat across different inputs/state.
For example, six targets ask "Who should I send the email to?" and four ask
"What time tonight should I schedule it, and how many minutes should it last?"
Shared questions can be correct and useful; they are not automatically duplicate
examples or evidence that the model memorized them. Counts alone do not measure
semantic diversity.

The known bare-urgency policy error remains in the frozen v14 cap at rows 9, 11
and 81. These requests say to revise a subject "soon" without explicitly asking
for a deadline, yet their targets ask for a due date. Under the owner's decision,
the tool permits creating an undated task instead. Six other "due soon" requests
explicitly request deadlines and are not the same defect.

Generator 15 corrects this generic policy rule, and its separately built source
exists locally. The checkpoint's frozen v14 mix was not repointed or rebuilt.
Updating generator code or building another source does not retroactively fix
the data learned by an existing adapter. See the
[policy correction note](07-generator15-optional-task-deadline-fix.md).

The three known bad labels encourage unnecessary clarification, not failure to
clarify. They cannot alone explain the checkpoint's ACT bias or all observed
grounding/clarification errors. They were not silently removed from this audit's
accounting or used to reinterpret historical results.

## What This Means for More Training

The checked skills are not absent from training: reviewed examples teach asking
for missing required fields and copying supplied state into arguments, and all
16 have nontruncated supervision in the saved diagnostic. The existing selected
in-sample clarification probe nevertheless makes wrong decisions on correctly
labeled seen inputs; review of coverage alone cannot explain that away.

The data also provides narrow coverage for some compositions and retains a known
policy inconsistency. Repeating it for more epochs repeats both its limited
patterns and its wrong labels. Therefore this audit does not justify prescribing
more epochs as a fix, nor prove that adding data will solve the problem.

Keep the unweighted reference. A defensible future data experiment would use a
new versioned artifact, resolve independently established policy contradictions,
and test broader expression/state variation at approximately fixed class counts
and training settings. Do not author validation/benchmark-specific solutions,
combine that with a duration sweep, or claim improvement without a new run.
No such experiment was implemented or trained in this batch.

Only local notes were added/linked. No data, source code, configuration,
checkpoint, scorer or prior result was changed; nothing was staged or committed.
