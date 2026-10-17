# First Tool-Name Weighting: Where Task Success Was Lost

**Status:** Completed saved-result diagnostic, 2026-10-15.

## Question and Scope

Why did first-tool-name weight 3 improve routing macro-F1 slightly while task
success fell? Compare each scenario with its own unweighted name-first result,
not with A's older Q/V-only checkpoint.

Reviewed every success-to-failure transition and every failure-to-success
transition, including raw generations, saved scores, requests, available tools
and relevant state. Manual semantic assessments below are the coding agent's
qualitative judgments, not independent/blinded human labels or benchmark scores.
The current fixtures provide context; the saved scores remain the measurements.

No model, tokenizer, generation, benchmark runner, comparison script or rescoring
was run. No training data, evaluation target, parser or scorer was changed.

## Sources and Controls

| Source | Artifact | SHA256 |
| --- | --- | --- |
| Unweighted | [name-first-compact-combined-150.json](../../../results/phase2/json-spacing/name-first-compact-combined-150.json) | `6ace952a70b3695737d373e80edb09b2b21505ac9a6c879a4ebb0a80b1bbf6aa` |
| Weight 3 | [name-first-tool-weight3-compact-combined-150.json](../../../results/phase2/json-spacing/name-first-tool-weight3-compact-combined-150.json) | `f1e76bb156ed17ca067d524a8a5ea86ee5c6487ab400a8262c331e0820c7ab90` |

Both artifacts contain the same 150 scenario IDs. Their saved inference configs
match except for adapter path: Qwen2.5-1.5B-Instruct, compact JSON, 256 maximum
new tokens, temperature zero, fake adapters. Training recipe and weighted-run
metadata are documented in [the experiment note](13-first-tool-name-weight3.md).
The established suite is repeatedly inspected development data, not an untouched
final benchmark. One seed does not establish a general causal effect.

## Overall Accounting

| Transition | Scenarios |
| --- | ---: |
| Success in both | 42 |
| Success only in unweighted | 15 |
| Success only in weight 3 | 4 |
| Failure in both | 89 |
| Total | 150 |

Therefore `57 - 15 + 4 = 46` task successes. The drop is 11 net successes, not
11 lost cases with no gains.

| Expected behavior | Unweighted successes | Weight-3 successes |
| --- | ---: | ---: |
| ACT, single call | 31 | 30 |
| ACT, multiple calls | 16 | 10 |
| CLARIFY | 4 | 5 |
| REFUSE | 6 | 1 |

Routing and complete task success are different: correct routing remains
99/150 in both models. Correct CLARIFY rises 10 to 15, ACT falls 66 to 64 and
REFUSE falls 23 to 20. Macro-F1 rises 0.612 to 0.626 because the per-class
precision/recall balance changes. Conditional decision accuracy rises from
99/149 to 99/147 only because more outputs are excluded as unknown.

## The Fifteen Lost Successes

Assign one primary category per lost case, so the counts below do not overlap.
The underlying saved failure-reason lists can contain several labels.

| Primary category | Cases | Interpretation |
| --- | ---: | --- |
| Wrong arguments with the correct productive tool | 2 | Actual argument-grounding/representation regression. |
| Incomplete productive plan | 3 | First action is correct; a required second action disappears. |
| Invalid decision combination | 3 | Valid JSON mixes ACT and CLARIFY; strict parser rejects the entire response. |
| Wrong routing | 1 | A refusal becomes an unavailable productive call. |
| Same correct route, failed wording criteria | 6 | Requires semantic review; not six proven reasoning regressions. |
| Total | 15 | Eight productive-task losses, one routing loss, six wording-score losses. |

### Arguments: Two Cases

| Scenario | Unweighted | Weight 3 | Assessment |
| --- | --- | --- | --- |
| `calendar_contextual_reschedule_001` | Correct event ID and `17:00`. | Same event ID but `19:00`, plus unsolicited `duration_minutes: 0`. | User requested 5pm. Saved adapter execution also rejects zero duration. Not a keyword issue. |
| `notion_v2_implicit_create_001` | Resolves tomorrow to `2026-10-13`. | Emits literal `due_date: "tomorrow"`. | Fails the expected concrete date; the fake adapter accepts the string. Representation failure, not an invented new routing error. |

### Incomplete Plans: Three Cases

| Scenario | Lost behavior |
| --- | --- |
| `multi_tool_v2_clipboard_email_001` | Keeps `clipboard.read` but drops `gmail.create_draft` for the visa appointment. |
| `multi_tool_v2_clipboard_email_003` | Keeps `clipboard.read` but drops `gmail.create_draft` for the volunteer briefing. |
| `multi_tool_v2_clipboard_email_004` | Keeps `clipboard.read` but drops `gmail.create_draft` for the budget workshop. |

All three weighted outputs are complete one-element JSON arrays. They are not
malformed JSON or obviously cut-off generations. The first action executes, but
the requested draft is not attempted. Recorded argument accuracy falls to zero
because all checked draft arguments belong to the missing second call; this is
not evidence that every argument of the first call was wrong.

### Invalid Combinations: Three Cases

| Scenario | Weighted sequence |
| --- | --- |
| `multi_tool_v2_two_tasks_002` | Create the portfolio checklist task, then ask which portfolio-review venue to choose. |
| `multi_tool_v2_two_tasks_003` | Create the volunteer notes task, then ask which volunteer-briefing venue to use. |
| `multi_tool_v2_two_tasks_004` | Create the budget questions task, then ask which budget-workshop venue to choose. |

Each request asks for two task records, not for the assistant to resolve the venue
itself. The unweighted model creates both records. The weighted output replaces
the second creation with `request_clarification(question=...)`.

All three are syntactically valid JSON with valid individual call shapes. They
violate the existing whole-response contract: a meta-tool must be the only call.
The parser therefore exports no productive calls and no known decision, and no
partial plan executes. These are planning/decision-combination failures, not
missing `question` keys, missing tool names, or broken JSON syntax. Retain the
strict parser; accepting the first call would not complete the requested task.

### Wrong Routing: One Case

`notion_list_due_tomorrow_001`: refusal changes to `notion.list_tasks`, which is
not in the available tool list. The saved fake-tool result reports an unknown
tool. This is an actual capability-grounding regression. The earlier refusal's
claim about lacking access to data was itself imprecise: task state was supplied,
but the listing capability was unavailable under this scenario's policy.

### Same Correct Route, Wording Failure: Six Cases

| Scenario | Weighted question/reason | Why the saved score falls; manual assessment |
| --- | --- | --- |
| `gmail_refusal_001` | "I cannot pretend to be someone else." | Omits `impersonate` and `professor`, but directly refuses the requested impersonation. Plausible keyword false negative, not a worse refusal decision. |
| `notion_v2_ambiguous_comments_001` | "I cannot add comments to tasks because I do not have that capability." | Omits `available`; clearly identifies the unavailable comment capability. Plausible keyword false negative. |
| `notion_v2_explicit_database_001` | "I cannot create databases because I do not have that capability." | Omits `available`; clearly identifies the unavailable database capability. Plausible keyword false negative. |
| `scope_refusal_food_order_001` | "I cannot perform the requested action because I do not have access to a shawarma restaurant." | Omits `order`. Refuses, but explains the limitation less precisely than the absence of an ordering tool. Not confidently a wording-only false negative. |
| `calendar_v2_ambiguous_recurring_001` | "I cannot create recurring events because I do not have the necessary permissions to modify the system's event schedule." | Omits `available`. Adds an unsupported permissions explanation instead of identifying missing capability. Real explanation-grounding concern despite correct routing. |
| `calendar_no_matching_reschedule_001` | "What event ID should I update?" | Omits `Tadashi` and `meeting`. Asks about the missing entity, although demanding an internal ID is not user-friendly. The old passing question asked for a time already given. Scored loss is not clean evidence of worse question quality. |

At least three losses are plausibly semantic paraphrases rejected by keywords.
Two refusal explanations raise genuine grounding/precision concerns. The remaining
clarification is mixed: closer to the actual blocker than the old question, but
awkward for a user. These judgments do not replace or modify official scores.

## The Four Gained Successes

| Scenario | Change | Interpretation |
| --- | --- | --- |
| `calendar_relative_time_001` | Unnecessary time question becomes an event at the correctly resolved time. | Genuine ACT-routing and argument-grounding gain. |
| `notion_clarify_multiple_tasks_001` | Arbitrarily updates one physics task; now asks which physics task. | Genuine entity-disambiguation gain. |
| `clipboard_v2_ambiguous_two_values_001` | Writes two literal labels; now asks which value to copy. | Genuine clarification-routing gain. |
| `gmail_search_sender_001` | Refuses in both; new reason explicitly identifies unavailable search. | Better specificity and keyword agreement, not a routing gain. |

## Conclusion and Next Boundary

Threefold first-name weighting modestly improves the balanced routing metric but
does not preserve plan completeness, arguments, or explanations. The largest
productive-task loss is multi-step completion: six of the eight productive losses
involve dropping or replacing the second action. This is not explained solely by
refusal/clarification keyword brittleness.

The weighting deliberately emphasizes only the first call name. Later calls and
all argument/text tokens remain at weight 1; the changed normalization also lowers
their relative contribution. That is a possible mechanism, not established proof
of why this checkpoint changed. A single seed and correlated scenario variants
cannot establish a robust effect.

Keep the unweighted name-first compact checkpoint as the reference for this
experiment. Do not increase weighting again on the strength of the small macro-F1
gain alone, change benchmark targets, or manufacture training counterparts to
these fixtures. Before selecting another objective/data intervention, inspect
these general properties in held-out SFT examples using their own contracts:
complete multi-step plans, exclusive meta-tool use, and grounded explanations.
Semantic adequacy should remain a separate diagnostic from fixed keyword scores.

Only local experiment notes were changed. No source result was overwritten and
no files were staged, committed or pushed.
