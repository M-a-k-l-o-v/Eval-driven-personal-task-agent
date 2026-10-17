# Compact JSON: Are the Clarification Questions Useful?

**Status:** Completed manual diagnostic review, 2026-10-14.

## Scope and Evidence

Reviewed all ten rows in the compact-JSON result whose recorded decision was
CLARIFY and whose expected decision was CLARIFY. All ten are Calendar cases.
This is a selected subset, not an estimate of performance across all 150 cases
or other domains.

Source: [name-first-compact-combined-150.json](../../../results/phase2/json-spacing/name-first-compact-combined-150.json).
SHA256: `6ace952a70b3695737d373e80edb09b2b21505ac9a6c879a4ebb0a80b1bbf6aa`.

The coding agent inspected the saved request, initial state, tool schema and raw
question before inspecting the corresponding keyword requirements. This is not
an independent or blinded human study: the reviewer knows the experiment and
has inspected some related results before. The assessments below are qualitative
judgments, not new benchmark scores or an implemented LLM judge.

No inference, comparison script, retraining, rescoring, data rewrite, checkpoint
change or evaluation-target change was performed for this review.

## Review Criteria

- **Adequate:** Asks for the information or choice actually needed to resolve the
  blocker; does not have to repeat already-known names or time periods.
- **Incomplete:** Asks about a genuine blocker but leaves another required field
  unresolved. This can be a useful first question, not a complete clarification.
- **Wrong focus:** Asks only for known information without explaining why it
  needs reconsideration, and does not address the actual blocker.

Calendar creation requires title, date, time and duration in both the exposed
[tool schema](../../../src/baymax/eval/agent_runner.py) and
[adapter](../../../src/baymax/tools/fake_calendar.py). There is no default duration.
Duration omissions are therefore substantive, even when a scenario's keywords
only mention time. This review does not assume a follow-up conversation occurred.

## Ten Questions

### 1. Tonight's Chemistry Review

Scenario: `calendar_ambiguous_tonight_001`.
Request: "Schedule chemistry review tonight for 1 hour."
Question: "What time should I schedule the chemistry review?"

**Adequate.** The date and duration are known; exact start time is missing. The
question asks for precisely that. It need not repeat "tonight" to make sense in
this request's context. Keyword score: **0.5**, because "tonight" is absent.
This is a plausible keyword false negative for question adequacy.

### 2. Tomorrow's Revision

Scenario: `calendar_clarify_time_001`.
Request: "Schedule revision tomorrow."
Question: "What time should I schedule the revision?"

**Incomplete.** Asking for start time is useful, but duration is also unspecified.
Keyword score: **1.0**; recorded task success is true. The question is not enough
to ground every required creation argument, despite passing the current checks.

### 3. Calculus Conflicts With an Existing Event

Scenario: `calendar_conflict_001`.
Request specifies tomorrow at 4pm for one hour. State already contains a
two-hour event starting at 4pm on that date.
Question: "What time should I schedule the calculus revision?"

**Wrong focus.** The user already gave a time. Asking for an alternative could
help, but this question does not identify the conflict or explain why the given
time cannot simply be used. Keyword score: **0.0**. This is a substantive
state-grounding problem, not just missing synonyms.

### 4. Two Meetings With Marv

Scenario: `calendar_multiple_matching_meetings_001`.
Request: "Move my meeting with Marv to 5pm."
State contains two meetings with Marv, at 10am and 2pm.
Question: "Which event should I move?"

**Adequate.** It requests the missing entity selection and does not invent an ID.
Listing the two options would be clearer, but this question is understandable
in context. Keyword score: **0.5**, because "Marv" is absent. This is another
plausible keyword false negative for adequacy.

### 5. No Meeting With Tadashi

Scenario: `calendar_no_matching_reschedule_001`.
Request: "Move my meeting with Tadashi to 5pm."
State contains a meeting with Marv, but none with Tadashi.
Question: "What time should I move the Tadashi meeting to?"

**Wrong focus.** 5pm is already known; identifying the missing meeting is the
blocker. The question also presupposes a matching meeting exists. Keyword score:
**1.0**, because it includes "Tadashi" and "meeting"; recorded task success is
true. This is a keyword false positive for adequacy.

### 6. Supplier Call Without a Date

Scenario: `calendar_v2_ambiguous_missing_date_001`.
Request: "Schedule a supplier call at 2pm."
Question: "What time should I schedule the supplier call?"

**Wrong focus.** Time is known; date and duration are missing. The question asks
for neither. Keyword score: **0.0**. The output repeats the common time-question
pattern rather than identifying this request's missing fields.

### 7. Quarterly Planning Without a Time

Scenario: `calendar_v2_ambiguous_missing_time_001`.
Request: "Add quarterly planning on October 22."
Question: "What time should I add to my calendar?"

**Incomplete.** It asks for the missing time, although the wording is awkward.
Duration is also unspecified. Keyword score: **1.0**; recorded task success is
true. This is useful clarification, but not complete argument grounding.

### 8. Two Team Syncs

Scenario: `calendar_v2_contextual_multiple_sync_001`.
Request: "Move the team sync to noon."
State contains team syncs on October 13 and October 14.
Question: "Which team sync should I move?"

**Adequate.** It asks for the entity selection needed before updating an event.
The existing event supplies the remaining unchanged arguments. Keyword score:
**1.0**; recorded task success is true.

### 9. Around Lunch on Friday

Scenario: `calendar_v2_implicit_lunch_001`.
Request: "Put a mentor catch-up around lunch on Friday."
Question: "What time should I put it?"

**Incomplete.** Exact time is genuinely missing, so this is a useful question;
omitting "lunch" does not make it irrelevant. Duration is also missing. Keyword
score: **0.5**, because "lunch" is absent. There is both lexical brittleness and
a real completeness limitation.

### 10. Tomorrow Morning

Scenario: `calendar_v2_implicit_morning_001`.
Request: "Book reading time tomorrow morning."
Question: "What time should I book?"

**Incomplete.** It correctly asks for an exact start time, without inventing a
morning default. Duration is still unknown. Keyword score: **0.5**, because
"morning" is absent. Again, this is a useful but incomplete question.

## Findings

| Manual assessment | Cases |
| --- | ---: |
| Adequate for the identified blockers | 3 |
| Relevant but incomplete | 4 |
| Wrong focus | 3 |

All ten selected the correct route. That does not mean all ten asked a good
question. Conversely, a partial keyword score does not establish a bad question.
The saved keyword checks fully pass four of these cases, including one wrong-focus
case and two incomplete cases, while undercrediting two adequate questions.

The recurring issue is identifying **what is actually unknown or conflicting**:
eight of ten questions ask about time, including three cases where time is already
known. This is a descriptive pattern in a Calendar-only selected subset, not proof
that the model uses a single template or that other domains behave identically.

Keep routing, field/state grounding, question completeness and keyword agreement
separate. A bounded next diagnostic is to inspect these general properties on
held-out SFT questions against their own input schemas, before making a training
adjustment. Do not turn these evaluation fixtures into training examples, rewrite
their targets, or relax scoring to promote this checkpoint.
