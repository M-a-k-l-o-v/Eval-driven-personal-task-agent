# BAYMAX Scorer Fixes

**Status:** Proposed. The saved-output regrade pilot is ready for human review.

The current scorer checks whether an agent response matches a scenario fixture.
The fixes below address two cases where a match can mislead us: a question that
uses the expected words but misses the actual blocker, and a tool call that
matches the expected arguments but fails to produce the requested outcome.
The [current scoring reference](SCORING.md) documents the historical metrics.

## Current Scoring

[`score_response`](../src/baymax/eval/scorer.py#L70) compares the recorded
decision, tool calls, arguments, and message with `expected_behavior`. It checks
ACT calls by position and uses exact argument checks or `_contains` substrings.
For CLARIFY and REFUSE, it scores case-insensitive keyword matches and requires
zero productive calls. `task_success` requires full argument accuracy and an
empty failure list. The runner saves the fake adapter results and final state
alongside the score.

## 1. Grade the Blocker or Refusal Reason

The [Calendar audit](experiments/08-output-contract-learning/04-compact-json-clarification-question-review.md)
found errors in both directions. A question asking for the only missing start
time scored `0.5` because it omitted "tonight". A question asking for a time the
user had already supplied scored `1.0` even though the meeting itself was missing.
The scorer also uses `missing_clarification` or `missing_refusal` when a message
exists but matches only some of the keywords.

**Recommended fix.** Write the actual blocker or refusal reason into a separate
review rubric, derived from the request, tool contract, and initial state. For
clarification, specify the missing fields, conflicting facts, or entity choice
the question needs to resolve. For refusal, specify why the request cannot
proceed. Accept any wording that conveys the required meaning.

| Grade | What the reviewer should see |
| --- | --- |
| `2` | A question covers the required blockers, or a refusal explains the specific limitation. |
| `1` | A question resolves part of the blocker, or a refusal gives a relevant but incomplete explanation. |
| `0` | The message is missing, incorrect, or addresses the wrong issue. |
| `X` | Message adequacy is inapplicable because the output was invalid, took the wrong route, or attempted a productive call. |
| `?` | The rubric needs review before the response can be graded. |

Keep `X` in the routing or output failure counts. Report how many responses
qualified for wording review alongside the adequacy grades. Start with blinded
human review and compare disagreements before expanding coverage. ADR 0008 is no
longer deferred: Ronin will update the scorer himself, and the current scorer stays in use
until then. [Eval judge decision](decisions/0008-eval-judge-protocol.md)

**Acceptance checks.** Accept a correct question in different words. Give
partial credit when it asks for start time but leaves a required duration
unresolved. Reject a question that repeats the supplied time while ignoring the
missing event. For refusal, accept a specific explanation of unavailable email
sending and reject a request for a recipient when no send tool exists.

## 2. Verify ACT Outcomes

The [local runner](../src/baymax/eval/lora_runner.py#L139) records `tool_results`
and `final_state`. It passes only the scenario and parsed response to the scorer,
so a call can earn task success even if the adapter rejects it. In addition, a
no-argument fixture can award full argument accuracy despite unexpected fields.
The clipboard executor currently ignores arguments supplied to `clipboard.read`.
[Related clipboard bug](BUG_REPORT.md#bug-018--evaluation-silently-ignores-invalid-clipboard-read-arguments)

**Recommended fix.** Feed recorded execution evidence into versioned ACT
scoring. Validate arguments against the tool schema and check each required call
for a successful execution result. Give fixtures explicit outcome checks for
the expected state change or returned value. For a reschedule, check that the
specified event moved to the requested time and that unrelated state stayed
intact. For a clipboard read, check the returned text and unchanged state.
Report wrong arguments, failed execution, and unexpected effects individually.
Label a result without sufficient saved execution evidence as unverified.

**Acceptance checks.** Reject a matching call with a failed execution result, a
successful call that changes the wrong event, a read with undeclared arguments,
and a read that returns the wrong text. Accept a valid read with the expected
value and no state change.

## 3. Preserve the Decision Before Execution

The [routing report](../src/baymax/eval/routing_metrics.py#L14) currently infers
the expected class from which keyword metric is present. That coupling makes
message grading changes affect how routing results are interpreted. The core
adapter also derives the observed decision from the final response type, so a
rejected ACT plan can be reported as REFUSE when the runtime returns
`TASK_REFUSAL`. [Runtime decision bug](BUG_REPORT.md#bug-008--runtime-failures-are-mislabeled-as-model-refusal-decisions)

**Recommended fix.** Record the expected decision directly from the fixture and
record the selected model decision before validation and dispatch. Give parser
rejections, unknown decisions, and runtime failures explicit result fields.
Continue reporting macro-F1, per-class recall, and invalid-output rate. An
invalid or unknown prediction still counts as a missed expected class.

**Acceptance checks.** An ACT plan rejected during validation or dispatch still
records ACT as the selected decision and records the separate runtime failure.
An explicit model refusal remains REFUSE. Removing keyword scores leaves the
expected decision and its confusion-matrix row intact.

## Rollout and Verification

1. Finish the anonymous pilot. It contains ten saved responses across three
   clarification and two refusal cases. The automatic ACT pilot checks eight
   saved responses across four domains. The draft case still uses the original
   body substring check. The selected cases diagnose failure modes and cannot
   estimate how frequent those failures are across the full suite.
2. Agree on the rubrics and add regression cases for the observed false matches
   and misses. Use the existing scenario files and saved outputs as evidence.
3. Export the revised scoring under a new version with source-result and fixture
   identities. Preserve the original artifacts and historical scores. Compare
   both scoring versions on the same saved generations to measure the effect of
   the scorer change before using the new metric for model selection.
