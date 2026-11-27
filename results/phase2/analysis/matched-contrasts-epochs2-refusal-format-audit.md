# Matched-Contrast Two-Epoch Refusal and Format Audit

Captured on 2026-10-07 from saved results. No inference, rescoring, parser changes,
training-data changes or evaluation-fixture reads were performed.

The [case audit](matched-contrasts-epochs2-refusal-format-audit.json) records source
result hashes and result indices for all 29 expected REFUSE cases plus all 8
invalid matched outputs (31 distinct cases). All 300 saved generations from the
two runs were reparsed to confirm no parser drift.

## Measured Changes

| Measure | Anchored Reference | Matched Contrasts |
| --- | ---: | ---: |
| Task successes | 44/150 | 39/150 |
| Routing macro-F1 | 0.514 | 0.383 |
| Invalid outputs | 2/150 | 8/150 |
| Correct REFUSE routing | 16/29 | 5/29 |

The matched run lost 7 previously successful tasks and gained 2. It gained no
new correct REFUSE routing decisions.

Of the 16 reference REFUSE decisions, 5 remained REFUSE. The other 11 became:
6 productive ACT decisions, 2 valid CLARIFY decisions and 3 rejected outputs.
Those last 3 must not be treated as valid model decisions.

## What the Raw Outputs Show

This is not solely a keyword-scoring issue:

- A request to search emails produces the unavailable `gmail.search_messages`.
- A request to email with only a calendar capability produces
  `calendar.create_event` with email fields (`recipient`, `body`, `subject`).
- A request to recall an email produces an empty `gmail.send_email` call.
- Two unsupported requests instead produce valid clarification questions.

These observations indicate incorrect capability selection and routing. They do
not establish why training caused the change. Parser-valid productive calls can
still have invalid arguments or fail adapter execution; the invalid-output metric
is not a count of every possible tool-contract error.

Even the reference's correct REFUSE labels often have unsupported explanations,
such as invented authentication, ownership or terms-of-service restrictions.
Correct routing is not equivalent to a truthful reason or full task success.

## Eight Invalid Matched Outputs

Primary categories are exclusive and total 8. Diagnostic flags can overlap.

| Primary Cause | Count |
| --- | ---: |
| `request_clarification` uses `reason` instead of `question` | 2 |
| Missing clarification `question` | 1 |
| Missing refusal `reason` inside `arguments` | 2 |
| Productive calls omit tool names | 2 |
| Invalid JSON containing an ellipsis | 1 |

Six invalid records are REFUSE-expected and two are ACT-expected; no CLARIFY-expected
record was rejected. Both ACT records previously succeeded and now omit tool names.

One missing-reason case puts `reason` outside `arguments`; another supplies empty
arguments. One missing-question case also repeats the JSON `name` key. The existing
JSON parser uses the last duplicate key; this record is rejected for its meta-tool
arguments, not because duplicate keys are explicitly prohibited. That remains a
parser limitation, unchanged by this audit.

There are no mixed-meta/productive or repeated-meta-call failures in these eight
records. They are not the same format-error distribution as the four-epoch run.

All five parser-valid matched REFUSE decisions still have saved refusal keyword
accuracy below 1. This is a wording-score mismatch category, not an automatic
semantic judgment or authorization to manually improve scores.

## Training Composition Changed

Audited the first 1000 raw training rows against their source provenance. Both runs
contain 100 unique refusal rows, but their origins differ:

| Refusal Source Origin | Reference | Matched |
| --- | ---: | ---: |
| Triplets | 33 | 73 |
| Generated standalone examples | 22 | 0 |
| Capability-boundary pairs | 23 | 0 |
| Other contrasts or seed examples | 22 | 27 |

Source-ID families decreased from 45 to 20. These are authored lineages, not an
independently labeled count of semantic behavior types. In particular, a matched
triplet can keep a historical ID such as `refuse_gmail_search` while changing its
request to a draft request with no draft tool. Do not infer semantic search
coverage from the ID alone.

The 120 complete groups comprise 73 triplets and 47 pairs. Group-preserving
sampling prioritizes complete triplets, so sharper contrasts also changed the
within-class training composition. This experiment controls broad class counts,
xLAM rows and training settings, not identical BAYMAX coverage or target wording.

## Interpretation and Next Step

Measured conclusion: this matched variant is worse on the development benchmark.
Hypothesis: stronger concentration in selected triplet patterns reduced coverage
of other refusal situations; this is plausible, not demonstrated causation.

Keep the original anchored two-epoch reference. Before another data experiment,
audit training-only semantic subtype/domain coverage and preserve that breadth
when selecting complete contrasts. Any adjustment must be designed from training
data, adapter contracts and the general policy, not these individual eval cases.

Keep format validity separate from routing quality. Do not repair outputs, relax
the parser, change benchmark targets, add epochs or alter class quantities on the
strength of this audit alone. No new experiment has been prepared or run.
