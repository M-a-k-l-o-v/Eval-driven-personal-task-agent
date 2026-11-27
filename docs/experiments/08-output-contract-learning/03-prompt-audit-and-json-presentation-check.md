# SFT Versus Benchmark Input: Audit and JSON-Presentation Check

**Status:** Paired inference completed by the owner; full-suite spacing ablation prepared.

## Question

The name-first checkpoint clarified on 7/10 selected held-out SFT inputs, but on
0/40 expected-clarification cases in the established benchmark. Is a presentation
difference involved? Neither observation establishes the cause, and the two
sample sets are not matched tasks.

## Confirmed Audit Findings

The generic prompt-building code and saved ten-example probe were inspected.
No benchmark fixture was used to construct or rewrite a training example.

| Property | Finding |
| --- | --- |
| System prompt | All ten match the inference default and recorded training hash a31446de... |
| Header and request wrapper | Same header; same current_time / initial_state / request fields |
| Meta-tools | Shared definitions, identical names, schemas and leading order |
| Productive tool metadata | All ten match the generic benchmark definitions, descriptions and schemas |
| Tools in this sample | Every prompt contains two meta-tools and one productive tool |
| JSON serialization | SFT is compact with literal Unicode; benchmark uses default spaces and ASCII escaping |
| Time rendering | All ten SFT clocks retain 2026-06-01T09:00:00Z; benchmark code uses datetime.isoformat() |
| Request phrasing | Several successful SFT cases explicitly state that information has not been supplied |

For UTC, datetime.isoformat() can spell the same instant with +00:00 instead of Z.
The paired check below deliberately does **not** change this spelling. It also
does not vary tool counts, request wording, state content or metadata. Those are
separate potential distribution differences, not established explanations.

The three ACT mistakes in the probe involve an invalid due date, unavailable
checklist content and an ambiguous task reference. This is descriptive inspection
of held-out SFT rows, not permission to derive training targets from benchmark
failures or evidence for a general lexical-rule diagnosis.

## One Controlled Inference Check

The existing probe now supports `--benchmark-json`. It selects the **same ten**
validation rows in the same order and changes only JSON spacing/escaping to the
existing benchmark renderer's conventions. Decoded requests, tools, schemas,
state, clock strings and expected targets are unchanged.

The saved original-SFT report is required. Preflight rejects a different selection,
config, dataset, adapter hash or generation setting. The expected assistant answer
is never included in either model prompt. Model loading, greedy decoding, 256
new tokens and strict parsing are unchanged. No productive tool is executed.

Run from the repository root:

```powershell
.venv\Scripts\python scripts\probe_sft_clarification.py `
  --benchmark-json `
  --reference-report tmp\name-first-heldout-clarification-10.json `
  --output tmp\name-first-heldout-clarification-benchmark-json-10.json
```

The owner completed both probes. Saved reports have matching config, dataset,
adapter-weight hashes and generation settings. Inspection confirmed identical
decoded input values, tool order, system messages and expected targets. All ten
inputs are ASCII, so this pair changed JSON separator spacing only.

| Input presentation | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| Exact compact SFT inputs | 3 | 7 | 0 | 0 |
| Benchmark JSON spacing | 10 | 0 | 0 | 0 |

Raw outputs confirm all seven previous clarification decisions changed to
productive calls; none of the new outputs mention request_clarification. For
example, row 25 switched from asking for an event time to supplying 21:00 and a
30-minute duration. These are routing observations, not semantic question scores.
Reports remain under ignored tmp/ and existing files were not overwritten.

This supports substantial presentation sensitivity on these ten selected inputs.
It does not establish its magnitude across the benchmark or robust policy learning.

## Full-Suite Spacing Ablation

The next owner-run diagnostic evaluates the same name-first checkpoint on the
unchanged 150 scenarios from scenarios/v1 and scenarios/v2. Use --compact-json
with the existing run-agent-lora command and a new output file:

```powershell
$adapter = "models\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first\adapter"
.venv\Scripts\python -m baymax.eval.cli run-agent-lora `
  --scenarios scenarios\v1 `
  --scenarios scenarios\v2 `
  --adapter $adapter `
  --compact-json `
  --output results\phase2\json-spacing\name-first-compact-combined-150.json
```

Only separators in the available-tool and request-context JSON change. Unicode
escaping, timestamp spelling, input values, key/tool order, system prompt,
checkpoint, greedy decoding, 256-token generation cap, parser, fake execution and
scoring are unchanged. In particular, the renderer does not convert +00:00 to Z.
Literal Unicode versus ASCII escaping is not part of this ablation.

The default renderer remains spaced. Local result metadata records the format and
inference configuration. Compact mode refuses to overwrite an existing output.
No retraining, target changes or rescoring of historical artifacts is involved.
The full-suite result is pending; the coding agent ran only mocked unit tests.
This repeatedly inspected suite is development evidence, not an untouched final
test set. Compare routing macro-F1 and confusion matrices separately from keyword
wording scores and task success.

## Semantic Review Is Separate

Routing, response adequacy and keyword agreement must not be conflated. A valid
clarification call can ask the wrong question; a useful question can use wording
not recognized by keyword checks.

The proposed next measurement is a small blinded human review of boundary
responses, using request/context and the general policy rather than visible
keyword scores. It would report adequacy separately and preserve official scores.
This audit does not implement that review, an LLM judge, or new scoring rules.
The coding agent's unblinded inspection is not independent human validation.
