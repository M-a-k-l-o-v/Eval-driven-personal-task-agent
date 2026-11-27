# BAYMAX Training Data Contract Audit

Date: 2026-10-05. Generator version: 12 -> 13.

## Scope

Checked all 1,128 current BAYMAX synthetic examples against the actual fake Calendar, Notion, Gmail, and clipboard adapters. Checked target arguments against the shared productive/meta-tool schemas and executed all 388 productive targets using fresh in-memory adapters.

No evaluation fixtures were read or changed. No xLAM data, inference prompt, scorer, or benchmark artifact was modified.

This is an exhaustive mechanical contract check plus review of shared authoring patterns. It is not an automated semantic judge or a claim that every example's policy interpretation is resolved.

## Defects Fixed

| Defect | Examples affected | Correction |
| --- | ---: | --- |
| Plain-string clipboard initial state | 188 | Author state as `{"clipboard": {"text": "..."}}`, matching ClipboardState. |
| Incomplete existing calendar events | 17 | Add explicit fictional date/time/duration to the authored event records, preserving multiple candidate entities. |
| Questions gathering only some missing calendar details | 43 | Ask for start time and duration; invalid-date requests also ask for a valid date. |

These are 248 distinct corrected examples: 205 state changes and 43 question changes.

Every user request, behavior label, available-tool list, example ID, contrast group, and contrast role stayed unchanged. All ACT and REFUSE targets stayed unchanged. In particular, invalid dates, negative durations, unknown IDs, empty selections, and missing-information requests were not repaired into ACT requests.

Examples of preserved requests:
- `Schedule ... on 2026-02-30.` still targets clarification, now asking for a valid date, start time, and duration.
- `Set the duration of evt_briefing to minus 20 minutes.` remains a clarification example.
- `Move my meeting to 3pm.` retains two candidate events and still asks which meeting.

The builder now rejects incompatible initial state or unexecutable productive targets before creating output files, identifying the source example ID. Meta-tool targets must be a single correctly shaped clarification/refusal call, not mixed with productive tools. Regression tests check every generated target against the shared JSON schemas.

## Policy Questions Identified in V13

These questions were left unchanged in the v13 artifact. The owner-approved resolutions and v14 changes are recorded below.

1. **Bulk outreach and consent (41 examples).** Some targets refuse "bulk unsolicited email" when requests only say "every contact" or "every address." Consent is not established. Decide whether indiscriminate outreach itself is categorically refused; otherwise the reason assumes a fact not supplied.
   Example: `baymax_contrast_refuse_spam_004` asks to send a promo to every contact.
2. **Email versus draft (1 example).** `baymax_clarify_multi_step_001` asks "Create the task and email Marv about it," but offers only notion.create_task and gmail.create_draft. If email means send, the capability is unavailable and refusal should precede clarification. Decide the intended interpretation.
3. **Composing drafts (12 examples).** `baymax_triplet_clarify_gmail_draft_*` supplies a topic and deadline but targets "What should the draft say?" Decide whether routine drafting may compose wording from supplied facts or requires exact body content. These may be unnecessary clarification examples under a composition policy.

The full local audit records all affected IDs and before/after values:
`data/sft/reviews/adapter-contracts-v13/audit.json`.

## New Artifact

Generated a separate local dataset:
`data/sft/v1/baymax-control-adapter-contracts-v13/`

- Train: 790; validation: 225; test: 113.
- Validation report: valid; 1,128 converted; zero skipped.
- Revision: `0874c55778f123f570d4a34728f9a15b05d7d8536dab6c3f99573cf8cf2b80ba`.

The old v12 dataset remains untouched. The active training configuration and mixed xLAM artifact were not changed. The v13 artifact remains the historical mechanical-fix version; use the separately versioned v14 artifact for the resolved policy examples.

## Verification and Limits

- All 1,128 initial states now load through FakeToolExecutor.
- Every target passes shared argument-schema checks; all 388 productive target sequences execute successfully.
- 45 targeted tests passed across synthetic generation, mixing, and dataset validation.
- Scoped Ruff lint/format and Pyright checks passed.
- Split content remains disjoint; contrast groups stay in one partition.
- All 1,128 saved rows in both v12 and v13 match their audited source examples; no example changed its train/validation/test partition.

These checks do not prove an ACT target is semantically grounded or a CLARIFY/REFUSE label is correct. Adapter date/time strings are not a full calendar semantics validator. No inference, training, benchmark improvement claim, or long-running experiment was performed.

Full repository tests and full-repository type/lint checks were not run; verification was scoped to keep commands short.

## Owner-Approved Policy Follow-Up: V14

The project owner resolved the three questions on 2026-10-05:

| Situation | Decision |
| --- | --- |
| Indiscriminate outreach to every contact/address that can be found | REFUSE. Describe indiscriminate outreach, not an assumed lack of consent. Messages to explicitly identified recipients/groups are not refused merely because they are group messages; the actual recipients and other necessary facts must still be known. |
| User requests sending, but only a draft tool is available | REFUSE the unsupported send request. Creating a draft is not equivalent to sending; refusal takes precedence over clarification about another missing detail. |
| User requests a draft and supplies enough factual content | ACT by composing wording from those facts. Do not invent statuses, commitments, dates, or other facts. CLARIFY when a fact requested for the draft is genuinely missing. |

Generator version 14 changes 66 examples relative to v13:

- 41 bulk refusal reasons now describe indiscriminate outreach without asserting that recipients did not consent.
- One seed request is reclassified from clarification to refusal and renamed `baymax_refuse_multi_step_send_unavailable_001`. It still asks to create a task and email Marv, with only a draft capability available.
- Twelve draft triplets now teach composition from a supplied topic/deadline in the ACT variant. Their CLARIFY variants explicitly request an agreed deadline that is not supplied, so the missing-information distinction remains genuine. This changes 24 examples without dropping the ACT/CLARIFY/REFUSE triplet structure.

No intentionally invalid-date, negative-duration, unknown-ID, or empty-selection user requests were made valid. No evaluation fixtures were used or changed.

New local artifact: `data/sft/v1/baymax-control-policy-v14/`.

- 1,128 examples; train 790, validation 225, test 113; validation report valid, zero skipped.
- Behavior counts: ACT single 319, ACT multi 69, CLARIFY 346, REFUSE 394.
- Revision: `51c2aa286b9c6ffecf227e5de358f362aeddb336676122d0ead35e11a5029951`.
- All adapter/schema checks and 30 generator tests pass; scoped Ruff and Pyright checks pass.

V12/v13 data, historical checkpoints/results, inference/training prompts, and the active training configuration remain untouched. This updates training examples, not a runtime policy enforcement engine. No retraining or evaluation inference was run.

## Read-Only Numeric Audit

`scripts/audit_sft_numeric_arguments.py` scans converted local SFT datasets with a
`split-manifest.json`, preserving source IDs, split names, file paths, line numbers,
and example hashes. It needs no Hugging Face authentication or model loading.

```powershell
.venv\Scripts\python scripts\audit_sft_numeric_arguments.py `
  --dataset-dir data\sft\v1\xlam-content-groups-v1 `
  --output data\sft\reviews\xlam-numeric-audit.json
```

The report distinguishes invalid scalar integer targets from possible decimal-to-integer
conversions in user requests. For example, a request containing `261.63` with an integer
target of `261` is a review candidate, not an automatically established defect.
Findings are grouped by issue, tool, and argument to help identify repeated patterns.
Explicit rounding language and other occurrences of the target number are review hints,
not grounds for silently removing a finding.

Malformed rows are reported separately. Missing files or inconsistent manifest counts
fail explicitly. The script never changes dataset rows or overwrites an existing report;
choose a fresh output path. Keep reports under the ignored local data directory because
they include request contents.

Use `--max-rows 1000` for a short smoke audit. This scans a prefix in
train/validation/test order, not a representative sample, and marks capped coverage as
partial. Only supplied scalar integer arguments are inspected: this does not validate
all argument types, defaults, calculations, unit conversions, or semantic correctness.
Review candidates before proposing any versioned dataset exclusion or correction.

## Broad xLAM Sample Review

Following the complete numeric scan (60,000 rows, 1,069 flagged rows, zero row parse
errors), manually inspected 100 unique examples from the pinned local xLAM training
split. Seed 42: 50 numeric-flagged rows spread across issue/tool groups, plus 25
unflagged single-call and 25 unflagged multi-call examples. No evaluation fixtures
were consulted.

| Review outcome | Flagged sample | Unflagged sample |
| --- | ---: | ---: |
| Target or prompt contract inconsistency | 40 | 3 |
| Metadata-only defect | 3 | 12 |
| Needs additional context | 4 | 3 |
| No issue found | 3 | 32 |

Found non-numeric issues including invented missing Sudoku rows, invented page IDs,
unresolved inter-tool result placeholders, swapped coordinates, silently substituted
units, invalid or contradictory defaults, and a login target contradicting the actual
injected password-refusal prompt. Many numeric type inconsistencies originate in the
tool schemas rather than an unfaithful target; do not fix those by changing user values.

The deliberately enriched/stratified sample cannot estimate the dataset's overall
defect rate. Some numeric flags match unrelated numbers; optional/null semantics and
external entity/enum mappings require context. No upstream tools were executed.

Local evidence: `data/sft/reviews/xlam-broad-review-100/README.md`, `samples.json`, and
`review.json`. Every selected row has provenance, a content hash and a review note.
Source data, training configuration, prompts, checkpoints and benchmarks are unchanged.
This records an inspection, not a filtering decision or a causal explanation of model
performance.

## Credential Policy Reconciliation

Status: owner-approved policy on 2026-10-05; read-only contract audit implemented.
A separately versioned, reviewed credential filter is now implemented (see below).
Switching future mixes/configs to that artifact remains a separate step.
No prompt or target has been rewritten by this reconciliation.

### Existing Decisions

- The canonical decision policy refuses unsafe actions and credential disclosure
  (`docs/BENCHMARKING.md`, Canonical Decision Policy).
- Authentication belongs in adapters, with credentials read from environment variables
  and never exposed in returned data or traces (ADR 0002, section 6; ADR 0010).
- Current synthetic examples refuse retrieving, copying or emailing saved passwords.
  They also allow drafting a password-reset message and copying supplied password
  guidance. The word "password" alone is therefore not a refusal condition.
- Missing inputs must not be invented. For a supported safe action, ask for the missing
  non-sensitive details; unavailable or unsafe actions take refusal precedence.

| Situation | Established handling |
| --- | --- |
| Draft a supplied password-reset message or copy non-sensitive password guidance | Apply ordinary ACT/CLARIFY rules for the available tool; do not refuse solely because the topic mentions passwords. |
| Retrieve, copy or disclose an actual saved password | REFUSE under existing synthetic-data policy. |
| Authenticate a supported adapter using its configured environment credentials | Adapter responsibility; the model does not need the secret as a generated argument. |
| Invent credentials or session/signature values missing from request, state and defaults | Invalid supervision, regardless of whether login is otherwise permitted. Do not repair by fabricating values. |
| External tool requires the model to emit a user-supplied password or authentication token | Exclude the confirmed conflicting example from future BAYMAX training mixes. Keep the external source unchanged; do not invent a replacement target. |

### Confirmed Mismatch

The actual inspected xLAM prompt says to refuse "secrets/passwords". Review R078
(source ID 50853) instead targets a login call containing a supplied plaintext
password. This is contradictory supervision under that injected instruction. It
does not establish that user-authorized login is inherently unsafe in all systems.

The wording is also too broad to clearly distinguish a harmless password-reset
message from handling a secret. Any later wording change should make that distinction
explicit, remain aligned across training/inference/meta-tool descriptions, and receive
a new recorded prompt identity. Historical artifacts must not be overwritten.

### Approved Mix Decision: Enforcement Pending

Owner decision: keep credentials outside model-generated arguments. Exclude confirmed
credential-emitting examples from a future separately versioned BAYMAX training mix,
while preserving the external source and split assignments. Do not replace their
targets with invented refusal answers. Do not exclude harmless guidance, content
objects named "token", or identifiers merely on a keyword match.

Alternative not selected: explicitly support user-supplied login credentials. This requires an
owner-approved policy/contract change and aligned supervision before retaining these
examples as ACT targets; it must not be introduced by silently weakening a refusal
prompt. No implementation of this alternative is authorized here.

This decision applies to incompatible supervision in the BAYMAX mix, not a claim
that generic xLAM domains are intrinsically invalid. External metadata defects,
unknown optional/null semantics and unresolved dependencies remain separate audit
issues; they should not be reclassified as safety refusals just to make targets fit.

## Contract Audit and Exclusion Proposals

The existing audit CLI now supports `--checks contracts` through the pure helper
`src/baymax/models/sft_contract_audit.py`. Numeric mode remains the default.

```powershell
.venv\Scripts\python scripts\audit_sft_numeric_arguments.py `
  --checks contracts `
  --dataset-dir data\sft\v1\xlam-content-groups-v1 `
  --output data\sft\reviews\xlam-contract-audit.json
```

The contract report contains two distinct dispositions:

- `review`: declared target/default type mismatches, literal prose-default conflicts,
  unavailable tools, undeclared arguments, duplicate/ambiguous tool names, uncertain
  nulls, unsupported type expressions, ambiguous credential fields and possible
  unevaluated numeric/container arguments. These are not removal decisions.
- `proposed_exclusion`: a target explicitly emits a nonempty password or other
  authentication credential, identified by an explicit field name or clear parameter
  description. Generic token/key identifiers and harmless guidance are not blanket
  exclusions. Nested credential leaves are inspected as well.

`proposed_exclusions` coalesces each affected row with its split, source ID/index,
line, path, content hash and reason. It does not apply a filter. Contract reports omit
raw user requests, argument/default values and full schemas, including on row errors.
Numeric mode retains its previous evidence-oriented output and should stay local.

Supported basic type checks cover scalar Python/JSON types and Python-style typed
list, tuple, dictionary, optional and union expressions. This is not a full JSON
Schema validator: required fields, nested JSON Schema constraints, formats, enums,
bounds, consent and factual grounding are not comprehensively checked. Unknown types
are reported as unknown coverage, not accepted silently. Duplicate-name target calls
cannot be bound to a unique schema; their target checks are explicitly skipped and
flagged while unambiguous tools in the same row remain checked.

Prose-default checking recognizes only explicit literal "default is/to" wording,
not arbitrary natural-language defaults or unit conversions. Placeholder checking
does not interpret code or bind previous results. String payloads that legitimately
contain code/placeholder text are not treated as executable arguments. The separate
numeric audit is still needed for possible silent rounding. Invented Sudoku rows or
other missing facts still require semantic review.

Verification: 76 focused tests passed across contract and numeric auditing, with
scoped Ruff format/lint and Pyright checks. The final capped smoke report is
`data/sft/reviews/xlam-contract-smoke-1000-v2.json`: 1,000 rows scanned, 527 flagged
rows, 9 proposed credential exclusions, zero row-processing errors, partial coverage.
Most flags concern metadata, not proven wrong assistant decisions. The earlier smoke
report is retained; duplicate tool names initially surfaced as row errors and are now
explicit ambiguity findings instead.

At that checkpoint, no full-dataset scan, filtering, training or evaluation inference
had been run. The subsequent owner-run scan and reviewed filtering are recorded below.
No dependency was added. Full-repository checks were not run to respect the owner's
restriction on long commands; verification was scoped to the affected code.

## Reviewed Credential Filter v1

Completed on 2026-10-05 in three steps:

1. Reviewed the 618 exclusion proposals by 42 tool/parameter/schema groups. The
   descriptions identify login passwords, API keys, authorization/client tokens,
   or actual passwords submitted for analysis. All proposals were confirmed under
   the owner-approved adapter-managed authentication/secret-handling policy. This
   was a schema-group policy review, not a full semantic review of 618 requests.
2. Inspected the two row-processing errors independently. Train source IDs 7354
   (line 26790) and 16055 (line 29009) have `flatten_list` arguments containing
   integer lists nested nine levels deep. They exceed the audit's eight-level
   coverage limit; they are not malformed JSON. Neither has credential fields.
   Both are retained, with the audit coverage limitation explicitly recorded.
3. Created `data/sft/v1/xlam-content-groups-auth-filtered-v1/` using the new
   `scripts/filter_sft_dataset.py` / `src/baymax/models/sft_filter.py` path.

| Split | Parent rows | Excluded | Retained |
| --- | ---: | ---: | ---: |
| train | 42000 | 425 | 41575 |
| validation | 12000 | 141 | 11859 |
| test | 6000 | 52 | 5948 |
| total | 60000 | 618 | 59382 |

Review ledger: `data/sft/reviews/xlam-credential-filter-review-v1.json`.
It records approved row locations, source IDs/indices, canonical content hashes,
schema-group hashes/reasons and the two retained coverage gaps, without credential
values. The derived artifact includes copies of this ledger, `filter-provenance.json`
and an updated `split-manifest.json`. Its revision is:

```text
33cd2b207c89b907288a562d496dde3e8f2e4ba6d87f9b3cc5083ea13fba3d3b
```

The filter requires a complete, hash-pinned contract audit and explicit row
approvals. It verifies selected row hashes and manifest provenance, refuses existing
or overlapping output directories, and publishes staged output only after all selected
rows verify. Retained JSONL bytes, row order, split membership, source references,
seed, prompt identity and original source revision (in provenance) are preserved.
The derived revision/config hash are new; intended split ratios are inherited rather
than rebalancing after removal. The existing skipped-row ledger gains the exclusions
so loaded/converted/skipped counts remain consistent with the existing validator.

Actual artifact verification compared every retained row byte-for-byte with the
parent, checked retained source references and input/output file hashes, reconciled
the 618 removals, and confirmed both deep-list rows remain. Source files and parent
manifest are unchanged. Dataset/review files remain local and git-ignored.

Windows permission correction: the original `TemporaryDirectory` staging folder
had protected, owner-only ACLs. Renaming it published those permissions, preventing
the owner's account from reading the artifact created by the sandbox account.
Existing artifact ACLs were reset to inherit from the dataset parent directory;
bytes and artifact identity are unchanged. Staging now uses a uniquely named normal
directory that inherits project permissions, with bounded cleanup on failure.
All 26 filter tests passed, including a Windows-specific published-ACL regression.

The remaining review flags, numeric-rounding candidates and semantic defects are
not removed or repaired. This is a policy-filtered artifact, **not** a certification
that the remaining dataset is clean. Training configs, mixes, prompts, evaluation
fixtures and historical checkpoints/results were not changed; no training or model
evaluation was run.

The filter has focused regression tests for preservation, provenance/hash failures,
explicit approval, coverage-gap handling, no overwriting and deterministic identity.
Full-repository checks and the full-dataset validator are left to the owner to avoid
long commands. Before using the new artifact in a mix, run:

```powershell
.venv\Scripts\python scripts\validate_sft_dataset.py `
  --dataset-dir data\sft\v1\xlam-content-groups-auth-filtered-v1
```

## Next Mix: Filtered xLAM + Corrected BAYMAX V14

The owner ran full filtered-dataset validation: valid, 59,382 converted, 618
skipped. The sole top-level issue is the expected skipped-row warning.

New training config: `configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14.json`.
New mixed artifact destination: `data/sft/v1/xlam-baymax-auth-filtered-v14/`.
Neither replaces a historical config or dataset. The mix has not been built yet;
its expected revision is computed from the pinned input manifests and the existing
mix revision function, not from an invented training result. Validate config/manifest
identity after the owner builds it.

Inputs are the validated auth-filtered xLAM artifact and the already-corrected,
validated `baymax-control-policy-v14` pool. Seed 1, shuffle enabled, one copy per
input row, zero extra clarification/refusal repeats. For the 1,000-row train prefix,
use 450 unique BAYMAX rows: 150 clarification, 150 refusal, 150 productive;
fill the remaining 550 slots with unique xLAM rows. The existing sampler selects
unique rows and preserves input split membership. These are pre-tokenization quotas,
not measured post-truncation counts. Existing validation/test sampling stays unchanged.

Base model, LoRA settings, training hyperparameters and prompt hash match the previous
content-groups-v12 config. Model output/adapter paths are separately versioned.
Because v14 also includes approved BAYMAX contract/policy corrections, comparison
with the old v12 checkpoint is an updated-data comparison, not a credential-filter-only
ablation. Other xLAM audit findings remain unresolved. No training/evaluation scenarios
were read or used to prepare this configuration.

Build only once into the fresh output directory:

```powershell
$mix = "data\sft\v1\xlam-baymax-auth-filtered-v14"
if (Test-Path -LiteralPath $mix) { throw "Mix output already exists; do not overwrite it." }
.venv\Scripts\python scripts\mix_sft_datasets.py `
  --xlam-dir data\sft\v1\xlam-content-groups-auth-filtered-v1 `
  --baymax-dir data\sft\v1\baymax-control-policy-v14 `
  --output-dir $mix `
  --seed 1 `
  --train-cap-row-count 1000 `
  --train-cap-baymax-row-count 450 `
  --train-cap-baymax-clarification-count 150 `
  --train-cap-baymax-refusal-count 150
```

Then validate the mixed dataset and the new config with `--check-paths`, and inspect
both the 1,000-row train artifact and the 200-row validation artifact at sequence
length 1024 before retraining. The validation cap can still have sparse boundary
coverage; report it rather than assuming the validation loss represents all classes.
No mix build, tokenization report, training, model inference or full-repository checks
were run by the coding agent during this setup.

## Representative Validation Cap

The owner-built auth-filtered-v14 train artifact has 976 usable rows, 150 unique
clarification and 150 unique refusal rows, no exact duplicates. The existing
200-row validation prefix has 191 usable rows but only one refusal and no clarification
rows. Its aggregate loss therefore cannot meaningfully monitor boundary behavior.

The mixer now supports opt-in `--validation-cap-row-count` and three validation
BAYMAX quota options. For a 200-row cap, the new recipe selects 30 unique clarification,
30 unique refusal, 30 unique productive BAYMAX and 110 unique xLAM rows from the
existing validation partition only. This mirrors the train prefix's source/class
proportions. It deterministically shuffles the selected prefix, then appends every
unselected validation row in its previous order. No row is relabeled, repeated,
discarded or moved between splits. Quota shortages fail explicitly. Quotas describe
the raw prefix, not guaranteed post-tokenization counts.

The training sampler, train/test bytes and source references are unchanged. Opt-in
validation settings receive a versioned mix hash; disabled validation sampling keeps
the historical hash unchanged. The LoRA/training hyperparameters, prompt, seed and
base model are unchanged. The current trainer does not enable early stopping or
best-checkpoint selection; the cosine schedule is not driven by validation loss.
Later use of those mechanisms must record the new validation identity. Losses on
this balanced slice should not be directly compared to losses on the old sparse slice.

New output: `data/sft/v1/xlam-baymax-auth-filtered-v14-val200/`.
New config: `configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_val200.json`.
Historical artifacts/configs remain untouched. The new config's expected revision
is computed from the pinned input manifests and sampling recipe; validate it against
the generated manifest after building.

```powershell
$mix = "data\sft\v1\xlam-baymax-auth-filtered-v14-val200"
$config = "configs\sft\qwen2_5_1_5b_lora_auth_filtered_v14_val200.json"
if (Test-Path -LiteralPath $mix) { throw "Output already exists; do not overwrite it." }
.venv\Scripts\python scripts\mix_sft_datasets.py `
  --xlam-dir data\sft\v1\xlam-content-groups-auth-filtered-v1 `
  --baymax-dir data\sft\v1\baymax-control-policy-v14 `
  --output-dir $mix `
  --seed 1 `
  --train-cap-row-count 1000 `
  --train-cap-baymax-row-count 450 `
  --train-cap-baymax-clarification-count 150 `
  --train-cap-baymax-refusal-count 150 `
  --validation-cap-row-count 200 `
  --validation-cap-baymax-row-count 90 `
  --validation-cap-baymax-clarification-count 30 `
  --validation-cap-baymax-refusal-count 30
.venv\Scripts\python scripts\validate_sft_dataset.py --dataset-dir $mix
.venv\Scripts\python scripts\validate_training_config.py --config $config --check-paths
.venv\Scripts\python scripts\report_sft_artifact.py `
  --config $config --split validation --max-rows 200 --max-sequence-length 1024
```

Focused tests cover class/source quotas, unique selection, deterministic output,
preservation of complete validation contents and split membership, byte-identical
train/test output, unchanged disabled-sampler identity, invalid quotas and shortages.
No full mix rebuild, tokenizer report, training or benchmark inference was run by
the coding agent; full-repository checks remain deferred under the long-command rule.

## ACT-200 Mixture Ablation

New config: `configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_act200_val200.json`.
New dataset destination: `data/sft/v1/xlam-baymax-auth-filtered-v14-act200-val200/`.

Reuse the same auth-filtered xLAM and corrected BAYMAX v14 source artifacts. At a
1,000-row training cap, select 400 unique BAYMAX examples: 100 clarification,
100 refusal, and 200 productive; fill the remaining 600 slots with unique xLAM.
Use seed 1, shuffle enabled, one copy per input row, and no extra repeats.
Keep the validation cap at 200 rows: 30 clarification, 30 refusal, 30 productive
BAYMAX, and 110 xLAM. Do not change source split membership or the test split.

The expected mixed revision is
`d8249989ac5289dd80865e9cb6be4c35bbde98a274933440b21f06770d739225`.
This is computed from the pinned source manifests and recipe, not proof that the
new dataset was built. Validate the generated manifest and inspect the actual
post-tokenization artifact before training. Keep base model, prompt, seed, LoRA,
epochs, sequence length, batch sizes, and generation settings fixed. Compare
primarily against the v14-val200 run; act-balanced is a historical reference,
not a single-variable control. Preserve all previous datasets and results.

## Matched Contrasts At ACT-200 Counts

Opt-in `--matched-contrasts` tightens the existing 100 triplets, without reading
evaluation fixtures. Contact lookup and duplicate-entity triplets share the exact
request but change state; calendar and task examples remove only the required time
or due date; capability contrasts share request/state but remove the needed tool.
Bulk outreach and password safety contrasts remain represented. Existing pairs,
other examples, prompt and disabled-generator behavior are preserved.

The builder exports `contrast-groups.json`, bound to the new source revision.
`--preserve-contrast-groups` prioritizes complete triplets and pairs, keeps every
selected grouped boundary example with its counterparts, and fills remaining
productive slots with standalone examples. Ungrouped seed examples can also fill
boundary quotas. Not every selected productive example needs a boundary counterpart.
Selection uses the existing seeded order, excludes exact duplicates, never borrows
from other splits, and fails if quotas cannot be satisfied. The group metadata is
hashed into the new mix identity. Opt-in builds/mixes reject nonempty destinations.

Use the same raw cap: 100 clarification, 100 refusal, 200 productive BAYMAX and
600 xLAM rows. Validation still selects 30/30/30 BAYMAX plus 110 xLAM. This is a
new synthetic source revision, including revised validation/test examples; do not
interpret validation losses as a same-artifact comparison with ACT200. Whole
contrast groups and connected duplicate content stay within one source partition.
The 150-scenario benchmark, parser, scoring rules, prompt, seed, one epoch and LoRA
hyperparameters are unchanged. The benchmark is a repeatedly used development
benchmark, not an untouched final test set.

New config: `configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_matched_act200_val200.json`.
Expected mix revision: `961023206f87a928d2c5b9881e5c03984d560e63ef31a28fcf8c89c9464a9f40`.
This identity is computed from pinned inputs, not evidence of a completed real build.

### Output Contract Gate

Dataset validation and training preflight now reject unknown tool names, missing
call fields, wrong declared argument types/constraints, and invalid meta-tool fields.
`request_clarification` requires only a nonempty `question`; `refuse_request`
requires only a nonempty `reason`. Neither can coexist with productive calls.
Shared meta-tool definitions must match; no target repair is performed.

The separate contract audit checks explicit credential keys even when a target
tool is unavailable or has duplicate definitions. It retains the schema ambiguity
finding and does not select an arbitrary definition. Re-run the audit before
approving new credential exclusions; previously filtered artifacts remain unchanged.

Standard JSON Schema and native xLAM parameter maps are supported. Native maps
enforce explicit requiredness, not requirements guessed from prose. Unverifiable
types and schema references fail closed for review. Errors identify file/line/call,
not argument values. Contract validity does not prove correct intent, factual
grounding, or policy compliance. The gate does not replace the separate data audits.

Before training, run the capped dry-run below. A whole-pool validation failure may
also describe unused tail rows: inspect its report, do not silently drop rows or
disable the gate. The actual selected train and validation rows must all pass.
Check post-tokenization reports too: raw class quotas do not guarantee trainable
label counts or that every counterpart survives truncation.

```powershell
$source = "data\sft\v1\baymax-control-policy-v14-matched"
$mix = "data\sft\v1\xlam-baymax-auth-filtered-v14-matched-act200-val200"
$config = "configs\sft\qwen2_5_1_5b_lora_auth_filtered_v14_matched_act200_val200.json"
.venv\Scripts\python scripts\build_baymax_sft_dataset.py `
  --output-dir $source --seed 1 --matched-contrasts
.venv\Scripts\python scripts\mix_sft_datasets.py `
  --xlam-dir data\sft\v1\xlam-content-groups-auth-filtered-v1 `
  --baymax-dir $source --output-dir $mix --seed 1 `
  --train-cap-row-count 1000 --train-cap-baymax-row-count 400 `
  --train-cap-baymax-clarification-count 100 --train-cap-baymax-refusal-count 100 `
  --validation-cap-row-count 200 --preserve-contrast-groups
.venv\Scripts\python scripts\validate_sft_dataset.py --dataset-dir $mix
.venv\Scripts\python scripts\validate_training_config.py --config $config --check-paths
.venv\Scripts\python scripts\train_sft_lora.py `
  --config $config --dry-run --local-low-vram `
  --max-train-rows 1000 --max-eval-rows 200 --max-sequence-length 1024
.venv\Scripts\python scripts\report_sft_artifact.py `
  --config $config --split train --max-rows 1000 --max-sequence-length 1024
.venv\Scripts\python scripts\report_sft_artifact.py `
  --config $config --split validation --max-rows 200 --max-sequence-length 1024
```

Run these commands individually and stop on an error. Training and evaluation
are owner-run steps after preflight and artifact review, not completed results.

## Reviewed Target-Compatibility Filtering

Owner-approved direction on 2026-10-06: exclude examples confirmed incompatible
with the training target contract, not every broad audit flag. Preserve the source,
historical mixes and results; do not coerce values or invent repaired targets.

The existing audit CLI supports `--checks targets`. It reuses the exact validator
called by training preflight and adds explicit credential checks. It proposes whole
rows with invalid call/meta-tool shapes, unavailable or duplicate tool names,
undeclared arguments, target type/constraint violations, or unverifiable contracts.
These are pipeline-compatibility exclusions, not claims that every upstream target
is semantically wrong. Default-only metadata warnings do not propose removals.
Numeric rounding, factual grounding and policy semantics remain separate reviews.

```powershell
.venv\Scripts\python scripts\audit_sft_numeric_arguments.py `
  --checks targets `
  --dataset-dir data\sft\v1\xlam-content-groups-auth-filtered-v1 `
  --output data\sft\reviews\xlam-target-compatibility-audit-v1.json
```

The audit records per-row source provenance, content hashes, whole-file split hashes,
exclusion reasons and redacted validator messages. It never removes rows. A full
scan must be owner-run;
capped reports are partial and cannot authorize a filter. Inspect the resulting
counts and reasons before approving removals. Credential inspection errors remain
separate coverage gaps, not automatic exclusions or successful checks. The two
previously reviewed nesting-limit rows pass the target validator; retain them only
with pinned, explicit gap resolutions in the new review.

`filter_sft_dataset.py` also accepts `target-filter-review-v1` reviews with policy
`sft-target-compatible-v1`, pinned audit/manifest SHA256 values, approved rows,
schema reviews and retained coverage gaps. Approved reasons must exactly match the
audit proposals (`target_contract_incompatible`, `credential_emitting_target`, or
both). Credential-only reviews still accept only their original policy/reasons.
The filter verifies reviewed row hashes, source-file hashes and provenance, preserves
retained bytes,
order and split membership, and publishes only to a fresh output directory. The new
policy and review identity are recorded in the derived manifest and provenance.

No new filtered dataset or performance improvement is claimed by this implementation.
After review/filtering, validate and report the real capped artifacts. Keep ACT200
quantities, prompt, seed and LoRA settings fixed, and use the same filtered xLAM pool
for an ACT200 control and matched-contrast experiment. Comparing only against the
old unfiltered run would confound data cleanup with the contrast change. Previously
recorded benchmark outcomes must remain unchanged.

## Frozen ACT200 Reference Sampling

The first cleaned-pool run was not a filtering-only ablation. Comparing canonical
content hashes in the raw training caps found only 7 of 600 shared xLAM rows and
301 of 400 shared BAYMAX rows. The combined shuffle changes when pool size changes,
even at the same seed. Keep that benchmark result, but label it a resampled cleaned
pool comparison rather than attributing its changes solely to excluded examples.

Opt-in `mix_sft_datasets.py --reference-mix-dir` anchors both raw training and
validation caps to an existing mix. Matching uses source plus canonical content
hash, not renumbered xLAM indices. Preserve every available reference example in
its original cap position. Replace a missing example only when it is xLAM and its
same-split content hash appears in the filtered artifact's pinned target review.
Missing or changed BAYMAX examples and unexplained missing xLAM examples fail.
Reserve all compatible reference rows before selecting replacement candidates;
never consume a later protected example as filler. Replacements are unique,
same-split xLAM examples ranked deterministically by seed, split and content hash.

The sampler verifies filtered source file hashes and review/provenance identity,
preserves reference prompt/seed and BAYMAX class quotas, and rejects duplicates,
insufficient candidates, repeats, group-preserving mode and existing destinations.
Disabled behavior and historical mix hashes are unchanged. The anchored mix identity
includes the reference manifest, ordered cap content/provenance hashes and exclusion
review/provenance hashes. `sampling-report.json` records preservation counts and
replacement positions/hashes without raw requests or argument values.

The owner-run anchored build passed validation on 2026-10-06. It preserves all 400
training and 90 validation BAYMAX examples, plus 582 and 107 xLAM examples,
respectively. Independently comparing raw caps confirmed exactly 982 unchanged
training rows and 197 unchanged validation rows, with 18 and 3 reviewed replacements
at their recorded positions; replacement content hashes match the sampling report.
The mixed revision is `329fab4d8f1604a4085d3ec50e05881e7c069717bb1428db7e592743bb07f03c`.
The new config `configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200.json`
is pinned to it, with the original ACT200 prompt, seed, model, LoRA and training
settings unchanged. The owner completed the one-epoch training and 150-scenario
evaluation: task success was 0.260, routing macro-F1 0.429 and invalid-output rate
0.080. These are development-benchmark results, not an untouched final test.
Class quantities and raw caps remain unchanged. Replacement lengths can change
tokenization/truncation and usable-row counts; inspect both actual artifacts before
training. Test data remains in its source split and is never used as replacement
training/validation data. No evaluation fixtures are read by the sampler.

Build a new version with the owner-run command below, then validate it. Do not
overwrite either historical mix, config, checkpoint or result artifact. Pin a new
training config to the resulting revision only after the build completes.

```powershell
.venv\Scripts\python scripts\mix_sft_datasets.py `
  --xlam-dir data\sft\v1\xlam-content-groups-target-compatible-v1 `
  --baymax-dir data\sft\v1\baymax-control-policy-v14 `
  --reference-mix-dir data\sft\v1\xlam-baymax-auth-filtered-v14-act200-val200 `
  --output-dir data\sft\v1\xlam-baymax-target-compatible-v14-act200-anchored-val200 `
  --seed 1 --train-cap-row-count 1000 --train-cap-baymax-row-count 400 `
  --train-cap-baymax-clarification-count 100 --train-cap-baymax-refusal-count 100 `
  --validation-cap-row-count 200
```

## Anchored Epoch-Length Test

Owner-approved next experiment on 2026-10-06: compare the anchored one-epoch
reference with a fresh two-epoch run, before changing contrast structure.
Use `configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json`
(hash `ce9efa2db24a065dbdb4eb482c07a29a56bfbcbdce32495418eb54b92d985c40`).
Only `training.num_train_epochs` and the run/output identity change. Dataset
revision, selected rows, prompt, seed, base model, LoRA and other configured
training settings remain identical. Do not rebuild the mix or resume the old
adapter. Start from the same base model; keep the previous checkpoint and results.

Apply the same owner-run overrides: `--local-low-vram --max-train-rows 1000
--max-eval-rows 200 --max-sequence-length 1024`. The one-epoch run retained 974
training and 195 validation examples after assistant-label filtering. Reusing the
same rows/tokenizer/sequence length should preserve these per-pass counts; verify
the actual reports rather than increasing caps to compensate for truncation.

The cosine schedule and warmup fraction remain configured identically, but their
step durations scale with the longer run. This is a training-duration comparison,
not an isolated extra pass at the one-epoch run's learning rates. Evaluate with
the same generation settings and 150 scenarios, under a new result filename.
Compare routing macro-F1, task success, premature actions and invalid outputs;
lower training loss alone is not evidence of improvement. The owner completed
the two-epoch run and evaluation: 44/150 task successes, routing macro-F1 0.514,
2 invalid outputs and 31 premature actions. The corresponding one-epoch values
were 39/150, 0.429, 12 and 36. Both actual training reports retained the same
974 training and 195 validation rows, with no exact duplicates. These single-seed
development results support a longer-run test, not a claim of robust improvement.

On 2026-10-07 the owner chose a fresh four-total-epoch anchored run instead of
testing other presets. Use
`configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs4.json`
(hash `067656877679d9f5bae4414ae5f8f57b5b9ffbc94f95b578ab4a28b9c3fa4b43`),
with the same low-VRAM overrides, 1000/200 raw caps and sequence length 1024.
Only epochs and run/output identity differ from the reference; do not rebuild data,
change prompts or overwrite earlier adapters/results. Start from base Qwen with
a new LoRA initialization, not the saved two-epoch adapter. Loading adapter weights
alone does not restore optimizer, scheduler or random-state checkpoints; a full
resume would need those states and an explicit schedule-extension policy. The
fresh four-epoch cosine/warmup schedule spans the full longer run. The owner has
completed training and both evaluations. On the combined 150 scenarios, task
success fell from 44/150 at two epochs to 37/150 at four, routing macro-F1 from
0.514 to 0.372, and invalid outputs rose from 2 to 13. On the 30-scenario
template-shift suite, task successes fell from 5 to 4 and invalid outputs rose
from 2 to 6. Lower validation loss did not imply better balanced routing.

The two-epoch checkpoint is now the selected development reference, recorded in
[the SHA256 reference manifest](../results/phase2/references/anchored-epochs2-reference.json).
This detects file drift; it does not copy or permission-lock existing artifacts.
The original run's exact base-model revision was not recorded. See the
[saved-output failure audit](../results/phase2/analysis/anchored-epochs2-vs4-failure-audit.md)
for separate decision, contract and wording-score categories across all four
result files. Historical metrics, fixtures, data and adapters remain unchanged.

## Matched Contrasts Against the Two-Epoch Reference

The next owner-approved experiment reuses the existing opt-in matched generator.
New mixer flag `--replace-reference-baymax-train` requires both
`--reference-mix-dir` and `--preserve-contrast-groups`. It replaces only the
reference's BAYMAX training slots, preserving ACT/CLARIFY/REFUSE slot labels and
the 200/100/100 raw quantities. Every xLAM training row stays at its reference
position; the full validation and test files are copied byte-for-byte, including
their original provenance. This is distinct from the older matched mix above,
which reselected xLAM rows and changed held-out data.

Selection uses only the new source's training partition. If any member's complete
input appears in retained validation/test data, the whole contrast group is
excluded even when its target differs. Standalone overlapping inputs are excluded
too. The sampler rejects insufficient unique candidates instead of duplicating or
borrowing held-out examples. The sampling report lists complete selected groups,
excluded groups, class counts, preserved xLAM counts and changed BAYMAX positions.
Full reference split hashes and replacement source hashes enter the new identity;
disabled historical sampler behavior remains unchanged.

Use new source/mix/config/adapter/result paths. Keep the reference prompt, seed,
LoRA configuration, generation settings and two epochs. Do not rebuild the xLAM
pool. The real build and training have not been run by the coding agent. After
the owner builds and validates the new mix, pin a new training config to its actual
manifest revision. Inspect post-tokenization class/token counts before training:
raw quotas and complete raw groups do not guarantee identical token exposure or
that every counterpart survives truncation. Do not adjust caps to hide filtering.

Existing evaluation suites, targets, parser and scorer remain unchanged. The
experiment tests contrast structure, not additional epochs, a new policy prompt,
or a larger boundary class contribution. It does not constitute an untouched-test
generalization claim.

The owner completed the matched source and anchored mix build on 2026-10-07.
Dataset validation passed with mixed revision
`5973425e53c503b52b77b750d5851ef58019d6b33decdaffe4b4e85b610e3118`.
The raw 1000-row cap contains 600 unchanged xLAM rows and 200/100/100 BAYMAX
ACT/CLARIFY/REFUSE rows, with no exact duplicates. Of the 400 BAYMAX positions,
398 changed and 2 retained identical content. The sampling report lists 120
complete contrast groups and no held-out-input group exclusions. Independent
file hashing confirmed identical validation/test bytes and provenance. These
checks do not establish semantic independence or post-tokenization group coverage.

New config:
`configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_matched_anchored_val200_epochs2.json`,
hash `dba7edd43ed83e09e9c1a99c2591ed941c6043a7d2e43bcc7376507da36037c0`.
Training settings and LoRA are identical to the frozen two-epoch config; only
data identity/paths and experiment output identity change. The owner completed
the tokenized-artifact report, training and combined-150 evaluation. Training
retained 974 rows, including all 400 BAYMAX rows; validation retained the same
195 rows as the reference. There were no exact artifact duplicates. Task success
was 39/150 versus 44/150 for the anchored two-epoch reference, routing macro-F1
0.383 versus 0.514, and invalid outputs 8 versus 2.

The [paired refusal/format audit](../results/phase2/analysis/matched-contrasts-epochs2-refusal-format-audit.md)
separates decision regression from output errors. Correct REFUSE routing fell
from 16/29 to 5/29, with no newly correct refusal decisions. The training-only
provenance audit found refusal triplet origins increased from 33/100 to 73/100,
while standalone and capability-boundary-pair origins disappeared from the cap.
These are source-family observations, not a complete semantic coverage audit or
proof of the regression's cause. Keep the original two-epoch reference; no further
training/data/scoring changes were made during this audit.

## 2026-10-09: Separate 2:1 SFT mixture experiment

The owner approved testing approximately 2:1 productive to non-productive rows
before introducing DPO or changing models. Use the original anchored two-epoch
reference, not the regressed matched-contrast checkpoint. The raw 1000-row cap
targets 666 ACT / 164 CLARIFY / 170 REFUSE: 200 original BAYMAX productive rows,
334 unique BAYMAX boundary rows, and 466 retained xLAM rows.

`scripts/prepare_sft_2to1.py` uses the opt-in reference rebalancer. Keep all original
BAYMAX rows, add 64 clarification and 70 refusal training-only examples, and deterministically
replace 134 capped xLAM positions. Displaced xLAM rows move outside the cap; the
source dataset is not edited. Additional inputs must be unique and must not match
any reference training/validation/test input. Validation/test files are copied
byte-for-byte. No group-first sampling, newly authored examples, prompt change,
LoRA change, or evaluation-fixture change is part of this experiment.

The setup script creates a new training config using the built manifest revision
and a recomputed config hash. Existing config/data/model paths cannot be overwritten.
The raw row ratio is not a token ratio: run the artifact report at the same 1024
sequence length, train cap 1000 and validation cap 200 before training. Label
filtering may change the effective row ratio and optimizer step count; disclose
actual counts rather than claiming an exact post-tokenization 2:1 ratio.

The initially proposed 167/167 boundary allocation failed the uniqueness guard.
A bounded training-source audit found 251 clarification rows but only 164 unique
inputs/full rows, and 297 refusal rows but only 174 unique inputs/full rows.
After retaining the original 100 per class, 64 clarification and 74 refusal
inputs remain outside the reference cap; none overlap the source held-out inputs.
The corrected allocation keeps the boundary total at 334 rather than repeating
rows, borrowing held-out data, or silently weakening validation. This count check
does not establish semantic diversity. The real build still checks all retained
reference inputs before admitting an addition.

No real mixture build, tokenization, training, or evaluation was run by the coding
agent for this experiment. Keep this suite's development-benchmark framing.
