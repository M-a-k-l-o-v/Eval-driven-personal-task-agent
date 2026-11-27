# Training Behavior Gaps

This document tracks boundary examples BAYMAX should learn during Phase 2 SFT.
The v1 eval scenarios remain held out and must not be copied into training data.

## Clarification Examples

Use these when the request is supported, but the model needs more information
before it can safely call a tool.

- Missing recipient: "Email them the notes."
- Unclear pronoun: "Send him the update."
- Missing email body: "Draft an email to maya@example.com."
- Missing calendar time: "Schedule biology revision tomorrow."
- Vague time: "Schedule this tonight."
- Multiple matching calendar events: "Move my meeting to 3pm."
- Multiple matching Notion tasks: "Mark the review task done."
- No matching entity: "Move my physics meeting" when no such meeting exists.
- Invalid or impossible date/time: "Schedule this on 2026-02-30."
- Invalid email address: "Email Maya at maya-at-example."
- Empty clipboard write: "Copy this" with no text to copy.
- Multi-step missing dependency: "Create the task and email Marv about it."

## Refusal Examples

Use these when the request should not be done, even with more information.

- Unavailable service: taxi, food delivery, hotel booking, ticket purchase.
- Unsupported read/list/search operation when only create/update tools exist.
- Secret or password access.
- Private file exfiltration.
- Impersonation.
- Bulk unsolicited email.
- External payment or purchase.
- Unsupported deletion when no delete tool exists.
- Real-world action outside available tools.
- Account or data access not present in `initial_state`.

## Confirmation-Before-Action Examples

Use these only when the action is supported by the available tool schema but
should pause for user confirmation. In the current SFT output shape, these are
`request_clarification` examples.

- A supported destructive or irreversible operation.
- A supported destructive bulk update.
- A consequential external communication that states a major commitment or
  status change.

A routine explicit send, create, update, clipboard read, or clipboard write does
not require confirmation. Cancellation or deletion must be refused when no
available tool schema supports that operation; confirmation cannot make an
unavailable capability valid.

## Template Rule

Each boundary template should ideally have a contrast example:

```text
unsupported, unavailable, or unsafe -> refuse_request
supported but under-specified        -> request_clarification
supported, safe, and complete        -> direct tool call
```

This helps the model learn the boundary instead of memorizing one keyword.

## Generator v12: Capability Boundary Expansion

`src/baymax/models/boundary_cases.py` contains 64 authored requests, 16 per
domain. Each produces a CLARIFY/REFUSE pair: the request and initial state stay
the same, but the required productive tool is available only in the CLARIFY
member. Both members stay in the same partition. These 128 additional SFT rows
were authored from tool contracts and the decision policy, without reading
evaluation fixtures or using them as training inputs.

Cases cover missing fields, ambiguous references, invalid values, conflicting
instructions, and consequential-send confirmation. New refusals specifically
teach unavailable-capability precedence over missing information. This is not
128 independent task families, nor comprehensive new unsafe-request coverage;
there are 64 paired requests, and refusal wording remains templated.

The old 1000 source rows remain; generator v12 contains 1128 source rows.
With seed 1, `data/sft/v1/baymax-control-boundary-v12` contains 790 train,
225 validation, and 113 test rows. Full-content cross-partition validation
passes. Existing contrast groups and duplicate-content groups remain indivisible.

The mixer now rejects unmet unique-row quotas rather than silently filling
missing BAYMAX boundary slots with productive xLAM examples. Regression tests
cover 1000/1500-row caps and insufficient clarification, refusal, productive,
and xLAM pools.

### Diagnostic Mix: Not Ready for Training

`data/sft/v1/xlam-baymax-disjoint-70-30-v12` was built with seed 1, no repeats,
and a 1000-row prefix containing 450 BAYMAX rows (150 per decision) plus 550
xLAM rows. After the existing 1024-token trainable-label filter:

| Behavior | Rows / Unique Rows | Trainable Tokens |
| --- | ---: | ---: |
| Direct tool call | 356 | 10785 |
| Multi-tool sequence | 327 | 17082 |
| Clarification | 150 | 4033 |
| Refusal | 150 | 4331 |

There are 983 usable rows: 450 BAYMAX and 533 xLAM, with zero exact duplicates.
Seventeen xLAM rows have no trainable labels after truncation. Boundary examples
are 30.5% of usable rows but only 23.1% of trainable tokens.

**Do not train this mix yet.** Full validation found 111 validation and 57 test
overlap reports, all on xLAM rows. Three reports reference the first 1000 train
rows. No overlap was found in the separately validated BAYMAX dataset. The next
step is to correct external-data splitting and rebuild/revalidate the mix,
not to weaken the overlap check or modify benchmark expectations. No training
config was switched to this diagnostic mix, and historical datasets were kept.

### External Split Fix

The xLAM builder now groups identical converted SFT content before assigning
train/validation/test partitions (`content-groups-v1`). Different source IDs or
equivalent input JSON formatting cannot separate these copies. All source rows
are retained; split ratios are approximate when a group crosses a boundary.
This prevents exact-content overlap, not semantic overlap between paraphrases.
The strategy is included in the build config hash; the upstream dataset revision
and prompt are unchanged. Existing generated files are not repaired in place.

Rebuild into a new directory from the same pinned source, then validate:

```powershell
.venv\Scripts\python scripts\build_sft_dataset.py `
  --dataset-id Salesforce/xLAM-function-calling-60k `
  --revision 26d14ebfe18b1f7b524bd39b404b50af5dc97866 `
  --seed 1 `
  --output-dir data\sft\v1\xlam-content-groups-v1

.venv\Scripts\python scripts\validate_sft_dataset.py `
  --dataset-dir data\sft\v1\xlam-content-groups-v1
```

Only after that reports `Valid: True`, rebuild the mix into another new directory:

```powershell
.venv\Scripts\python scripts\mix_sft_datasets.py `
  --xlam-dir data\sft\v1\xlam-content-groups-v1 `
  --baymax-dir data\sft\v1\baymax-control-boundary-v12 `
  --output-dir data\sft\v1\xlam-baymax-content-groups-70-30-v12 `
  --seed 1 `
  --train-cap-row-count 1000 `
  --train-cap-baymax-row-count 450 `
  --train-cap-baymax-clarification-count 150 `
  --train-cap-baymax-refusal-count 150

.venv\Scripts\python scripts\validate_sft_dataset.py `
  --dataset-dir data\sft\v1\xlam-baymax-content-groups-70-30-v12
```

The user rebuilt and validated both datasets. The combined dataset reports
`Valid: True`, with 60445 converted rows: 42107 train, 12225 validation, and
6113 test. Its revision is
`487395b4922e78a86c1ac9bb36f3bf71166776dcd5bea43ffa27f750417dec53`.

The separate config `configs/sft/qwen2_5_1_5b_lora_content_groups_70_30_v12.json`
pins that revision, the manifest, and the passing validation report. Config/hash
and path validation pass. Prompt, LoRA, training settings, and seed match the
previous mix config. The next step is the exact 1000-row artifact report at
1024 tokens. Earlier diagnostic counts above do not describe this rebuilt mix.
No training or benchmark runs accompany this fix.
