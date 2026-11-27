# A Larger Unique Training Pool, Still Two Epochs

**Status:** Owner built, validated, trained and evaluated; four-epoch follow-up completed.
Date: 2026-10-16.

## Question

Does a larger pool with additional unique BAYMAX coverage improve on the corrected
1k model without increasing epochs or changing loss weighting?

The owner approved 2,000 raw training rows at two epochs, not 2,000 rows at four
epochs. This is a size-and-coverage experiment, not an isolated test of wording
diversity. At two epochs, twice as many usable rows means roughly twice the sample
exposure and optimizer steps. The cosine schedule also spans the longer run.

## Parent and Reference

The immediate parent is [grounding v1](18-controlled-grounding-data-revision.md):

- Config: `configs/sft/qwen2_5_1_5b_lora_grounding_v1_epochs2.json`.
- Dataset: `data/sft/v1/name-first-grounding-v1`.
- Dataset revision: `91ce08228633738365461a556668a6883b5faa806b84d969dddd8164114603a1`.
- Training config hash: `f060ddbc1df6182840086ffc4b60f27785390ee2ee41938e2578bea3f3663986`.

This retains the independently justified corrections, rather than restoring
incorrect examples to recover a score. Grounding v1 scored 53/150 successes and
0.591 macro-F1. The unweighted reference remains the selected model at 57/150 and
0.612; do not silently replace it with this parent or the new experiment.

## Exact Raw Quantities

| Contribution | Parent 1k | Added | Expanded 2k |
| --- | ---: | ---: | ---: |
| BAYMAX single productive calls | 175 | 175 | 350 |
| BAYMAX multi-tool sequences | 25 | 25 | 50 |
| BAYMAX clarification | 100 | 100 | 200 |
| BAYMAX refusal | 100 | 100 | 200 |
| xLAM productive examples | 600 | 600 | 1,200 |
| Total | 1,000 | 1,000 | 2,000 |

The 2k raw mixture is 800 BAYMAX and 1,200 xLAM, with broad routing quantities
1,600 ACT / 200 CLARIFY / 200 REFUSE. These are not post-tokenization counts.
The raw 1k parent was checked model-free: 1,000 unique inputs, zero exact duplicates,
and the BAYMAX quantities above. The larger artifact still needs owner inspection.

## Where the New Examples Come From

`sft_expansion_examples.py` independently authors 100 training-only groups. Each
has two ACT examples, one CLARIFY and one REFUSE. Eight families cover:

- Clipboard-to-task content binding, including an empty-clipboard contrast.
- Calendar creation with explicit timing versus missing time and/or duration.
- Calendar updates using a unique state entity versus multiple matching entities.
- Task deadlines: known dates, explicitly undated urgency and unresolved deadlines.
- Task completion using state IDs versus ambiguous same-title tasks.
- Routine email with known/contact-resolved recipients versus missing addresses;
  indiscriminate bulk outreach is refused.
- Clipboard-to-draft body binding versus missing contents or unavailable tools.
- Draft recipients resolved from state; a send request is refused when only drafts
  are supported.

Instruction banks contain three to six phrasings per relevant family. Contents,
dates, start times, durations, contact state, entity IDs and distractors vary.
The pool is still templated: 400 unique inputs are not 400 independent reasoning
patterns, and no claim of unrestricted linguistic generalization is made.

All additions are explicitly training-only. No evaluation fixture or held-out
target supplies their content. Existing source validation executes only fake
in-memory adapters, and strict schema checks validate all generated targets.

## Sampling and Separation

`scripts/prepare_sft_expansion.py` derives a new dataset and a freshly hashed config.
It does not change the default mixer, its fixed BAYMAX quotas, or any old artifact.

The original first 1,000 training lines are retained byte-for-byte. A new 1,000-row
block is shuffled with the existing seed and placed after them. The 600 external
additions are the next eligible unique xLAM rows from the parent's training tail;
they move into the cap and are removed from that tail, not copied twice.

Retained parent rows keep their source identity and split. Full validation/test
files remain byte-identical. New inputs must be distinct from every parent input
and from each other. A frozen-prefix input overlapping any held-out input blocks
publication; conflicting unused training-tail inputs are excluded and recorded.
They cannot be selected as xLAM additions. Exact input
checks are not proof of semantic/template independence. No new held-out set is
created; inherited split-ratio metadata describes the parent split policy, not a
new 70/20/10 split of the expanded data.

Before any tail-overlap exclusions, the full dataset grows by 400 BAYMAX rows,
not 1,000, because external rows are relocated. Review `expansion-sources.jsonl` for all authored examples and
`expansion-report.json` for source hashes, moved external identities, before/after
cap counts, unique inputs and duplicate counts. Populated/overlapping destinations,
quota mismatches, insufficient unique data and source drift fail explicitly.

### Build Guard Correction

The first owner build stopped with an input overlap in the parent's training
tail. The historical splitter/validator compares complete examples, including
answers; identical inputs with different answers can therefore remain in different
splits. This is not evidence that the separate 150 benchmark scenarios were used
for training. A bounded read-only check found zero held-out input matches in the
parent's first 2,000 training rows; the rest was not audited by the coding agent.

The corrected preparation preserves the frozen 1k prefix and excludes conflicts
only outside it, with source references and held-out locations in
`expansion-report.json:excluded_parent_tail` and manifest skipped rows. Source
files and historical results are untouched. Full builds still check all inputs.
Regression tests reproduce a same-input/different-answer overlap that passes the
old validator and prove both safe tail exclusion and strict frozen-prefix failure.
The new guard policy is included in the derivation hash.

Use new `coverage-2k-v2` output/config identities; the failed v1 folder contains
only an invalid validation report and is not reused. Do not validate a failed build.

## Fixed Training Settings

Qwen2.5-1.5B-Instruct; fresh LoRA r=8, alpha=16, Q/K/V/O; dropout 0.05;
learning rate 2e-4; two epochs; seed 1; unweighted first-tool loss; same prompt,
compact input serialization, name-first targets and generation/scoring settings.
Laptop overrides remain context 1,024, batch one and accumulation 16.
No adapter is resumed. Validation cap remains the same 200 rows.

## Owner-Run Steps

```powershell
.venv\Scripts\python scripts\prepare_sft_expansion.py
if ($LASTEXITCODE -ne 0) { throw "Preparation failed; do not validate or train." }

.venv\Scripts\python scripts\validate_sft_dataset.py `
  --dataset-dir data\sft\v1\name-first-coverage-2k-v2

$config = "configs\sft\qwen2_5_1_5b_lora_coverage_2k_v2_epochs2.json"

.venv\Scripts\python scripts\report_sft_artifact.py `
  --config $config --split train `
  --max-rows 2000 --max-sequence-length 1024

.venv\Scripts\python scripts\report_sft_artifact.py `
  --config $config --split validation `
  --max-rows 200 --max-sequence-length 1024
```

Review those counts and the source examples before training. In particular, both
200-row boundary classes should retain labels, and no duplicates should appear.
After review, the intended fresh run is:

```powershell
.venv\Scripts\python scripts\train_sft_lora.py `
  --config configs\sft\qwen2_5_1_5b_lora_coverage_2k_v2_epochs2.json `
  --local-low-vram `
  --max-train-rows 2000 --max-eval-rows 200 `
  --max-sequence-length 1024
```

The owner reviewed the tokenized artifact before training. The coding agent did
not run the full build, tokenizer, training or evaluation. The repeatedly inspected
150-case suite is development evidence, not an untouched final test.

## Completed Artifact and Results

Dataset revision: `43892841fad91d9c328f3577e831c5d192242f84ceabe82d18e7deb30d2c6ab4`.
Config hash: `e48c75d01c9c14eb6ef1c24d4e8d1ae516fc3bf164a288918517c6f1ce2db524`.
The owner build excluded four unused-tail overlaps. The final full dataset has
58,419 converted rows; train 40,821, validation 11,719 and test 5,879.

| Tokenized training class | Rows | Trainable tokens |
| --- | ---: | ---: |
| Direct tool call | 885 | 27,313 |
| Multi-tool sequence | 667 | 34,244 |
| Clarification | 200 | 5,789 |
| Refusal | 200 | 5,515 |

The usable 1,952-row cap contains all 800 BAYMAX rows and 1,152 xLAM rows,
with zero exact duplicates. The unchanged validation artifact retains 195 rows,
including 30 clarification and 30 refusal examples.

| Run | Successes / 150 | Macro-F1 | Invalid / 150 | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Grounding 1k | 53 | 0.591 | 0 | 0.826 | 0.163 | 0.379 |
| Coverage 2k / 2 epochs | 64 | 0.615 | 2 | 0.806 | 0.163 | 0.603 |

The [saved two-epoch run](../../../results/phase2/json-spacing/name-first-coverage-2k-v2-compact-combined-150.json)
is timestamped 2026-10-16T19:43:41Z. Its matrix has expected rows and predicted
columns:

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 59 | 15 | 6 | 1 |
| CLARIFY | 24 | 12 | 4 | 0 |
| REFUSE | 3 | 0 | 25 | 1 |

The observed size/coverage change improves task completion and refusal compared
with the corrected 1k parent, but still acts prematurely in 24/40 clarification
cases. It does not isolate data diversity from additional training exposure.
The [four-epoch follow-up](20-unique-2k-four-epochs.md) changes duration separately.
