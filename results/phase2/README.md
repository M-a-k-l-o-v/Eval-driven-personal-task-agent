# Phase 2 Results

This catalog covers all 59 existing result/analysis artifacts. Evaluations use
Qwen/Qwen2.5-1.5B-Instruct, either the base model or a LoRA adapter, with fake
Calendar, Notion, Gmail, and clipboard tools.

## Reading the Names

- `v1-50`: full 50-scenario v1 evaluation.
- `combined-150`: combined v1/v2 evaluation with 150 scenarios.
- `template-shift-30`: separate 30-scenario suite testing changed wording and structure.
- `subset-7`: seven refusal scenarios, not a full benchmark.
- `train1k` / `train1500`: historical training-cap labels, not evaluation counts.
- `epochs2`: the two-epoch experiment.
- `policy-prompt-v1/v2`: inference-time policy-prompt experiments.
- `training-prompt-a314`: restored training prompt, SHA256
  `a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468`.
- `decision-metrics`: adds ACT/CLARIFY/REFUSE decision accuracy while retaining keyword metrics.
- `strict-parser-rescore`: saved generations reparsed and fake tools replayed;
  no new inference, and latency/cost are inherited from the source run.
- `early-parser-fix`: an earlier parser/adaptation experiment, distinct from strict rescores.

Descriptions are reconstructed from existing filenames, project notes, and
saved metadata. Scenario counts are checked against the JSON files. Older
results generally omit the exact adapter path, training recipe, prompt hash,
and scorer version. Their experiment labels are not a substitute for those
missing records. `historical-prompt` means the exact runtime prompt is not
established by the saved result.

## Comparisons and Provenance

Use the same suite, prompt, parser, and scorer when assessing an experiment.
Scenario expectations and scoring changed during development; a shared
scenario count alone does not establish an identical benchmark version.
Clarification/refusal wording scores are separate from decision correctness.

The format-aligned 50-scenario artifact was our historical reference checkpoint.
The latest experiment is the generator12/content-group-splits 1k run listed
below. The strict-rescore artifacts are the corrected historical parser results
used in our latest comparison. Older adapters predate the fixes to LoRA
initialization seeding and training-data split leakage, so comparisons against
them are historical evidence rather than an isolated test of the training mix.
The repeatedly used suites are development evaluations, not untouched final tests.

Every JSON/text artifact was renamed without changing its bytes. See
[FILE_RENAMES.json](FILE_RENAMES.json) for all original/current repository-relative
paths and SHA256 hashes. Embedded historical paths (including strict-rescore
`source_path`) remain as originally recorded; resolve them through that map.

## Latest Comparison

Run from the repository root:

```powershell
.venv\Scripts\python scripts\compare_eval_results.py `
  acr-1k=results\phase2\parser-integrity-v1\combined-150\act-clarify-refuse-train1k-combined-150-strict-parser-rescore.json `
  schema-diverse=results\phase2\parser-integrity-v1\combined-150\schema-diverse-train1k-combined-150-strict-parser-rescore.json `
  act-balanced=results\phase2\parser-integrity-v1\combined-150\act-balanced-train1k-combined-150-strict-parser-rescore.json `
  act-midpoint=results\phase2\parser-integrity-v1\combined-150\act-midpoint150-train1k-combined-150-strict-parser-rescore.json `
  new-70-30=results\phase2\qwen-lora\baymax-mix\content-group-splits-generator12-train1k-70act-30boundary-combined-150.json
```

## File Catalog

Paths below are relative to this directory.

### Base Model

| File | Scenarios | What Was Tested |
| --- | ---: | --- |
| [qwen-base-combined-150-historical.json](qwen-base/qwen-base-combined-150-historical.json) | 150 | Unfine-tuned Qwen on the combined 150-scenario suite. Historical runtime prompt is not recorded in the result. |
| [qwen-base-combined-150-policy-prompt-v1.json](qwen-base/qwen-base-combined-150-policy-prompt-v1.json) | 150 | Base-model evaluation with the expanded decision/confirmation policy prompt v1. |
| [qwen-base-combined-150-policy-prompt-v2.json](qwen-base/qwen-base-combined-150-policy-prompt-v2.json) | 150 | Base-model evaluation with the longer ACT/CLARIFY/REFUSE policy prompt v2. |
| [qwen-base-combined-150-training-prompt-a314.json](qwen-base/qwen-base-combined-150-training-prompt-a314.json) | 150 | Base-model comparison using the restored a314 training system prompt. |

### LoRA: Synthetic BAYMAX Control

| File | Scenarios | What Was Tested |
| --- | ---: | --- |
| [synthetic-control-original-v1-50.json](qwen-lora/baymax-control/synthetic-control-original-v1-50.json) | 50 | Early LoRA trained on BAYMAX synthetic control data, evaluated on all 50 v1 scenarios. |
| [synthetic-control-original-v1-50-early-parser-fix.json](qwen-lora/baymax-control/synthetic-control-original-v1-50-early-parser-fix.json) | 50 | Original control checkpoint evaluated with the earlier output-adaptation/parser fixes; distinct from the later strict parser rescores. |
| [synthetic-control-contrast-pairs-v1-50.json](qwen-lora/baymax-control/synthetic-control-contrast-pairs-v1-50.json) | 50 | Control-data checkpoint with contrast-pair examples, evaluated on the 50-scenario suite. |
| [synthetic-control-policy-prompt-refusal-subset-7.json](qwen-lora/baymax-control/synthetic-control-policy-prompt-refusal-subset-7.json) | 7 | Policy-prompt experiment on the seven refusal scenarios, using the early control checkpoint. |
| [synthetic-control-policy-prompt-refusal-subset-7-early-parser-fix.json](qwen-lora/baymax-control/synthetic-control-policy-prompt-refusal-subset-7-early-parser-fix.json) | 7 | Seven-scenario refusal subset with the policy prompt and earlier parser/adaptation changes. |
| [synthetic-control-targeted-refusals-v1-50.json](qwen-lora/baymax-control/synthetic-control-targeted-refusals-v1-50.json) | 50 | Control-data checkpoint incorporating targeted refusal examples, evaluated on all 50 v1 scenarios. |
| [synthetic-control-targeted-refusals-prompt-only-subset-7.json](qwen-lora/baymax-control/synthetic-control-targeted-refusals-prompt-only-subset-7.json) | 7 | Seven-scenario refusal-focused prompt experiment. The legacy label does not establish whether a separate checkpoint was trained. |

### LoRA: xLAM + BAYMAX Mix

| File | Scenarios | What Was Tested |
| --- | ---: | --- |
| [original-xlam-control-mix-v1-50.json](qwen-lora/baymax-mix/original-xlam-control-mix-v1-50.json) | 50 | Original mixture of external xLAM and synthetic BAYMAX data; 50-scenario evaluation. |
| [original-xlam-control-mix-v1-50-early-parser-fix.json](qwen-lora/baymax-mix/original-xlam-control-mix-v1-50-early-parser-fix.json) | 50 | Original mixed-data checkpoint with earlier parser/adaptation fixes; not a strict-complete-response rescore. |
| [synthetic-v7-smoke-v1-50.json](qwen-lora/baymax-mix/synthetic-v7-smoke-v1-50.json) | 50 | Legacy v7 synthetic-data smoke experiment evaluated on all 50 scenarios. Exact training exposure is not recorded here. |
| [synthetic-v7-weighted-v1-50.json](qwen-lora/baymax-mix/synthetic-v7-weighted-v1-50.json) | 50 | Legacy v7 experiment with additional boundary-example weighting; exact repeat counts are not stored in the result. |
| [synthetic-v8-meta-tools-weighted-v1-50.json](qwen-lora/baymax-mix/synthetic-v8-meta-tools-weighted-v1-50.json) | 50 | Weighted v8 experiment representing clarification/refusal through internal meta-tool calls. |
| [format-aligned-reference-v1-50.json](qwen-lora/baymax-mix/format-aligned-reference-v1-50.json) | 50 | Historical aligned reference checkpoint after training/evaluation input format alignment; evaluated on 50 scenarios. |
| [clarification-extra6-refusal-extra12-v1-50.json](qwen-lora/baymax-mix/clarification-extra6-refusal-extra12-v1-50.json) | 50 | Mixed-data weighting experiment labelled six extra clarification repeats and twelve extra refusal repeats. |
| [boundary-weighting-legacy13-v1-50.json](qwen-lora/baymax-mix/boundary-weighting-legacy13-v1-50.json) | 50 | Legacy boundary-weighting experiment labelled boundary13. The exact weighting recipe is not established by its saved metadata. |
| [unique-boundaries-train1k-v1-50.json](qwen-lora/baymax-mix/unique-boundaries-train1k-v1-50.json) | 50 | Unique-first BAYMAX boundary sampling, training cap labelled 1000 rows; 50-scenario evaluation. |
| [unique-boundaries-train1500-v1-50.json](qwen-lora/baymax-mix/unique-boundaries-train1500-v1-50.json) | 50 | Larger training cap labelled 1500 rows with the fixed unique boundary pool; 50-scenario evaluation. |
| [unique-boundaries-train1k-combined-150-historical-prompt.json](qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-historical-prompt.json) | 150 | The 1k boundary checkpoint evaluated on the combined 150-scenario suite before the explicitly restored-prompt run. |
| [unique-boundaries-train1500-combined-150-historical-prompt.json](qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-historical-prompt.json) | 150 | The 1500-row boundary checkpoint on the combined suite before the explicitly restored-prompt run. |
| [unique-boundaries-train1k-combined-150-training-prompt-a314.json](qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-training-prompt-a314.json) | 150 | The 1k boundary checkpoint reevaluated with the restored a314 training prompt. |
| [unique-boundaries-train1500-combined-150-training-prompt-a314.json](qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-training-prompt-a314.json) | 150 | The 1500-row boundary checkpoint reevaluated with the restored a314 training prompt. |
| [format-aligned-combined-150-historical-prompt.json](qwen-lora/baymax-mix/format-aligned-combined-150-historical-prompt.json) | 150 | Aligned checkpoint on the original combined suite; historical runtime prompt is not recorded in the artifact. |
| [format-aligned-combined-150-policy-prompt-v1.json](qwen-lora/baymax-mix/format-aligned-combined-150-policy-prompt-v1.json) | 150 | Existing aligned adapter evaluated using policy prompt v1, which differed from its training prompt; no retraining for this prompt ablation. |
| [format-aligned-combined-150-policy-prompt-v2.json](qwen-lora/baymax-mix/format-aligned-combined-150-policy-prompt-v2.json) | 150 | Existing aligned adapter with longer policy prompt v2; an inference-prompt ablation, not a newly trained adapter. |
| [format-aligned-combined-150-training-prompt-a314.json](qwen-lora/baymax-mix/format-aligned-combined-150-training-prompt-a314.json) | 150 | Aligned checkpoint reevaluated with its restored a314 training prompt. |
| [act-clarify-refuse-train1k-combined-150.json](qwen-lora/baymax-mix/act-clarify-refuse-train1k-combined-150.json) | 150 | 1000-row mixed-data training experiment incorporating 100 authored ACT/CLARIFY/REFUSE triplets, followed by 150-scenario evaluation. |
| [act-clarify-refuse-train1k-epochs2-combined-150.json](qwen-lora/baymax-mix/act-clarify-refuse-train1k-epochs2-combined-150.json) | 150 | Two-epoch counterpart to the 1k ACT/CLARIFY/REFUSE experiment. |
| [schema-diverse-train1k-combined-150.json](qwen-lora/baymax-mix/schema-diverse-train1k-combined-150.json) | 150 | 1k training experiment varying tool argument schemas and partial-update calls; combined-suite evaluation. |
| [act-balanced-train1k-combined-150-legacy-metrics.json](qwen-lora/baymax-mix/act-balanced-train1k-combined-150-legacy-metrics.json) | 150 | ACT-balanced mixed-data experiment before explicit decision metrics were added. |
| [act-balanced-train1k-combined-150-decision-metrics.json](qwen-lora/baymax-mix/act-balanced-train1k-combined-150-decision-metrics.json) | 150 | ACT-balanced checkpoint reevaluated with explicit ACT/CLARIFY/REFUSE decision metrics; keyword metrics retained. |
| [act-midpoint150-train1k-combined-150-decision-metrics.json](qwen-lora/baymax-mix/act-midpoint150-train1k-combined-150-decision-metrics.json) | 150 | Legacy midpoint150 ACT-mix experiment with a 1k training cap and decision metrics. The label alone does not describe the total ACT exposure. |
| [content-group-splits-generator12-train1k-70act-30boundary-combined-150.json](qwen-lora/baymax-mix/content-group-splits-generator12-train1k-70act-30boundary-combined-150.json) | 150 | Latest run: generator v12, corrected xLAM/BAYMAX exact-content splitting, 1k cap with 150 unique clarification + 150 unique refusal + 700 ACT rows before filtering; strict parser and decision metrics. |

### Template-Shift Evaluations

| File | Scenarios | What Was Tested |
| --- | ---: | --- |
| [qwen-base-template-shift-30.json](template-shift/qwen-base-template-shift-30.json) | 30 | Base Qwen on the independently authored 30-scenario wording/template-shift suite. |
| [format-aligned-template-shift-30.json](template-shift/format-aligned-template-shift-30.json) | 30 | Aligned reference adapter evaluated on the 30-scenario template-shift suite. |
| [act-clarify-refuse-train1k-template-shift-30.json](template-shift/act-clarify-refuse-train1k-template-shift-30.json) | 30 | 1k ACT/CLARIFY/REFUSE checkpoint tested on different wording and request structure. |
| [act-clarify-refuse-train1k-epochs2-template-shift-30.json](template-shift/act-clarify-refuse-train1k-epochs2-template-shift-30.json) | 30 | Two-epoch ACT/CLARIFY/REFUSE checkpoint on the template-shift suite. |
| [schema-diverse-train1k-template-shift-30.json](template-shift/schema-diverse-train1k-template-shift-30.json) | 30 | Schema-diverse 1k adapter on the 30-scenario transfer suite. |
| [act-balanced-train1k-template-shift-30-legacy-metrics.json](template-shift/act-balanced-train1k-template-shift-30-legacy-metrics.json) | 30 | ACT-balanced adapter on template-shift scenarios before explicit decision metrics. |
| [act-balanced-train1k-template-shift-30-decision-metrics.json](template-shift/act-balanced-train1k-template-shift-30-decision-metrics.json) | 30 | ACT-balanced adapter on template-shift scenarios with explicit decision metrics. |
| [act-midpoint150-train1k-template-shift-30-decision-metrics.json](template-shift/act-midpoint150-train1k-template-shift-30-decision-metrics.json) | 30 | Midpoint150 ACT-mix adapter on the template-shift suite with decision metrics. |

### Strict Parser Rescores: Combined Suite

| File | Scenarios | What Was Tested |
| --- | ---: | --- |
| [act-clarify-refuse-train1k-combined-150-strict-parser-rescore.json](parser-integrity-v1/combined-150/act-clarify-refuse-train1k-combined-150-strict-parser-rescore.json) | 150 | Saved 150-scenario ACT/CLARIFY/REFUSE generations reparsed with strict-complete-response-v1; fake tools replayed, no fresh inference. |
| [schema-diverse-train1k-combined-150-strict-parser-rescore.json](parser-integrity-v1/combined-150/schema-diverse-train1k-combined-150-strict-parser-rescore.json) | 150 | Saved 150-scenario schema-diverse generations reparsed with the strict parser; no retraining or fresh inference. |
| [act-balanced-train1k-combined-150-strict-parser-rescore.json](parser-integrity-v1/combined-150/act-balanced-train1k-combined-150-strict-parser-rescore.json) | 150 | Saved 150-scenario ACT-balanced decision-metrics run reparsed with the strict parser. |
| [act-midpoint150-train1k-combined-150-strict-parser-rescore.json](parser-integrity-v1/combined-150/act-midpoint150-train1k-combined-150-strict-parser-rescore.json) | 150 | Saved 150-scenario midpoint150 decision-metrics run reparsed with the strict parser. |

### Strict Parser Rescores: Template Shift

| File | Scenarios | What Was Tested |
| --- | ---: | --- |
| [act-clarify-refuse-train1k-template-shift-30-strict-parser-rescore.json](parser-integrity-v1/template-shift/act-clarify-refuse-train1k-template-shift-30-strict-parser-rescore.json) | 30 | Saved 30-scenario ACT/CLARIFY/REFUSE generations reparsed with strict-complete-response-v1; fake tools replayed, no fresh inference. |
| [schema-diverse-train1k-template-shift-30-strict-parser-rescore.json](parser-integrity-v1/template-shift/schema-diverse-train1k-template-shift-30-strict-parser-rescore.json) | 30 | Saved 30-scenario schema-diverse generations reparsed with the strict parser; no retraining or fresh inference. |
| [act-balanced-train1k-template-shift-30-strict-parser-rescore.json](parser-integrity-v1/template-shift/act-balanced-train1k-template-shift-30-strict-parser-rescore.json) | 30 | Saved 30-scenario ACT-balanced decision-metrics run reparsed with the strict parser. |
| [act-midpoint150-train1k-template-shift-30-strict-parser-rescore.json](parser-integrity-v1/template-shift/act-midpoint150-train1k-template-shift-30-strict-parser-rescore.json) | 30 | Saved 30-scenario midpoint150 decision-metrics run reparsed with the strict parser. |

### Data Audits and Failure Reports

| File | Scenarios | What Was Tested |
| --- | ---: | --- |
| [original-mix-token-audit-5000-and-boundary-failures.json](analysis/original-mix-token-audit-5000-and-boundary-failures.json) | Analysis | 5000-row token-distribution audit of the early mixture, plus grouped evaluation failures and raw boundary failures. |
| [clarification-extra6-refusal-extra12-token-audit-5000-snapshot1.json](analysis/clarification-extra6-refusal-extra12-token-audit-5000-snapshot1.json) | Analysis | 5000-row token audit and failure analysis for the 6/12 weighting experiment; first stored snapshot. |
| [clarification-extra6-refusal-extra12-token-audit-5000-snapshot2.json](analysis/clarification-extra6-refusal-extra12-token-audit-5000-snapshot2.json) | Analysis | Same token counts as snapshot1, but expanded failure comparisons (16 behavior/run entries and 99 boundary failures). |
| [boundary-weighting-legacy13-token-audit-5000-snapshot1.json](analysis/boundary-weighting-legacy13-token-audit-5000-snapshot1.json) | Analysis | 5000-row token audit with 75 recorded boundary failure generations. |
| [boundary-weighting-legacy13-token-audit-5000-snapshot2.json](analysis/boundary-weighting-legacy13-token-audit-5000-snapshot2.json) | Analysis | Same token counts as snapshot1, with a different failure snapshot containing 74 boundary failure generations. |
| [qwen-base-policy-prompt-v1-failure-summary.txt](analysis/qwen-base-policy-prompt-v1-failure-summary.txt) | Analysis | Text summary of base-model failures from the decision-policy prompt v1 evaluation. |
| [format-aligned-policy-prompt-v1-failure-summary.txt](analysis/format-aligned-policy-prompt-v1-failure-summary.txt) | Analysis | Text summary of aligned-adapter failures from the decision-policy prompt v1 evaluation. |
