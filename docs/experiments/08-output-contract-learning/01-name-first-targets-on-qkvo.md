# Name-First Tool Calls: Serialization-Only SFT on Experiment E

**Status:** Preparation implemented; full dataset conversion, token reports,
training and evaluation are pending owner execution. No new benchmark result.

[Experiment index](../README.md)

## Question

Does predicting the tool name before its arguments help the model express its
decision using the correct output contract? The name identifies whether the
arguments need `question`, `reason`, or a productive tool's own schema.

This is a hypothesis about generation order, not proof that arguments-first
JSON caused our failures. A valid output can still contain a wrong decision.

## Exactly One Change

Starting from E's same frozen source dataset, reorder the two outer keys of each
assistant tool call:

```json
{"arguments":{"question":"Which item do you mean?"},"name":"request_clarification"}
```

becomes:

```json
{"name":"request_clarification","arguments":{"question":"Which item do you mean?"}}
```

The question above illustrates the interface; no new example is authored for
this experiment. The converter uses existing training data, not eval fixtures.

| Item | Held Fixed |
| --- | --- |
| Inputs | Exact system/user message strings, tools, request wrapper and world state |
| Answers | Same names, argument fields, values, nested key order and call sequence |
| Selection | All source rows in their existing order, with unchanged split membership |
| Raw cap | 1,000 training rows and 200 validation rows |
| BAYMAX contribution | Same 200 ACT / 100 CLARIFY / 100 REFUSE rows |
| External contribution | Same 600 xLAM rows inside the raw training cap |
| Model initialization | Fresh pretrained Qwen/Qwen2.5-1.5B-Instruct plus fresh LoRA |
| LoRA | E: rank 8, alpha 16, q/k/v/o projections, dropout 0.05 |
| Training | Two epochs, learning rate 2e-4, cosine with 3% warmup, seed 1 |
| Actual overrides | Context 1,024, batch one, gradient accumulation 16 |
| Evaluation | Same prompt, generation settings, strict parser, scorer and 150-case suite |

There is no new prompt, weighting, resampling, split, repair, constrained decoding,
or continuation of E's saved adapter. Shared JSON serialization helpers remain
unchanged; their sorting also affects input construction and sampling hashes.

## Artifact Safety and Provenance

The preparation command derives a **new** dataset and E-based training config.
It refuses existing dataset/config/model output paths. All three splits are
converted consistently, without moving examples between them.

Source targets must already pass the existing output-contract validator and use
canonical compact, arguments-first JSON. Invalid or differently serialized source
targets fail explicitly instead of being repaired or silently normalized.

`serialization-provenance.json` records the transformation version, parent
manifest hash/revision, per-split input/output file hashes, row counts and call
counts. A new dataset revision and training-config hash identify the candidate.
Source-row references retain their original identities; their `row_hash` is not
redefined as a hash of the transformed example. Transformed content is covered
by the new file hashes. No source file or historical result is overwritten.

The generated config appears only after conversion succeeds. It retains E's
declared training settings; the same low-VRAM command overrides still apply.
Its dataset validation report is generated separately, not copied from E.

## First Step: Prepare and Validate

Run from the repository root in PowerShell. These commands do not train a model:

```powershell
.venv\Scripts\python scripts\prepare_sft_name_first.py

$dataset = "data\sft\v1\xlam-baymax-target-compatible-v14-act200-anchored-val200-name-first"
$config = "configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json"

.venv\Scripts\python scripts\validate_sft_dataset.py --dataset-dir $dataset
.venv\Scripts\python scripts\validate_training_config.py --config $config --check-paths
```

Stop if either validator fails. The preparation scans the full source dataset,
not just the capped rows, and may take time. It does not load a model/tokenizer,
fetch data, or access evaluation scenarios.

## Check Actual Trainer Exposure Before Training

Changing serialization can change tokens or truncation boundaries. Do not assume
the same usable rows or assistant-token counts merely because raw caps match.
Run the existing artifact report on **both** configs at the same caps:

```powershell
$reference = "configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo.json"
.venv\Scripts\python scripts\report_sft_artifact.py --config $reference --split train --max-rows 1000 --max-sequence-length 1024
.venv\Scripts\python scripts\report_sft_artifact.py --config $config --split train --max-rows 1000 --max-sequence-length 1024
.venv\Scripts\python scripts\report_sft_artifact.py --config $reference --split validation --max-rows 200 --max-sequence-length 1024
.venv\Scripts\python scripts\report_sft_artifact.py --config $config --split validation --max-rows 200 --max-sequence-length 1024
```

Record class/source counts, unique rows, duplicates and trainable tokens. If usable
row membership changes, investigate truncation before proceeding; do not silently
repair comparability by changing the cap or dropping additional examples.

## Training Gate

After the artifact checks have been reviewed, use a fresh run:

```powershell
.venv\Scripts\python scripts\train_sft_lora.py `
  --config $config `
  --local-low-vram `
  --max-train-rows 1000 `
  --max-eval-rows 200 `
  --max-sequence-length 1024
```

The new model directory ends in `epochs2_qkvo_name_first`. Do not overwrite or
resume E's adapter. Training and evaluation are run by the owner.

## How to Judge the Result

The direct comparison is **E versus E-name-first**. A remains the balanced routing
reference, not the initialization or data source for the candidate.

| E Measured Development Reference | Value |
| --- | --- |
| Task success | 52/150 (0.347) |
| Routing macro-F1 | 0.437 |
| Invalid outputs | 14/150 (0.093) |
| Premature ACT on boundary cases | 40/69 |

Report invalid-output rate and its structural categories separately from routing
macro-F1, the full decision confusion matrix, premature-action counts and task
success. Fewer malformed calls alone do not prove better ACT/CLARIFY/REFUSE
judgment. Wording-based clarification/refusal scores are also distinct from
correct routing. Do not rescore repaired outputs as successes.

The established 150-case suite has been repeatedly inspected and is a development
benchmark, not an untouched final test. A single-seed result cannot establish a
general causal claim. No candidate result is available yet.

## Supporting Files

- [E's recipe and measured results](../07-sft-hyperparameters/03-all-four-attention-projections.md)
- [Target-only converter](../../../src/baymax/models/sft_serialization.py)
- [Experiment preparation command](../../../scripts/prepare_sft_name_first.py)
- [Regression tests](../../../tests/unit/test_sft_serialization.py)
