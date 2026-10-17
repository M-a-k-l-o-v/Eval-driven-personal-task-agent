# Experiment E: QKVO LoRA on All Four Attention Projections

**Status:** Completed; evaluated on 12 October 2026. Task-completion candidate, not the balanced reference.

[Experiment index](../README.md)

## Question and Controlled Change

Does adapting key and output projections alongside query and value projections
improve task completion and balanced routing under A's original two-epoch recipe?

| Setting | A: reference | E: candidate |
| --- | --- | --- |
| Target modules | q_proj, v_proj | q_proj, k_proj, v_proj, o_proj |
| Rank / alpha | 8 / 16 | 8 / 16 |
| Learning rate | 0.0002 | 0.0002 |
| Epochs | 2 | 2 |
| Initialization | Fresh pretrained Qwen + fresh LoRA | Fresh pretrained Qwen + fresh LoRA |

Only the target-module list changes, apart from run identity and output paths.
E is not the rank-16 recipe or the lower-learning-rate recipe. Broader coverage
also increases trainable adapter parameters: this is not a constant-parameter
coverage comparison. Its result cannot attribute a gain solely to module placement.

## Fixed Recipe

- Base model: Qwen/Qwen2.5-1.5B-Instruct; pretrained weights are not reinitialized.
- Frozen data: xlam-baymax-target-compatible-v14-act200-anchored-val200.
- Revision: 329fab4d8f1604a4085d3ec50e05881e7c069717bb1428db7e592743bb07f03c.
- Prompt SHA256: a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468.
- Raw training / validation caps: 1,000 / 200.
- Actual command overrides: context 1,024, train/eval batch one, accumulation 16.
- Seed 1, dropout 0.05, cosine scheduling with 3% warmup, existing PEFT initialization.
- The same assistant-label masking, target-contract checks and strict inference parser.

The JSON retains A's declared context and batch settings. The command applies the
same actual low-VRAM overrides as A. A retained 974 training and 195 validation
rows. E's final usable-row counts are not embedded in its evaluation JSON;
the reference counts are not new measurements of E.

## Training Command

Run from the repository root in PowerShell:

```powershell
$config = "configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo.json"
.venv\Scripts\python scripts\train_sft_lora.py `
  --config $config `
  --local-low-vram `
  --max-train-rows 1000 `
  --max-eval-rows 200 `
  --max-sequence-length 1024
```

The separate output directory ends in `epochs2_qkvo`. No previous model is overwritten.
Additional adapted modules may increase memory pressure. Do not silently reduce
sequence length or change quantization to finish this run; record any such change
as a different experiment.

## Results

Both runs use the established combined v1 + v2 development suite: 150 cases
with 81 ACT, 40 CLARIFY and 29 REFUSE expectations. The owner command uses
the same prompt, greedy decoding, 256 new tokens and the existing strict parser.

| Metric | A: query/value only | E: query/key/value/output |
| --- | ---: | ---: |
| Full task successes | 44/150 (0.293) | 52/150 (0.347) |
| Routing macro-F1 | 0.514 | 0.437 |
| Invalid outputs | 2/150 (0.013) | 14/150 (0.093) |
| Tool-call accuracy | 0.600 | 0.527 |
| Argument accuracy | 0.771 | 0.748 |
| Clarification wording accuracy | 0.075 | 0.113 |
| Refusal wording accuracy | 0.172 | 0.086 |
| Decision accuracy among known decisions | 0.568 | 0.581 |
| Decision coverage | 0.987 | 0.907 |
| Unavailable-tool hallucination rate | 0.020 | 0.053 |
| Average latency (ms) | 1828 | 3794 |
| Reported API cost (USD) | 0.0000 | 0.0000 |

Ratios are rounded to three decimals. Wording checks are separate from routing.
Decision accuracy excludes unknown decisions; read it alongside coverage.
Zero reported API cost excludes local hardware, electricity and training cost.

### Routing Counts

| Measurement | A | E |
| --- | ---: | ---: |
| Correct ACT decisions | 57/81 | 63/81 |
| Correct CLARIFY decisions | 11/40 | 10/40 |
| Correct REFUSE decisions | 16/29 | 6/29 |
| Premature ACT on boundary cases | 31/69 | 40/69 |
| Invalid/unknown decisions | 2/150 | 14/150 |

Premature ACT here is the count derived from the CLARIFY/REFUSE matrix rows,
not necessarily the scorer's failure-reason count.

### A: Reference Decision Matrix

Rows are expected decisions; columns are recorded predictions. The diagonal
measures routing correctness, not full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 57 | 21 | 3 | 0 |
| CLARIFY | 20 | 11 | 9 | 0 |
| REFUSE | 11 | 0 | 16 | 2 |

### E: QKVO Decision Matrix

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 63 | 12 | 2 | 4 |
| CLARIFY | 27 | 10 | 2 | 1 |
| REFUSE | 13 | 1 | 6 | 9 |

### Invalid-Output Inspection

| Observed issue | Expected cases | Responses |
| --- | --- | ---: |
| refuse_request put reason outside arguments | REFUSE | 6 |
| request_clarification had empty arguments | CLARIFY 1, REFUSE 2 | 3 |
| Productive calls omitted or separated names from arguments | ACT | 4 |
| Other malformed output | REFUSE | 1 |

Six of the nine invalid REFUSE-case outputs selected the refusal meta-tool name,
but supplied reason at the call's top level instead of inside arguments:

```json
{"name": "refuse_request", "arguments": {}, "reason": "Unsupported operation."}
```

The contract requires:

```json
{"name": "refuse_request", "arguments": {"reason": "Unsupported operation."}}
```

These are generic illustrations of the nesting defect, not repaired benchmark
responses. Selecting refuse_request does not establish that its explanation meets
the scenario's wording requirements. No rejected response received success credit.

## Interpretation and Decision

E completed eight more tasks than A, the largest task-success gain among the
A/B/C/E ablations. However, macro-F1 fell, invalid outputs rose, and latency was
about 2.1 times A's measured latency. E is not a better balanced replacement.

**Keep A as the balanced reference and E as a task-completion candidate.** A
substantial part of E's observed refusal loss involves invalid expression of an
attempted refusal. This supports investigating output-contract learning separately;
it does not erase the other incorrect decisions or establish a hypothetical repaired score.

The next proposed test is an independently authored output-contract learning
experiment with the policy class quantities and other training settings held fixed.
No data adjustment or new run has been made by this documentation task.

## Provenance and Limits

- [E evaluation JSON](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-qkvo-train1k-val200-epochs2-combined-150.json): recorded 2026-10-12T19:19:35.198791+00:00.
- [A evaluation JSON](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-epochs2-combined-150.json): recorded 2026-10-06T15:29:45.372202+00:00.
- E canonical config SHA256: ee2c1b9efdec057796d0c0a9d16b20f75763151fcc34e6fe95ec004bc3bf00e4.

Metrics and matrices were checked with the existing comparison helper. Raw
generations were inspected read-only. Original scores, results and parser rules
remain unchanged. These are single-seed development observations, not an untouched
final test. Adding modules also increases adapter size, so this does not isolate
placement at equal parameter count. Full training/generation metadata and exact
base-model revision are not embedded in the result JSON.

## Supporting Files

- [E config](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo.json)
- [A: selected reference](../04-training-duration-and-contrasts/01-anchored-data-at-two-epochs.md)
