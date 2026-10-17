# Experiment C: Lower Learning Rate at Two Epochs

**Status:** Completed; evaluated on 12 October 2026. A remains the balanced reference.

[Experiment index](../README.md)

## Question and Context

Does halving the learning rate improve balanced ACT / CLARIFY / REFUSE routing
without sacrificing productive task completion?

[Experiment B](01-rank16-at-fixed-two-epoch-recipe.md) increased adapter rank but
worsened routing and validity. C starts from A's original recipe, not B's rank-16
settings and not a saved trained adapter. Smaller updates preserving decision
boundaries is a hypothesis to test, not an established explanation of earlier runs.

## Controlled Change

| Setting | A: reference | C: candidate |
| --- | --- | --- |
| Peak learning rate | 0.0002 | 0.0001 |
| LoRA rank / alpha | 8 / 16 | 8 / 16 |
| Target modules | q_proj, v_proj | q_proj, v_proj |
| SFT epochs | 2 | 2 |
| Initialization | Fresh Qwen base + fresh LoRA | Fresh Qwen base + fresh LoRA |

Only the learning rate changes, apart from the run name, config hash and output
paths needed to keep the experiment separate. Cosine scheduling and 3% warmup
remain unchanged; C halves the schedule's learning-rate amplitude.

## Fixed Data and Actual Overrides

- Base model: Qwen/Qwen2.5-1.5B-Instruct.
- Data: xlam-baymax-target-compatible-v14-act200-anchored-val200, unchanged splits.
- Revision: 329fab4d8f1604a4085d3ec50e05881e7c069717bb1428db7e592743bb07f03c.
- Prompt SHA256: a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468.
- Raw training / validation caps: 1,000 / 200.
- Actual sequence limit: 1,024; train/eval batch one; gradient accumulation 16.
- Seed 1, dropout 0.05, assistant-only token cross-entropy and the same masking.

The command below applies the same low-VRAM overrides as A; the JSON retains
A's declared context and batch settings. A retained 974 train and 195 validation
rows after token filtering. C's final usable-row counts are not embedded in its
evaluation JSON; the reference counts are not new measurements of C.

## Training Command

Run from the repository root in PowerShell:

```powershell
$config = "configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_lr1e-4.json"
.venv\Scripts\python scripts\train_sft_lora.py `
  --config $config `
  --local-low-vram `
  --max-train-rows 1000 `
  --max-eval-rows 200 `
  --max-sequence-length 1024
```

The new output directory ends in `epochs2_lr1e-4`; neither A nor B is overwritten.

## Results

Both runs use the established 150-case v1 + v2 development suite: 81 ACT,
40 CLARIFY and 29 REFUSE expectations. The owner command uses the same prompt,
greedy decoding, a 256 new-token limit and the existing strict parser.

| Metric | A: learning rate 0.0002 | C: learning rate 0.0001 |
| --- | ---: | ---: |
| Full task successes | 44/150 (0.293) | 45/150 (0.300) |
| Routing macro-F1 | 0.514 | 0.436 |
| Invalid outputs | 2/150 (0.013) | 15/150 (0.100) |
| Tool-call accuracy | 0.600 | 0.513 |
| Argument accuracy | 0.771 | 0.726 |
| Clarification wording accuracy | 0.075 | 0.050 |
| Refusal wording accuracy | 0.172 | 0.172 |
| Decision accuracy among known decisions | 0.568 | 0.560 |
| Decision coverage | 0.987 | 0.893 |
| Unavailable-tool hallucination rate | 0.020 | 0.010 |
| Average latency (ms) | 1828 | 2492 |
| Reported API cost (USD) | 0.0000 | 0.0000 |

Ratios are rounded to three decimals. Wording metrics are not routing metrics.
Zero reported API cost excludes local hardware, electricity and training cost.

### Routing Counts

| Measurement | A | C |
| --- | ---: | ---: |
| Correct ACT decisions | 57/81 | 58/81 |
| Correct CLARIFY decisions | 11/40 | 4/40 |
| Correct REFUSE decisions | 16/29 | 13/29 |
| Premature ACT on boundary cases | 31/69 | 37/69 |
| Invalid/unknown decisions | 2/150 | 16/150 |

Premature ACT is derived from the matrix's CLARIFY/REFUSE rows, not necessarily
the count of the scorer's identically themed failure-reason label.

### A: Reference Decision Matrix

Rows are expected decisions; columns are recorded predictions. The diagonal
measures correct routing, not full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 57 | 21 | 3 | 0 |
| CLARIFY | 20 | 11 | 9 | 0 |
| REFUSE | 11 | 0 | 16 | 2 |

### C: Lower-Learning-Rate Decision Matrix

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 58 | 13 | 5 | 5 |
| CLARIFY | 24 | 4 | 4 | 8 |
| REFUSE | 13 | 0 | 13 | 3 |

### Invalid/Unknown Output Inspection

| Observed issue | Responses |
| --- | ---: |
| request_clarification used reason instead of question | 10 |
| Calls omitted tool names | 4 |
| Incomplete JSON | 1 |
| Empty array with no decision | 1 |

The ten wrong-field calls include six expected CLARIFY cases, two ACT cases
and two REFUSE cases. An attempted clarification is not necessarily correct routing.
The empty array parses but expresses no decision; that explains 15 invalid
outputs versus 16 invalid/unknown decisions. None were repaired or rescored.

The [target-contract audit](../06-measurement-and-data-audits/06-clarification-output-contract-audit.md)
checked the selected 1,000 raw training and 200 validation targets. All 100 training
and 30 validation clarification targets correctly use question. This does not
establish that the model learned the contract or that every target is semantically perfect.

## Interpretation and Decision

C gained one full task success, but macro-F1, argument accuracy and output
validity worsened. The lower rate did not preserve the boundaries better in
this run. Keep A as the balanced reference; do not promote C on its one-task gain.

Some apparent clarification losses are invalid expression of an attempted
clarification, not only selection of the wrong decision. Output-contract learning
should be investigated separately without weakening the strict parser. This
experiment alone does not prove that the lower rate caused undertraining or
that more epochs would resolve the defects.

## Provenance and Limits

- [C evaluation JSON](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-lr1e-4-train1k-val200-epochs2-combined-150.json): recorded 2026-10-12T17:34:32.282617+00:00.
- [A evaluation JSON](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-epochs2-combined-150.json): recorded 2026-10-06T15:29:45.372202+00:00.
- C canonical config SHA256: 80b81f44f0daf9f4d18ba8f29eb7139a216d72c5ecec6b5839ffd77e40eeee02.

Metrics and matrices were checked using the existing comparison helper on saved
artifacts. No training, inference or result modification was performed to write
this note. Results are single-seed observations on repeatedly inspected development
suites, not an untouched final benchmark. Exact base-model revision and complete
training/generation metadata are not embedded in the result JSON.

## Supporting Files

- [Candidate config](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_lr1e-4.json)
- [A: selected rank-8 two-epoch reference](../04-training-duration-and-contrasts/01-anchored-data-at-two-epochs.md)
