# Experiment B: Rank 16 at the Fixed Two-Epoch SFT Recipe

**Status:** Completed; evaluated on 12 October 2026. Rank 8 remains the reference.

[Experiment index](../README.md)

## Context

The [anchored two-epoch SFT reference](../04-training-duration-and-contrasts/01-anchored-data-at-two-epochs.md)
remains our reference for balanced ACT / CLARIFY / REFUSE routing. More SFT epochs
and the DPO trials have not established a better balanced model. This experiment
tests adapter capacity without changing the data or training objective.

## What Changes

| Setting | A: reference | B: candidate |
| --- | --- | --- |
| LoRA rank | 8 | 16 |
| LoRA alpha | 16 | 32 |
| Alpha / rank | 2 | 2 |
| Target modules | q_proj, v_proj | q_proj, v_proj |

Rank and alpha change together to retain the existing LoRA scaling ratio.
This doubles the trainable adapter parameter count for the same target modules.
It does not test adding k_proj/o_proj, lowering the learning rate or adding data.

## Fixed Recipe

| Setting | Both runs |
| --- | --- |
| Initialization | Fresh Qwen/Qwen2.5-1.5B-Instruct, not a continued adapter |
| Dataset | xlam-baymax-target-compatible-v14-act200-anchored-val200 |
| Dataset revision | 329fab4d8f1604a4085d3ec50e05881e7c069717bb1428db7e592743bb07f03c |
| Prompt SHA256 | a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468 |
| Raw caps | 1,000 training / 200 validation rows |
| Actual sequence limit | 1,024 with the command below |
| Actual batch / accumulation | 1 / 16 with --local-low-vram |
| Learning rate / epochs | 0.0002 / 2 |
| Dropout / seed | 0.05 / 1 |
| Scheduler / warmup | Cosine / 0.03 |

The JSON retains the reference's declared 2,048 context and batch settings;
the same command-line overrides reproduce its actual low-VRAM recipe.
The reference retained 974 training and 195 validation rows after token filtering.
The candidate's final usable-row counts are not embedded in its evaluation JSON;
the reference counts must not be presented as newly measured candidate counts.

## Run

From the repository root in PowerShell:

```powershell
$config = "configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_rank16.json"
.venv\Scripts\python scripts\train_sft_lora.py `
  --config $config `
  --local-low-vram `
  --max-train-rows 1000 `
  --max-eval-rows 200 `
  --max-sequence-length 1024
```

The candidate writes to its own `models` directory ending in `epochs2_rank16`.
Do not overwrite the reference adapter or reduce context to work around memory
pressure without recording that as a different experiment.

## Results

Both runs use the established combined v1 + v2 development suite: 150 cases,
with 81 ACT, 40 CLARIFY and 29 REFUSE expectations. The supplied evaluation
command uses greedy decoding, a 256 new-token limit and the existing strict parser.

| Metric | A: rank 8 | B: rank 16 |
| --- | ---: | ---: |
| Full task successes | 44/150 (0.293) | 46/150 (0.307) |
| Routing macro-F1 | 0.514 | 0.450 |
| Invalid outputs | 2/150 (0.013) | 7/150 (0.047) |
| Tool-call accuracy | 0.600 | 0.547 |
| Argument accuracy | 0.771 | 0.785 |
| Clarification wording accuracy | 0.075 | 0.050 |
| Refusal wording accuracy | 0.172 | 0.103 |
| Decision accuracy among known decisions | 0.568 | 0.552 |
| Decision coverage | 0.987 | 0.953 |
| Unavailable-tool hallucination rate | 0.020 | 0.053 |
| Average latency (ms) | 1828 | 1980 |
| Reported API cost (USD) | 0.0000 | 0.0000 |

Ratios are rounded to three decimals. Wording scores are separate from routing:
a correct clarification label can still miss the required question details.
Local API cost zero does not mean training, hardware or electricity are free.

### Routing Counts

| Measurement | A: rank 8 | B: rank 16 |
| --- | ---: | ---: |
| Correct ACT decisions | 57/81 | 60/81 |
| Correct CLARIFY decisions | 11/40 | 11/40 |
| Correct REFUSE decisions | 16/29 | 8/29 |
| Premature ACT on CLARIFY/REFUSE cases | 31/69 | 42/69 |

Premature ACT here is the matrix count of productive decisions on boundary
requests, not the potentially different count of the scorer's failure-reason label.

### A: Rank-8 Decision Matrix

Rows are expected decisions; columns are recorded predictions. The diagonal
measures correct routing, not full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 57 | 21 | 3 | 0 |
| CLARIFY | 20 | 11 | 9 | 0 |
| REFUSE | 11 | 0 | 16 | 2 |

### B: Rank-16 Decision Matrix

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 60 | 14 | 2 | 5 |
| CLARIFY | 23 | 11 | 6 | 0 |
| REFUSE | 19 | 0 | 8 | 2 |

Invalid/unknown is the decision-matrix category. Invalid-output rate is computed
separately from parser failures; the counts happen to match in these two runs.

## Interpretation and Decision

The candidate completed two more tasks and slightly improved argument accuracy,
but routing macro-F1 fell by 0.064. Correct refusals halved, premature ACT rose
by eleven cases, and invalid outputs increased from two to seven. Correct
clarification routing did not improve. Rank 16 is not a better balanced replacement.

**Keep A, the rank-8 two-epoch SFT model, as the reference.** This run does not
support insufficient adapter capacity as the main problem. It does not prove
capacity never matters, and a single-seed result cannot establish the cause of
the regression or whether a two-task difference is reproducible.

The next test recommended after B was C: return to rank 8 / alpha 16 and lower
only the learning rate from 2e-4 to 1e-4. That test has now been run; see
[Experiment C](02-lower-learning-rate-at-two-epochs.md) for its results. B remains
a separate completed experiment, not part of a combined rank-and-learning-rate change.

## Sources and Reproducibility

| Artifact | Purpose |
| --- | --- |
| [A evaluation JSON](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-train1k-val200-epochs2-combined-150.json) | Original reference results |
| [B evaluation JSON](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-anchored-rank16-train1k-val200-epochs2-combined-150.json) | Rank-16 results, recorded at 2026-10-12T16:58:08.530056+00:00 |
| [B training config](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_rank16.json) | Candidate settings and data identity |
| [Comparison helper](../../../scripts/compare_eval_results.py) | Reads saved metrics and derives routing from recorded decisions |

B canonical config SHA256:
`a09ca9973edbc9127f7641cc5e10cb809baf18f13a3504c011350835fc0bd663`.

Numbers were checked against the saved result files using the comparison helper.
No training, inference or result-file rewriting was performed to record this note.
These repeatedly inspected suites are development comparisons, not an untouched
final test. Exact base-model revision and complete generation/training metadata
are not embedded in the result JSON, which limits retrospective verification.
