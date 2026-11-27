# The Same 2k Pool at Four Epochs

**Status:** Owner training/evaluation completed; final reported local candidate at project closeout.
Date: 2026-10-17.

## Question and Controls

The owner requested a fresh four-epoch run on the same
[expanded 2k dataset](19-unique-2k-training-pool.md).
Compare against that dataset's two-epoch run, not the earlier 1k duration trial.
The unweighted compact/name-first checkpoint remains the historical development
reference. The four-epoch candidate's measured result is recorded below; no
statistical significance or untouched-test generalization is established.

Only `num_train_epochs` changes from 2 to 4. Run/config identity, output paths and
config hash also change so the two-epoch artifact is not overwritten. Data, split
membership, prompt, LoRA, learning rate, seed, loss weighting and laptop overrides
stay fixed. This starts from the base model with a fresh LoRA adapter, not from the
saved two-epoch adapter. The cosine schedule spans the longer training horizon,
so this is not an identical first two epochs followed by extra updates.

## Identity

| Item | Value |
| --- | --- |
| Config | `configs/sft/qwen2_5_1_5b_lora_coverage_2k_v2_epochs4.json` |
| Config SHA256 | `da38eb64730f8db30b1bf1a5d6f5296a805351598ca2057a1f70e6874141f2e5` |
| Dataset | `data/sft/v1/name-first-coverage-2k-v2` |
| Dataset revision | `43892841fad91d9c328f3577e831c5d192242f84ceabe82d18e7deb30d2c6ab4` |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| LoRA | r=8, alpha=16, Q/K/V/O, dropout=0.05 |
| Learning rate / epochs / seed | 2e-4 / 4 / 1 |
| First-tool loss weighting | 1x, unweighted |
| Laptop settings | Context 1,024, batch one, accumulation 16 |
| Output | `models/qwen2_5_1_5b_lora_coverage_2k_v2_epochs4/adapter` |

The owner reported 1,952 usable training rows from a 2,000-row raw cap: 800 BAYMAX,
1,152 xLAM, 200 clarification and 200 refusal rows, with zero exact duplicates.
The same validation cap yields 195 usable rows. These are existing artifact counts,
not a new four-epoch measurement. At the same usable counts, four epochs give
approximately twice the sample exposure of two epochs. Model quality is measured
separately by the saved evaluation below.

## Owner Command

```powershell
.venv\Scripts\python scripts\train_sft_lora.py `
  --config configs\sft\qwen2_5_1_5b_lora_coverage_2k_v2_epochs4.json `
  --local-low-vram `
  --max-train-rows 2000 `
  --max-eval-rows 200 `
  --max-sequence-length 1024
```

No dataset rebuild or additional examples are needed. The coding agent did not
run training or evaluation. Preserve raw results and compare both durations under
the same compact JSON input, greedy decoding, 150-scenario suite and scoring.
Report task success, routing macro-F1, invalid outputs and the decision matrix;
lower training loss alone is not a model-quality win. This repeatedly inspected
suite is development evidence, not an untouched final test.

## Saved Evaluation Results

Sources: [two epochs](../../../results/phase2/json-spacing/name-first-coverage-2k-v2-compact-combined-150.json)
and [four epochs](../../../results/phase2/json-spacing/name-first-coverage-2k-v2-epochs4-compact-combined-150.json).
The four-epoch result is timestamped 2026-10-17T16:52:19Z. Both use the same 150
scenarios, compact JSON, temperature zero, at most 256 generated tokens and fake
adapters. Values below are transcribed from saved artifacts, not re-evaluated.

| Measurement | Two epochs | Four epochs |
| --- | ---: | ---: |
| Task successes / 150 | 64 | 73 |
| Task success rate | 0.427 | 0.487 |
| Routing macro-F1 | 0.615 | 0.681 |
| Invalid outputs / 150 | 2 | 0 |
| Tool-call accuracy | 0.667 | 0.753 |
| Argument accuracy | 0.806 | 0.836 |
| Clarification wording accuracy | 0.163 | 0.350 |
| Refusal wording accuracy | 0.603 | 0.690 |
| Conditional decision accuracy | 0.649 | 0.707 |
| Decision coverage | 0.987 | 1.000 |
| Hallucination rate | 0.000 | 0.007 |
| Mean inference latency, ms | 2,758 | 5,652 |
| Reported API cost, USD | 0.0000 | 0.0000 |

Four epochs, expected rows and predicted columns:

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 62 | 13 | 6 | 0 |
| CLARIFY | 17 | 17 | 6 | 0 |
| REFUSE | 1 | 1 | 27 | 0 |

The recorded four-epoch run completes nine additional tasks and has better
macro-F1 than the two-epoch counterpart. Compared with the historical unweighted
1k reference, it completes 73 versus 57 tasks and records macro-F1 0.681 versus
0.612. This is a useful development result for closing the project, not proof that
four epochs always help: the earlier 1k four-epoch trial regressed, and no repeated
training seeds were measured here.

Clarification remains incomplete: 17/40 requests that should clarify still cause
ACT, and 6/40 cause REFUSE. ACT recall is 62/81, and refusal recall is 27/29.
The recorded latency is higher; hardware/background conditions were not controlled
well enough to attribute that solely to the training duration. A zero local API
cost field does not mean zero hardware, energy or training cost.

The project owner accepted the work as complete on 2026-10-17. This does not imply
production readiness, real integrations, a new MLX checkpoint release or an
untouched final evaluation. Model weights and generated training data remain
local and are not part of the Git closeout.
