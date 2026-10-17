# External Tool Data Only: First Local Fine-Tune

[Experiment index](../README.md)

## Question and Context

Could ordinary function-calling SFT make local Qwen better at BAYMAX tasks before adding domain-specific clarification and refusal training?

This is the xLAM-only starting point. Its files live under Phase 1 because that is where the first local comparison was stored; it belongs in the training history nonetheless.

## Setup

- Base model: Qwen/Qwen2.5-1.5B-Instruct.
- External source: Salesforce/xLAM-function-calling-60k, downloaded through Hugging Face.
- Owner's medium-run logs describe a 1,000-row training cap, 200 validation rows, one epoch, context 1,024, batch one and gradient accumulation 16.
- LoRA requested: rank 8, alpha 16, q_proj/v_proj, dropout 0.05, no bias.
- The original medium run reported NaN validation loss. A later fixed run was evaluated; the saved result names do not establish which precise changes explain the fix.
- Bare Qwen is the comparison, not the scripted baseline or hosted GPT model.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Bare Qwen: full suite](../../../results/phase1/qwen-base/full.json) | 50 | 11/50 (0.220) | 0.540 | 0.869 | 0.000 | 0.071 |
| [xLAM-only LoRA: full suite](../../../results/phase1/qwen-lora/xlam-medium/full.json) | 50 | 12/50 (0.240) | 0.480 | 0.854 | 0.028 | 0.000 |
| [xLAM-only LoRA: legacy full suite](../../../results/phase1/qwen-lora/xlam-medium/legacy-full.json) | 50 | 12/50 (0.240) | 0.480 | 0.854 | 0.028 | 0.000 |
| [Bare Qwen: one-case check](../../../results/phase1/qwen-base/calendar-create-only.json) | 1 | 0/1 (0.000) | 1.000 | 0.500 | n/a | n/a |
| [Bare Qwen: four-case smoke](../../../results/phase1/qwen-base/smoke-4-scenarios.json) | 4 | 0/4 (0.000) | 0.250 | 0.875 | 0.000 | n/a |
| [xLAM-only LoRA: one-case check](../../../results/phase1/qwen-lora/xlam-medium/calendar-create-only.json) | 1 | 0/1 (0.000) | 1.000 | 0.750 | n/a | n/a |
| [xLAM-only LoRA: three-case smoke](../../../results/phase1/qwen-lora/xlam-medium/limit-3.json) | 3 | 0/3 (0.000) | 0.000 | 1.000 | 0.000 | n/a |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Bare Qwen: full suite](../../../results/phase1/qwen-base/full.json): 2026-08-11 UTC, 50 cases.
- [xLAM-only LoRA: full suite](../../../results/phase1/qwen-lora/xlam-medium/full.json): 2026-08-11 UTC, 50 cases.
- [xLAM-only LoRA: legacy full suite](../../../results/phase1/qwen-lora/xlam-medium/legacy-full.json): 2026-08-11 UTC, 50 cases.
- [Bare Qwen: one-case check](../../../results/phase1/qwen-base/calendar-create-only.json): 2026-08-11 UTC, 1 cases.
- [Bare Qwen: four-case smoke](../../../results/phase1/qwen-base/smoke-4-scenarios.json): 2026-08-11 UTC, 4 cases.
- [xLAM-only LoRA: one-case check](../../../results/phase1/qwen-lora/xlam-medium/calendar-create-only.json): 2026-08-11 UTC, 1 cases.
- [xLAM-only LoRA: three-case smoke](../../../results/phase1/qwen-lora/xlam-medium/limit-3.json): 2026-08-11 UTC, 3 cases.

## What We Learned

The full-suite gain was one additional task: 12/50 versus 11/50 for bare Qwen. Tool-call accuracy fell, and clarification/refusal remained very weak. This was evidence that training and local inference could run, not a convincing policy-learning result.

The one-, three- and four-case files are smoke diagnostics. They should not be compared to the full-suite rates as if they were additional full experiments.

## Limits

These are historical parser/prompt measurements. Exact base revision and complete runtime provenance are not recoverable from these result files alone. The legacy and later full artifacts have the same aggregate success but different latency; that does not prove identical generations.

The NaN loss was a training diagnostic, not evidence that all later checkpoints were invalid. Do not infer a causal fix from filenames alone.

## Supporting Files

- [qwen2_5_1_5b_lora_v1.json](../../../configs/sft/qwen2_5_1_5b_lora_v1.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
