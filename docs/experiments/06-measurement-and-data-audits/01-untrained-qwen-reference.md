# Untrained Qwen: What the Baseline Actually Measures

[Experiment index](../README.md)

## Question and Context

Does fine-tuning help compared with the same local model without an adapter?

This note records the bare Qwen reference. Hosted GPT and scripted results are different baselines: hosted inference changes the model, while scripted responses test the harness rather than learned behavior.

## Setup

Base Qwen2.5-1.5B-Instruct, no LoRA adapter, evaluated through the local runner and fake tools.

Use the training-prompt-a314 artifact when comparing with similarly prompted historical adapters. The "historical" artifact predates some policy/scoring changes and is not interchangeable.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Bare Qwen / historical settings](../../../results/phase2/qwen-base/qwen-base-combined-150-historical.json) | 150 | 32/150 (0.213) | 0.500 | 0.791 | 0.036 | 0.019 |
| [Bare Qwen / matched training prompt](../../../results/phase2/qwen-base/qwen-base-combined-150-training-prompt-a314.json) | 150 | 32/150 (0.213) | 0.493 | 0.791 | 0.025 | 0.034 |
| [Bare Qwen](../../../results/phase2/template-shift/qwen-base-template-shift-30.json) | 30 | 7/30 (0.233) | 0.483 | 0.947 | 0.000 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Bare Qwen / historical settings](../../../results/phase2/qwen-base/qwen-base-combined-150-historical.json): 2026-09-26 UTC, 150 cases.
- [Bare Qwen / matched training prompt](../../../results/phase2/qwen-base/qwen-base-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.
- [Bare Qwen](../../../results/phase2/template-shift/qwen-base-template-shift-30.json): 2026-09-27 UTC, 30 cases.

## Interpretation

The historical combined suite measured 32/150 task successes. The matched-training-prompt evaluation also had 32/150, but some tool/wording metrics differed. Template-shift success was 7/30.

This establishes that "same task-success rate" can hide different behavior. It also explains why 50-, 150- and 30-case percentages should not be placed in one improvement curve.

## Limits

These base results predate later strict-parser measurements. A clean claim that a modern checkpoint beats base needs a method-matched base artifact, not simply the largest improvement over any historical baseline.

Reported local API cost of zero does not include electricity, hardware or training cost.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
