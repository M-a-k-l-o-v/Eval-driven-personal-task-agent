# BAYMAX Control Data Only

[Experiment index](../README.md)

## Question and Context

Would training directly on BAYMAX's tools and boundary decisions help more than generic external tool data?

"Control" here means the authored BAYMAX-control dataset. It does not mean an untrained experimental control.

## Setup

Synthetic training requests covered productive actions, clarification and refusal, using BAYMAX interfaces rather than evaluation fixtures. This checkpoint was evaluated without mixing xLAM into its training source.

The current control config is a navigation aid, not proof of the exact settings used by this historical checkpoint. The saved evaluation files do not record a complete frozen training recipe.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Control-only](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-original-v1-50.json) | 50 | 2/50 (0.040) | 0.500 | 0.578 | 0.000 | 0.071 |
| [Control-only / early parser fix](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-original-v1-50-early-parser-fix.json) | 50 | 10/50 (0.200) | 0.520 | 0.806 | 0.083 | 0.071 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Control-only](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-original-v1-50.json): 2026-08-22 UTC, 50 cases.
- [Control-only / early parser fix](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-original-v1-50-early-parser-fix.json): 2026-08-22 UTC, 50 cases.

## Interpretation

The original evaluation recorded 2/50 successes. The early-parser-fix artifact recorded 10/50. That large difference makes measurement behavior part of the explanation: it cannot be credited wholly to learning.

Even under the later measurement, tool selection and boundary behavior were weak. Purely authored data did not establish a robust replacement for external function-calling data.

## Limits

These two artifacts belong to one checkpoint investigation, not proof of two different training recipes. The precise parser-fix execution provenance is incomplete; unlike later strict rescoring, the filename alone does not establish that inference was not rerun.

Older synthetic partitioning and LoRA-initialization issues limit retrospective claims of clean controlled comparisons.

## Supporting Files

- [qwen2_5_1_5b_lora_baymax_control_v1.json](../../../configs/sft/qwen2_5_1_5b_lora_baymax_control_v1.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
