# Control Data With Targeted Refusals

[Experiment index](../README.md)

## Question and Context

Could explicit refusal examples reduce unsupported actions when early control training still rarely refused?

## Setup

Refusal-focused synthetic examples were added to the control-data path. A full 50-case evaluation and later seven-case prompt diagnostics survive.

The seven-case outputs are refusal-only diagnostics, not 50-case benchmarks. Some of those tests changed the inference prompt or parser; their numbers must not be attributed solely to training.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Targeted refusals](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-targeted-refusals-v1-50.json) | 50 | 12/50 (0.240) | 0.500 | 0.759 | 0.111 | 0.143 |
| [Targeted refusals / prompt-only](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-targeted-refusals-prompt-only-subset-7.json) | 7 | 0/7 (0.000) | 0.429 | 1.000 | n/a | 0.000 |
| [Control / policy prompt](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-policy-prompt-refusal-subset-7.json) | 7 | 0/7 (0.000) | 0.429 | 1.000 | n/a | 0.071 |
| [Control / policy prompt / early parser fix](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-policy-prompt-refusal-subset-7-early-parser-fix.json) | 7 | 1/7 (0.143) | 0.429 | 1.000 | n/a | 0.143 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Targeted refusals](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-targeted-refusals-v1-50.json): 2026-08-22 UTC, 50 cases.
- [Targeted refusals / prompt-only](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-targeted-refusals-prompt-only-subset-7.json): 2026-09-15 UTC, 7 cases.
- [Control / policy prompt](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-policy-prompt-refusal-subset-7.json): 2026-09-15 UTC, 7 cases.
- [Control / policy prompt / early parser fix](../../../results/phase2/qwen-lora/baymax-control/synthetic-control-policy-prompt-refusal-subset-7-early-parser-fix.json): 2026-09-15 UTC, 7 cases.

## Interpretation

The full run completed 12/50 tasks and had refusal wording accuracy 0.143. Most seven-case diagnostics still recorded no strict task success; the later policy-prompt/parser-fix diagnostic recorded one.

Giving the model refusal examples did not by itself resolve boundary routing or explanation quality.

## Limits

The saved files do not fully recover the authored-data changes or repeat counts. Single-class subset rates cannot establish balanced ACT/CLARIFY/REFUSE performance. Parser and prompt changes are additional variables.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
