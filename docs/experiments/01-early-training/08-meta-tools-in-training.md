# Clarification and Refusal as Explicit Meta-Tools

[Experiment index](../README.md)

## Question and Context

Was the model failing to abstain because clarification and refusal were not presented as recognizable actions?

## Setup

The synthetic-v8 weighted path represented non-productive decisions as explicit calls:

- request_clarification with arguments.question.
- refuse_request with arguments.reason.

The local runner converts those calls into a BAYMAX response decision and message, rather than executing them as external mutations.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Meta-tool mixture](../../../results/phase2/qwen-lora/baymax-mix/synthetic-v8-meta-tools-weighted-v1-50.json) | 50 | 12/50 (0.240) | 0.540 | 0.840 | 0.000 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Meta-tool mixture](../../../results/phase2/qwen-lora/baymax-mix/synthetic-v8-meta-tools-weighted-v1-50.json): 2026-09-15 UTC, 50 cases.

## Interpretation

The run completed 12/50 tasks, but both wording metrics were zero. Adding action names alone did not resolve the problem.

The next important investigation was whether the model saw the same prompt, tool definitions and structured request wrapper during training and evaluation.

## Limits

A zero keyword metric does not alone show that every model decision was ACT. Explicit routing metadata was not available in this artifact. The exact oversampling recipe is also not preserved.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
