# Original External and BAYMAX Training Mix

[Experiment index](../README.md)

## Question and Context

Could generic xLAM tool-use training and BAYMAX-specific boundary examples complement each other?

This is the early "old mix" reference. It is not the later format-aligned or anchored reference.

## Setup

Training combined external xLAM examples with authored BAYMAX control examples. The source recipes evolved, and the result files do not preserve a complete immutable account of the exact selected rows or sampling weights.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Original mixture](../../../results/phase2/qwen-lora/baymax-mix/original-xlam-control-mix-v1-50.json) | 50 | 12/50 (0.240) | 0.420 | 0.829 | 0.028 | 0.000 |
| [Original mixture / early parser fix](../../../results/phase2/qwen-lora/baymax-mix/original-xlam-control-mix-v1-50-early-parser-fix.json) | 50 | 12/50 (0.240) | 0.420 | 0.829 | 0.028 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Original mixture](../../../results/phase2/qwen-lora/baymax-mix/original-xlam-control-mix-v1-50.json): 2026-08-22 UTC, 50 cases.
- [Original mixture / early parser fix](../../../results/phase2/qwen-lora/baymax-mix/original-xlam-control-mix-v1-50-early-parser-fix.json): 2026-08-22 UTC, 50 cases.

## Interpretation

Both historical evaluations recorded 12/50 task successes, clarification accuracy 0.028 and refusal accuracy zero. Mixing the sources did not automatically teach abstention.

The initial source formats were not fully matched to evaluation: plain user queries and generic function arrays were being compared with structured time/state requests and BAYMAX control decisions. The later format-alignment experiment addresses that distribution shift.

## Limits

Identical aggregate numbers in the early-parser-fix file are not proof that every output was identical. These artifacts predate later parser and split-integrity protections.

Use the linked original token audit for the distribution observed in its own 5,000-row artifact, not as a description of every early mixed checkpoint.

## Supporting Files

- [original-mix-token-audit-5000-and-boundary-failures.json](../../../results/phase2/analysis/original-mix-token-audit-5000-and-boundary-failures.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
