# Aligning Training and Evaluation Formats

[Experiment index](../README.md)

## Question and Context

Was a training/evaluation format mismatch preventing the model from using clarification and refusal?

Generic examples taught a plain request plus tools. BAYMAX evaluation supplied a structured request with current_time, initial_state and request, and injected two meta-tools. Those are meaningfully different input distributions.

## Change

Align the structured wrapper, shared meta-tool definitions, names, ordering and system-prompt expectations between training and local evaluation. Non-ACT targets remain actual JSON calls, not prose-only answers.

This checkpoint became the historical "aligned" reference. It is different from the later target-compatible anchored two-epoch reference.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Format-aligned](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-reference-v1-50.json) | 50 | 14/50 (0.280) | 0.580 | 0.853 | 0.056 | 0.071 |
| [Format-aligned / historical settings](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-historical-prompt.json) | 150 | 42/150 (0.280) | 0.473 | 0.771 | 0.083 | 0.019 |
| [Format-aligned / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-training-prompt-a314.json) | 150 | 40/150 (0.267) | 0.453 | 0.771 | 0.025 | 0.017 |
| [Format-aligned](../../../results/phase2/template-shift/format-aligned-template-shift-30.json) | 30 | 5/30 (0.167) | 0.517 | 0.817 | 0.100 | 0.100 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Format-aligned](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-reference-v1-50.json): 2026-09-16 UTC, 50 cases.
- [Format-aligned / historical settings](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-historical-prompt.json): 2026-09-26 UTC, 150 cases.
- [Format-aligned / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/format-aligned-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.
- [Format-aligned](../../../results/phase2/template-shift/format-aligned-template-shift-30.json): 2026-09-27 UTC, 30 cases.

## Interpretation

On v1-50, success was 14/50 and clarification/refusal wording scores became nonzero. On the historical combined-150 evaluation, success was 42/150.

When tested with the recorded training prompt on combined-150, success was 40/150; template-shift success was 5/30. Alignment helped the engineering contract, but generalization remained weak.

## Limits

The evaluations differ in suite, prompt and policy-era scoring; they are not repeated measurements of one fixed benchmark setup. Historical results do not fully pin selected training rows or base-model revision.

A format correction is necessary infrastructure, not proof that the decision policy has been learned robustly.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
