# Extra Clarification and Refusal Repeats

[Experiment index](../README.md)

## Question and Context

The aligned model still acted prematurely. Could greater exposure to boundary examples help without changing LoRA settings?

## Change

The named recipe added six extra clarification repeats and twelve extra refusal repeats. Repeats increase exposure to existing examples; they are not six and twelve newly authored unique requests.

Post-mix token audits were retained in two snapshots. Treat each snapshot as evidence about its reported artifact, not automatic provenance for the evaluated checkpoint.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Extra boundary repeats](../../../results/phase2/qwen-lora/baymax-mix/clarification-extra6-refusal-extra12-v1-50.json) | 50 | 8/50 (0.160) | 0.420 | 0.689 | 0.028 | 0.000 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Extra boundary repeats](../../../results/phase2/qwen-lora/baymax-mix/clarification-extra6-refusal-extra12-v1-50.json): 2026-09-16 UTC, 50 cases.

## Interpretation

Success was 8/50, below the historical aligned 14/50. Clarification wording accuracy remained low and refusal was zero.

This trial did not support continuing to increase repetition blindly. It motivated examining what actually reached training after shuffling, capping and label filtering.

## Limits

The saved evaluation does not pin every training knob or the effective cap. Exact snapshot-to-checkpoint identity is incomplete. No memorization diagnosis follows from repetition counts alone.

## Supporting Files

- [clarification-extra6-refusal-extra12-token-audit-5000-snapshot1.json](../../../results/phase2/analysis/clarification-extra6-refusal-extra12-token-audit-5000-snapshot1.json)
- [clarification-extra6-refusal-extra12-token-audit-5000-snapshot2.json](../../../results/phase2/analysis/clarification-extra6-refusal-extra12-token-audit-5000-snapshot2.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
