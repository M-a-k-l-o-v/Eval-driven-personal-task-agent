# Increasing the Cap to 1,500 Without Repeating Boundaries

[Experiment index](../README.md)

## Question and Context

Would more unique external training data help once the sampler stopped duplicating the fixed BAYMAX boundary pool?

## Change

Increase the raw cap from 1,000 to 1,500, keeping BAYMAX at 150, including the same absolute quotas of 60 refusals and 45 clarifications. Extra capacity is filled primarily with unique xLAM rows.

The reported usable artifact had 1,447 rows: 657 direct, 685 multi-tool, 60 refusal and 45 clarification. Source counts were BAYMAX 150 and xLAM 1,297, with zero exact duplicates.

Refusal/clarification token shares fell as generic ACT exposure increased. A fixed row contribution does not mean a fixed class proportion.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Boundary pool / 1,500 cap](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-v1-50.json) | 50 | 10/50 (0.200) | 0.500 | 0.725 | 0.056 | 0.071 |
| [Boundary pool / 1,500 cap / historical settings](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-historical-prompt.json) | 150 | 35/150 (0.233) | 0.487 | 0.708 | 0.071 | 0.074 |
| [Boundary pool / 1,500 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-training-prompt-a314.json) | 150 | 34/150 (0.227) | 0.473 | 0.708 | 0.013 | 0.103 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Boundary pool / 1,500 cap](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-v1-50.json): 2026-09-26 UTC, 50 cases.
- [Boundary pool / 1,500 cap / historical settings](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-historical-prompt.json): 2026-09-26 UTC, 150 cases.
- [Boundary pool / 1,500 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1500-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.

## Interpretation

Under the matched training prompt, success was 34/150 versus 37/150 for the 1,000-row candidate. More rows did not yield a broad improvement in this run.

## Limits

Changing the cap also changes optimizer exposure and class proportions. This is not an isolated test of semantic diversity alone. Single runs do not establish that a larger dataset is generally harmful.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
