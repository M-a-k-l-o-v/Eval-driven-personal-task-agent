# Content-Group Splits With Seventy Percent ACT

[Experiment index](../README.md)

## Question and Context

Earlier training partitions could share duplicated content or related contrast examples. New content-group splitting protected train/validation/test separation.

The owner also tried a more boundary-heavy mixture. Therefore this was not only a split-fix experiment.

## Setup

- Generator12 synthetic data and content-grouped external data.
- Raw training cap 1,000: 150 BAYMAX ACT, 150 CLARIFY, 150 REFUSE and 550 xLAM ACT.
- Overall raw labels: 700 ACT / 150 CLARIFY / 150 REFUSE.
- Reported usable train artifact: 983 rows; direct 366, multi 317, refusal 150, clarification 150; BAYMAX 450, xLAM 533; zero exact duplicates.
- The 200-row validation prefix had almost no boundary data: one refusal and no clarification survived.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Content-group 70/30](../../../results/phase2/qwen-lora/baymax-mix/content-group-splits-generator12-train1k-70act-30boundary-combined-150.json) | 150 | 26/150 (0.173) | 0.427 | 0.579 | 0.113 | 0.207 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Content-group 70/30 (150 cases) | 0.346 | 0.147 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [Content-group 70/30](../../../results/phase2/qwen-lora/baymax-mix/content-group-splits-generator12-train1k-70act-30boundary-combined-150.json): 2026-10-05 UTC, 150 cases.

## Interpretation

The run completed 26/150 tasks. It improved some boundary wording metrics relative to ACT-heavy history, but productive performance was much worse.

Fixing leakage is essential for trustworthy methodology; it does not guarantee higher benchmark scores. This run cannot isolate whether splitting, composition or other data revisions explain the regression.

## Limits

Validation loss on this mostly-ACT prefix was a poor monitor of boundary behavior. This generator predates subsequent adapter and policy corrections.

## Supporting Files

- [qwen2_5_1_5b_lora_content_groups_70_30_v12.json](../../../configs/sft/qwen2_5_1_5b_lora_content_groups_70_30_v12.json)
- [TRAINING_DATA_AUDIT.md](../../../docs/TRAINING_DATA_AUDIT.md)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
