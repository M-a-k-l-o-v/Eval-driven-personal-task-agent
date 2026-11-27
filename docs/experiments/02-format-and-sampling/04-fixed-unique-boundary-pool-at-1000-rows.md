# Fixed Unique Boundary Pool at 1,000 Rows

[Experiment index](../README.md)

## Question and Context

Global shuffling and truncation could retain repeated boundary examples while discarding newly authored unique examples. Could deliberate unique-first sampling produce a better training artifact?

## Change

Preserve a fixed BAYMAX contribution and fill the rest of the cap with unique xLAM rows. The observed 1,000-row raw cap contained 150 BAYMAX examples, including 60 unique refusals and 45 unique clarifications.

The owner's post-tokenization report retained 967 rows: 411 direct calls, 451 multi-tool sequences, 60 refusals and 45 clarifications. BAYMAX remained 150; xLAM retained 817. Exact duplicates were zero.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Boundary pool / 1,000 cap](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-v1-50.json) | 50 | 12/50 (0.240) | 0.560 | 0.746 | 0.167 | 0.143 |
| [Boundary pool / 1,000 cap / historical settings](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-historical-prompt.json) | 150 | 37/150 (0.247) | 0.513 | 0.738 | 0.083 | 0.056 |
| [Boundary pool / 1,000 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-training-prompt-a314.json) | 150 | 37/150 (0.247) | 0.493 | 0.738 | 0.050 | 0.086 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [Boundary pool / 1,000 cap](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-v1-50.json): 2026-09-16 UTC, 50 cases.
- [Boundary pool / 1,000 cap / historical settings](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-historical-prompt.json): 2026-09-26 UTC, 150 cases.
- [Boundary pool / 1,000 cap / matched training prompt](../../../results/phase2/qwen-lora/baymax-mix/unique-boundaries-train1k-combined-150-training-prompt-a314.json): 2026-09-26 UTC, 150 cases.

## Interpretation

The 150-case training-prompt evaluation completed 37/150 tasks. This sampler solved the exact-duplicate problem in the reported artifact; that is separate from proving strong policy transfer.

## Limits

Earlier cap audits had different counts. The numbers above describe the corrected report, not every artifact historically called boundary-diverse.

The 50-case, historical-prompt and training-prompt evaluations are different measurement settings. Older split/initialization defects still limit causal claims.

## Supporting Files

The result links above are the surviving primary evidence. A separate immutable training record was not established for this historical experiment.

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
