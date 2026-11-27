# Early Balanced BAYMAX Data at Two Epochs

[Experiment index](../README.md)

## Question and Context

Would longer training help the early 100/100/100 BAYMAX mixture learn its policy?

## Change

An early two-epoch SFT run used the same general ACR mixture family. This is not the later cleaned, anchored two-epoch reference.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [100/100/100 BAYMAX / two epochs](../../../results/phase2/qwen-lora/baymax-mix/act-clarify-refuse-train1k-epochs2-combined-150.json) | 150 | 46/150 (0.307) | 0.570 | 0.791 | 0.075 | 0.138 |
| [100/100/100 BAYMAX / two epochs](../../../results/phase2/template-shift/act-clarify-refuse-train1k-epochs2-template-shift-30.json) | 30 | 5/30 (0.167) | 0.467 | 0.728 | 0.250 | 0.400 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Result Provenance

- [100/100/100 BAYMAX / two epochs](../../../results/phase2/qwen-lora/baymax-mix/act-clarify-refuse-train1k-epochs2-combined-150.json): 2026-09-27 UTC, 150 cases.
- [100/100/100 BAYMAX / two epochs](../../../results/phase2/template-shift/act-clarify-refuse-train1k-epochs2-template-shift-30.json): 2026-09-27 UTC, 30 cases.

## Interpretation

The historical full-suite count increased from 43 to 46 successes relative to the one-epoch ACR artifact. The template-shift count fell from six to five.

That result did not establish robust generalization from the extra epoch.

## Limits

These runs predate later strict-parser and data-integrity corrections. No strict-rescore artifact for this two-epoch candidate is present in the catalog. The current two-epoch config is not sufficient evidence of an immutable historical data identity.

## Supporting Files

- [qwen2_5_1_5b_lora_baymax_mix_2epoch_v1.json](../../../configs/sft/qwen2_5_1_5b_lora_baymax_mix_2epoch_v1.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
