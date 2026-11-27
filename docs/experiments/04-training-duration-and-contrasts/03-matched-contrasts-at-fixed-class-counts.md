# Matched Contrasts at Fixed Class Counts

[Experiment index](../README.md)

## Question and Context

Could tighter ACT/CLARIFY/REFUSE contrasts teach the decision boundary better than isolated examples, without changing broad class quantities?

A useful contrast changes missing information, state or capability while keeping the surrounding task similar. It should teach the relevant distinction, not a keyword shortcut.

## Controlled Change

Start from the anchored two-epoch recipe. Preserve all 600 raw xLAM positions and validation/test files byte-for-byte. Replace 398 of 400 BAYMAX training rows; keep two identical.

Raw quantities remain 200 BAYMAX ACT / 100 CLARIFY / 100 REFUSE. Selection prioritized complete training-only contrast groups: 120 groups, comprising 73 triplets and 47 pairs. Held-out overlap checks remained enforced.

Usable rows stayed 974, including all 400 BAYMAX examples. No exact duplicates were present. Training settings and prompt matched the two-epoch reference.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Matched contrasts / two epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-matched-anchored-train1k-val200-epochs2-combined-150.json) | 150 | 39/150 (0.260) | 0.480 | 0.698 | 0.087 | 0.052 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Matched contrasts / two epochs (150 cases) | 0.383 | 0.053 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Decision Matrix: Matched contrasts / two epochs (150 Cases)

Rows are expected decisions; columns are recorded predictions. The diagonal is correct routing, not necessarily full task success.

| Expected | ACT | CLARIFY | REFUSE | Invalid/unknown |
| --- | ---: | ---: | ---: | ---: |
| ACT | 48 | 29 | 2 | 2 |
| CLARIFY | 20 | 14 | 6 | 0 |
| REFUSE | 16 | 2 | 5 | 6 |

### Result Provenance

- [Matched contrasts / two epochs](../../../results/phase2/qwen-lora/baymax-mix/target-compatible-generator14-act200-matched-anchored-train1k-val200-epochs2-combined-150.json): 2026-10-07 UTC, 150 cases.

## Interpretation

Task successes fell from 44 to 39/150, macro-F1 from 0.514 to 0.383, and invalid outputs rose from two to eight. Correct refusal routing fell from 16/29 to 5/29.

The separate audit found real capability/routing errors as well as missing tool names and wrong meta-tool fields. This was not merely keyword scoring rejecting semantically correct refusals.

## Important Confound

Broad class counts stayed fixed, but within-class coverage did not. Refusal examples from triplet origins increased from 33 to 73; standalone and capability-pair origins disappeared from the cap. Source-ID family count fell from 45 to 20.

Those are authoring lineages, not a complete semantic taxonomy. The plausible hypothesis is narrower coverage, not a demonstrated causal explanation.

## Decision and Limits

Retain the original anchored two-epoch reference. This matched implementation regressed; it does not disprove contrastive training in general. A future contrast experiment would need to preserve semantic/domain coverage as well as label counts.

The earlier auth-filtered matched preflight was blocked by xLAM target failures; it was a different preparation, not a completed precursor checkpoint.

## Supporting Files

- [qwen2_5_1_5b_lora_target_compatible_v14_act200_matched_anchored_val200_epochs2.json](../../../configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_matched_anchored_val200_epochs2.json)
- [matched-contrasts-epochs2-refusal-format-audit.md](../../../results/phase2/analysis/matched-contrasts-epochs2-refusal-format-audit.md)
- [matched-contrasts-epochs2-refusal-format-audit.json](../../../results/phase2/analysis/matched-contrasts-epochs2-refusal-format-audit.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
