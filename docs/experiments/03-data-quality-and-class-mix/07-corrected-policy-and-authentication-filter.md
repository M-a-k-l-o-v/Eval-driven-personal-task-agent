# Corrected Policy, Credential Filtering and Representative Validation

[Experiment index](../README.md)

## Question and Context

The synthetic source had adapter-state defects and unresolved policy contradictions. External examples sometimes required model-emitted credentials. The small validation cap barely represented boundary decisions.

## Change

Use corrected generator14 BAYMAX examples and credential-filtered xLAM. Resolved policy includes refusing indiscriminate outreach, not substituting drafts for unsupported sends, and allowing drafts composed from supplied facts.

External credential filtering excluded 618 rows from the original 60,000. Source data was preserved in a separate artifact.

Keep the boundary-heavy raw training recipe: 150 BAYMAX ACT / 150 CLARIFY / 150 REFUSE plus 550 xLAM ACT. A representative validation prefix selects 30 examples per BAYMAX decision plus 110 xLAM, from validation only.

## Actual Artifact

Reported train: 976 usable rows, 374 direct, 302 multi-tool, 150 refusal and 150 clarification; BAYMAX 450 and xLAM 526; no exact duplicates.

Reported representative validation: 194 usable rows, 77 direct, 57 multi-tool, 30 refusal and 30 clarification; BAYMAX 90 and xLAM 104.

## Results

| Evaluation | Cases | Task success | Tool accuracy | Argument accuracy | Clarification wording | Refusal wording |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Corrected 70/30](../../../results/phase2/qwen-lora/baymax-mix/auth-filtered-generator14-train1k-val200-combined-150.json) | 150 | 25/150 (0.167) | 0.433 | 0.582 | 0.125 | 0.190 |

Values are ratios from 0 to 1, rounded to three decimals. Clarification/refusal columns measure wording criteria, not routing alone. Different suite sizes remain separate evaluations.

### Routing and Output Validity

| Evaluation | Macro-F1 | Invalid-output rate |
| --- | ---: | ---: |
| Corrected 70/30 (150 cases) | 0.372 | 0.113 |

Routing values use the existing comparison helper on recorded per-scenario decisions; missing historical aggregate fields are derived without rewriting source files. Invalid output and unknown decision are not identical categories.

### Result Provenance

- [Corrected 70/30](../../../results/phase2/qwen-lora/baymax-mix/auth-filtered-generator14-train1k-val200-combined-150.json): 2026-10-05 UTC, 150 cases.

## Interpretation

Success was 25/150. Relative to generator12's 26/150, there was no clear overall improvement. Necessary correctness and security fixes should not be described as an optimization win.

Representative validation improved what loss could monitor. The trainer did not use early stopping or best-checkpoint selection, so changing this validation slice alone did not directly optimize the final weights.

## Limits

Several source and validation changes occurred together. Validation losses from the sparse and representative prefixes are not directly comparable. Credential filtering did not remove all target-contract problems.

## Supporting Files

- [qwen2_5_1_5b_lora_auth_filtered_v14_val200.json](../../../configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_val200.json)
- [TRAINING_DATA_AUDIT.md](../../../docs/TRAINING_DATA_AUDIT.md)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
