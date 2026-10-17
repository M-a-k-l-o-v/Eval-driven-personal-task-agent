# Training Data Contract and Policy Review

[Experiment index](../README.md)

## Question and Context

Were model failures partly being taught by inconsistent training examples rather than caused only by insufficient training?

This was a series of related source-quality audits, not a claim that all external data was bad or a completed model-training experiment.

## BAYMAX Source Findings

An adapter audit checked 1,128 synthetic examples and executed all 388 productive target sequences in fresh fake state. It found:

- 188 clipboard states used a plain string instead of the adapter's state object.
- 17 calendar states lacked required event details.
- 43 clarification targets did not request all missing details.

The generator13 correction changed 248 distinct examples without turning invalid/ambiguous requests into valid ACT requests.

Owner-approved generator14 policy changes corrected 41 bulk-outreach explanations, one unavailable-send label, and 24 draft-composition triplet examples. Training targets were validated independently of benchmark outcomes.

## External Source Findings and Filtering

A numeric audit scanned 60,000 rows and flagged 1,069 candidates. Flags required review; numeric coincidence or explicit rounding could be valid.

A deliberately enriched review of 100 examples found both metadata and target/prompt defects. It cannot estimate the whole dataset's defect rate.

The broad contract scan flagged 31,821 rows. That was not the removal list: defaults, ambiguous schemas and context-dependent findings are not equivalent to confirmed incompatible targets.

Credential filtering excluded 618 rows. A narrower target-compatibility audit on the remaining source proposed 1,771 additional exclusions, leaving 57,611 accepted examples. Retained source bytes/order/splits and reviewed exclusion identities were preserved.

## Blocked Experiment

The first auth-filtered matched-data preparation failed whole-dataset validation and a capped training preflight reported 30 target issues, including duplicate tools, wrong types and unverifiable types.

No successful checkpoint/result is evidenced for that blocked preparation. The owner filtered the incompatible external targets before running subsequent cleaned experiments.

## Interpretation

Mechanical validity and policy correctness are different checks. Passing a schema does not prove an answer is grounded, necessary or appropriate. Conversely, every broad warning is not grounds for dropping a row.

These corrections made training evidence more trustworthy. The resampled and anchored cleaning notes record the actual model outcomes; improvement must come from those results, not the audit counts.

## Limits

Semantic quality remains incompletely audited, and no LLM judge was introduced. Evaluation cases were not used to rewrite source targets. The local review artifacts are not required to be committed or embedded in these notes.

## Supporting Files

- [TRAINING_DATA_AUDIT.md](../../../docs/TRAINING_DATA_AUDIT.md)
- [qwen2_5_1_5b_lora_auth_filtered_v14_matched_act200_val200.json](../../../configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_matched_act200_val200.json)
- [audit_sft_numeric_arguments.py](../../../scripts/audit_sft_numeric_arguments.py)
- [validate_sft_dataset.py](../../../scripts/validate_sft_dataset.py)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
