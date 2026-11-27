# Effective Token Distribution and Training-Cap Audit

[Experiment index](../README.md)

## Question and Context

Did nominal row weights reflect the learning exposure actually delivered to SFT?

## Method

Report assistant-label tokens after mixing, split selection, deterministic shuffling, caps, truncation and removal of rows with no usable labels. Count sources and exact unique rows as well as behavioral labels.

This is a data measurement, not a model training experiment.

## Saved 5,000-Row Token Snapshots

| Snapshot | Direct rows / tokens | Multi rows / tokens | Refusal rows / tokens | Clarification rows / tokens |
| --- | ---: | ---: | ---: | ---: |
| Original mix | 2,285 / 63,148 | 2,497 / 121,608 | 121 / 2,953 | 97 / 2,380 |
| Extra-six / extra-twelve | 2,125 / 60,266 | 2,331 / 115,158 | 356 / 8,745 | 188 / 4,658 |
| Legacy boundary weighting | 2,086 / 58,321 | 2,187 / 107,640 | 378 / 9,293 | 349 / 8,612 |

The two saved snapshots within each weighted family report the same token counts. That alone does not establish identical checkpoint identity.

## Cap Failure and Correction

At one stage the 1,000-row artifact retained 60 refusal rows but only 36 unique ones, and 40 clarification rows but only 27 unique ones. More authored examples were not necessarily surviving the cap.

Unique-first sampling corrected that. Increasing the raw cap then exposed another defect: at 1,500, refusal rose to 103 rows but only 60 unique, and clarification to 68 but only 45 unique. There were 67 exact duplicates overall.

The corrected fixed-contribution reports were:

| Raw cap | Usable rows | BAYMAX | xLAM | Unique refusal | Unique clarification | Exact duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 967 | 150 | 817 | 60 | 45 | 0 |
| 1,500 | 1,447 | 150 | 1,297 | 60 | 45 | 0 |

## Interpretation

Rows and tokens answer different questions. Multi-tool targets contribute more tokens per example; a substantial boundary row count can still be a small share of training loss.

The sampler fix preserves diversity and deterministic selection. It does not prove the model improved. The two boundary-pool training notes document those evaluations separately.

## Limits

Earlier duplicate counts and corrected cap counts come from owner reports. No fresh tokenizer or full-corpus audit was run to write these notes. Exact duplicates are not the same as near-duplicates or template similarity.

Do not assign a 5,000-row snapshot to a particular checkpoint without confirming its provenance.

## Supporting Files

- [original-mix-token-audit-5000-and-boundary-failures.json](../../../results/phase2/analysis/original-mix-token-audit-5000-and-boundary-failures.json)
- [clarification-extra6-refusal-extra12-token-audit-5000-snapshot1.json](../../../results/phase2/analysis/clarification-extra6-refusal-extra12-token-audit-5000-snapshot1.json)
- [clarification-extra6-refusal-extra12-token-audit-5000-snapshot2.json](../../../results/phase2/analysis/clarification-extra6-refusal-extra12-token-audit-5000-snapshot2.json)
- [boundary-weighting-legacy13-token-audit-5000-snapshot1.json](../../../results/phase2/analysis/boundary-weighting-legacy13-token-audit-5000-snapshot1.json)
- [boundary-weighting-legacy13-token-audit-5000-snapshot2.json](../../../results/phase2/analysis/boundary-weighting-legacy13-token-audit-5000-snapshot2.json)
- [report_sft_artifact.py](../../../scripts/report_sft_artifact.py)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
