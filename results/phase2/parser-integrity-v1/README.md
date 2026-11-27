# Parser Integrity Re-scoring

Files now include the experiment, suite size, and `strict-parser-rescore` in
their names. See the complete [Phase 2 catalog](../README.md) and
[rename map](../FILE_RENAMES.json) for old/current paths. Embedded source paths
remain historical; resolve them through the map when locating source files.

Saved generations reparsed on 2026-09-28 with `strict-complete-response-v1`.
No model inference or training was performed. Fake tool execution was replayed.
Each JSON records its source SHA256, source run ID, scenario SHA256, and inherited
latency/cost. Historical source files were not overwritten.

| Checkpoint | Previous 150 passes | Rescored 150 passes | Invalid outputs / 150 | Shift passes / 30 | Invalid shift outputs / 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACR-1k | 43 | 42 | 17 | 6 | 3 |
| Schema-diverse | 42 | 40 | 20 | 5 | 3 |
| ACT-balanced | 47 | 46 | 18 | 6 | 2 |
| ACT-midpoint | 46 | 46 | 16 | 6 | 3 |

These results correct parsing/scoring only. They do not repair the old checkpoints'
unseeded LoRA initialization or leakage between synthetic SFT partitions.
The separate benchmark fixtures have not been shown to overlap training data.
Repeated comparisons on these suites are development evidence, not untouched
final-test estimates.

## Corrected Data

Generator version 11 joins duplicate-content groups and contrast groups before
splitting. `data/sft/v1/baymax-control-disjoint` has 702 train, 200 validation and
98 test rows, with no complete SFT example shared between partitions.

Revision: `1ff9e3934ff3987c274aa4a903f7078caf96f89bda335777020f87b5ac6cf9ca`.

The training pool has 222 unique ACT, 122 unique CLARIFY and 132 unique REFUSE
examples. The previously planned 150/150 boundary quotas cannot be filled without
new unique examples or repeats. The earlier 70/30 mixed artifact still uses the
old split and must not be used as the corrected next run.
