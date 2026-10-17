# Generator 15: Urgency Without an Invented Deadline

**Status:** Generator correction implemented; project data not regenerated.
**Decision date:** 2026-10-14.

## Why

The [ten-target training audit](06-capped-training-clarification-target-audit.md)
found a bare-urgency task labeled CLARIFY despite the tool permitting an undated
task. The owner chose ACT without a due date. This is a policy correction from
training-data review, not a benchmark-driven rewrite or demonstrated model win.

## Exact Change

- Bare urgency, for example "Create a task to revise chemistry soon," now has
  productive Notion examples with no due_date argument. The task title preserves
  "soon" instead of silently dropping the urgency or inventing a date.
- Deadline-clarification templates in seed, generated and contrast-pair examples
  now explicitly say "due soon." Their targets still request the missing date.
  Existing triplets with explicit unspecified deadlines remain CLARIFY.
- A seed ACT example and alternating generated Notion ACT variants teach undated
  urgency. Existing dated ACT variants remain represented.
- Generator version advances from 14 to 15; recipe hashes and dataset revision
  therefore distinguish the new source. The overall source-count budget remains
  the existing one; this is not an expansion of the dataset.
- The builder and CLI default to data/sft/v1/baymax-control-policy-v15. All builds
  reject a populated destination before generating or writing any artifact.

## Preserved

No system prompt, model hyperparameter, runtime formatting, scorer, adapter,
evaluation fixture, historical dataset, checkpoint or result was changed.
New examples use the existing deterministic generation and group-preserving
split logic. Their content changes can change split membership and which rows
survive later sampling; do not claim that v15 is an anchored experiment without
checking the actual derived cap.

Existing mixed datasets and training configs still refer to their previous
artifacts. Merely updating this generator does not change data used by a future
Trainer run. No project dataset has been rebuilt or mixed in this batch.

## Verification

Focused tests exercise standard and matched generation, undated targets,
preservation of explicit-deadline clarification, executable adapter calls,
strict target contracts, version-dependent hashes and no-overwrite behavior.
The existing generator tests also check contrast-group structure and split
separation. No tokenizer/model or long training/evaluation job is needed.

## Next Owner-Run Preparation

Build only the new synthetic source into a new directory:

```powershell
.venv\Scripts\python scripts\build_baymax_sft_dataset.py `
  --output-dir data\sft\v1\baymax-control-policy-v15
```

Then validate and inspect that source before considering a new mix or training
configuration. If mixing later, explicitly pass the v15 --baymax-dir: the mixer's
historical defaults are not redirected by this change. Preserve the name-first
serialization stage separately if preparing another name-first experiment.
Do not overwrite an old mixed artifact or silently repoint a frozen config.
