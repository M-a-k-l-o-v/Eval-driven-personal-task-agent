# Existing Checkpoint: Training Loss by Decision Class

**Status:** Owner-run loss measurement completed.
Date: 2026-10-15.

## Question

The frozen name-first checkpoint clarified only four of ten reviewed training
inputs, despite verified assistant supervision. Does its aggregate loss hide
weaker learning of CLARIFY or REFUSE answers?

This is a checkpoint diagnostic, not another training experiment. Keep the
checkpoint, dataset, prompt, answer formatting and 1,024-token context unchanged.

## Command

From `project-baymax`, run:

```powershell
.venv\Scripts\python scripts\report_sft_loss.py `
  --split train `
  --max-rows 1000 `
  --max-sequence-length 1024 `
  --output tmp\name-first-training-loss-by-class.json
```

The default config selects the existing two-epoch QKVO name-first model. Use
`--config` to explicitly select another config, not an inference prompt override.
Optional `--dry-run` checks config, prompt, dataset and adapter identities without
loading a tokenizer/model or measuring loss. Use a different output for a dry run.

## Measurement

Read the first 1,000 training rows in their unchanged order. Reuse the actual
training tokenization and prompt masking, including truncation. Load the existing
adapter read-only, disable gradients and dropout, and score one row at a time.
Do not generate, repair, execute tools, or call an optimizer.

The model's built-in causal cross-entropy scores each visible assistant token
against its ground-truth preceding tokens. Prompt labels are ignored (`-100`),
and position zero has no preceding prediction. Visible chat-ending tokens count.
Rows with no scoreable labels are reported as skipped; partially truncated
answers remain included, matching the training pipeline's existing behavior.

Report ACT (both direct and multi-tool), CLARIFY and REFUSE separately:

- scored, skipped and truncated row counts;
- scored assistant-token counts;
- token-weighted mean loss, in nats;
- unweighted mean of per-row losses.

The JSON also records overall loss, per-row file lines/source refs and loss,
config/dataset/adapter hashes, actual library versions, model dtype and resolved
model commit when exposed by Transformers. Existing reports cannot be overwritten;
outputs inside dataset/model directories are rejected.

## Interpretation Limits

Higher loss means the checkpoint assigns less probability to the supplied answer
tokens on average. Token weighting measures average token prediction; row weighting
prevents long answers from dominating that particular average.

Neither is routing accuracy. Common JSON syntax and easy later tokens can hide a
bad initial tool choice. Teacher forcing supplies correct preceding answer tokens,
unlike free generation. Answer lengths, vocabulary and task mix also differ across
classes. A low loss therefore does not prove reliable decisions, and a class-loss
gap alone does not establish its cause or justify more epochs.

## Owner-Run Result

Source: `tmp/name-first-training-loss-by-class.json` and the owner's terminal output.

| Class | Scored rows | Skipped | Truncated | Scored tokens | Loss/token | Mean row loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACT | 774 | 26 | 32 | 30,864 | 0.039647 | 0.035162 |
| CLARIFY | 100 | 0 | 0 | 2,845 | 0.346352 | 0.330206 |
| REFUSE | 100 | 0 | 0 | 2,905 | 0.299602 | 0.293663 |

ACT contributes approximately 84% of scored tokens. The checkpoint assigns lower
average probability to the supplied boundary-answer tokens than to ACT-answer
tokens. This does not isolate routing from question/reason composition. All
boundary rows are visible to supervision, so truncation does not explain this
particular class difference.

Next diagnostic: [separate call-name and text losses](12-tool-name-versus-boundary-text-loss.md),
without changing this checkpoint or its training data.

No benchmark examples were read and no training target was changed.
