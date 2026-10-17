# Training Labels: Counts and Decoded-Target Check

**Status:** Owner completed both count and decoded-label diagnostics.
Date: 2026-10-15.

## Frozen Artifact

Use the name-first config and its existing version-14 mixed training artifact.
Do not repoint the config at the newly generated version-15 synthetic source.
This is diagnosis of the data path used for the existing checkpoint, not new
training or a comparison between models.

## Owner-Run Count Diagnostic

Saved report: tmp/name-first-training-tokenization-audit.json.

| Measurement | Rows/tokens |
| --- | ---: |
| Raw rows checked | 1000 |
| Rows with trainable labels | 974 |
| Rows without trainable labels | 26 |
| Truncated rows | 32 |
| Prompt fills context | 26 |
| Total trainable label tokens | 36614 |

The counts match the earlier name-first artifact report. The training function
skips rows with no unmasked assistant labels, so the 26 empty-label examples
do not contribute loss. Six truncated examples still have trainable tokens;
the current training path retains partially truncated targets. No filtering rule
was changed in this investigation.

The report stores only 30 of 32 problem entries. All 30 listed entries map to
xLAM productive-call targets. Do not treat this capped list as 32 individually
inspected problems. Report indices are zero-based: index 400 is file line 401.

A separate bounded read confirmed that the first 400 raw rows are BAYMAX
(100 clarification, 100 refusal, 175 direct calls and 25 multi-call sequences).
The first 100 rows are all clarification targets. No model/tokenizer was loaded
for that mapping; it uses saved report indices and existing source references.

Counts alone do not show that a particular question is entirely supervised.

## Small Decoded-Label Check

The owner completed the 100-row clarification check. All 100 rows retain labels,
none are truncated and none fill the context with prompt alone. They contain
2,845 trainable label tokens, with 23-48 per row. All four checks are true for
the three previews below; each decoded target includes the full question,
closing JSON and the normal im_end token. This supports correct supervision for
the inspected examples, not proof that the model learned them or a check of
historical gradients.

The existing diagnostic now has optional --inspect-row previews. They call the
same _tokenize_chat_example function used by training, with unchanged settings.
Each preview includes:

- expected_target: the complete original assistant target;
- decoded_trainable_labels: actual label IDs other than -100, decoded without
  stripping special tokens or cleaning up spaces;
- prompt_prefix_matches: independently checks that the full chat's token prefix
  matches the system/user chat plus generation header;
- prompt_tokens_masked: checks the prompt portion is ignored in loss;
- visible_assistant_labels_match_inputs: checks retained assistant token labels
  equal their corresponding input token IDs;
- complete_target_visible: checks the decoded trainable text starts with the
  entire intended target, not merely the function name or a partial question.

The JSON report gains label_previews only when requested. Default diagnostics
remain count-only. No training code, masking rule, collator, checkpoint, source
example or scorer is changed. Mock-tokenizer tests cover complete, partial,
fully masked and mismatched-template-boundary cases.

## Owner Command

This checks all 100 clarification examples at the original 1,024-token limit and
previews indices 0, 9 and 15 (file lines 1, 10 and 16). It loads the tokenizer,
not model weights. Use the new output filename to preserve the original report.

```powershell
.venv\Scripts\python scripts\diagnose_sft_tokenization.py `
  --config configs\sft\qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json `
  --split train `
  --max-rows 100 `
  --max-sequence-length 1024 `
  --inspect-row 0 --inspect-row 9 --inspect-row 15 `
  --output tmp\name-first-clarification-labels-100.json
```

Inspect the count summary and each preview's complete question and closing JSON.
All four preview checks should be true for fully retained, correctly masked
targets. If not, investigate that concrete failure before changing data or
training. If true, it supports correct supervision for the inspected rows; it
does not establish robust policy learning or prove every row was correct during
the historical run. The six partially truncated productive targets are a
separate issue, not established as the cause of clarification mistakes.
