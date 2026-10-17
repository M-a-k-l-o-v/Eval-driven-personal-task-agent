# DPO Experiment

## Scope

The owner approved testing a small preference-training stage after the original
anchored two-epoch SFT checkpoint. The 2:1 SFT candidate is not the starting point:
on combined-150 it improved boundary recall but reduced task success from 44 to 29.
Keep the model, inference prompt, strict parser and evaluation targets unchanged.
The repeatedly used suites remain development benchmarks, not untouched test sets.

## Preference Data

Run `scripts/build_dpo_dataset.py` to build data from the reference training cap's
BAYMAX rows. xLAM rows are not used as preference anchors in this first small stage.
The reference assistant answer is preserved as `chosen`; `prompt` contains the
original system/user messages. Each `rejected` answer addresses that same prompt.

Wrong-decision alternatives come from training-only counterparts in the existing
authored contrast groups. If a meta-tool counterpart is absent, an explicit generic
wrong-meta-tool answer is used and recorded without a counterpart source ID. If an
ACT counterpart is absent, the builder reports that missing direction rather than
inventing a productive call. Coverage therefore must be reviewed before training.

Chosen targets must pass the existing output-contract validator. Rejected answers
must also pass, except that a correctly formed productive call to an unavailable
tool is an intentional capability negative and is separately annotated. Malformed
JSON and wrong `question`/`reason` fields are not manufactured as routing negatives.
These are rule-derived preferences, not LLM-judge labels or online model generations.
Their semantic correctness still needs human sample review.

Only source training rows can provide anchors/counterparts. Reference validation
and test inputs are read solely to reject overlap; no evaluation fixtures/results
are read by the builder. Generator/source drift, contradictory answers to an
identical prompt, and source/provenance mismatch fail explicitly. Exact duplicate
prompts with identical chosen answers do not produce repeated pairs.

## Artifacts

- `train.jsonl`: conversational `prompt`, `chosen`, `rejected` for a DPO trainer.
- `pairs-review.jsonl`: same pairs plus source IDs, decision directions and negative kinds.
- `manifest.json`: source/reference hashes, seed, pair counts, unique anchor counts,
  missing productive counterparts and preference directions.

Two different rejections for one prompt are different preferences, not duplicate
rows. Inspect class exposure by both pair count and unique-anchor count. Freeze
the output after review; existing nonempty output directories are not overwritten.

## Training Status

Preference preparation and `scripts/train_dpo_lora.py` are implemented. The owner
approved TRL; the optional `dpo` extra pins TRL 1.14.0, PEFT 0.20.0, Transformers
5.15.0, Accelerate 1.14.0 and PyTorch 2.13.0. The installed CUDA PyTorch build is
retained by the pip installation below. Do not use `uv sync --all-extras` to replace
the laptop's existing training environment.

The first smoke reached reference scoring but failed on TRL 1.15's mandatory
Triton fused LM head (`_ChunkedLogProbFunction` was unavailable on native Windows).
The runner now pins 1.14 and explicitly sets `use_liger_kernel=False`, retaining
the ordinary PyTorch full-logits DPO path. No third-party source is patched and no
preference labels, reference checkpoint or DPO objective are changed. Full logits
can consume more GPU memory than fused scoring; laptop memory fit still needs the
user-run smoke test. Run metadata/hash includes the TRL version and loss backend.
Version 1.15 is rejected before model loading with a reinstall instruction.

The initial policy loads the original anchored two-epoch SFT adapter as trainable.
TRL copies its weights into a `ref` adapter, not bare Qwen. The runner verifies
weight equality and freezes the reference before optimization. Reference log
probabilities are precomputed with batch size one, avoiding a second full base
model in memory. Only the updated `default` policy adapter is saved, so the current
`run-agent-lora` evaluation path can load it normally.

Initial experimental settings: one DPO epoch, sigmoid loss, beta 0.1, learning rate
5e-6, constant scheduler, batch one, gradient accumulation 16, context 1024, seed 1.
These are conservative starting settings, not measured optimal values. Existing
LoRA rank/targets come from the saved adapter. No DPO validation preferences are
constructed; evaluate routing/task success with the existing development suites.

The config pins preference manifest/review and SFT adapter hashes. All chosen
contracts and annotated rejected contracts are checked, including intentional
unavailable-tool negatives. Context or completion truncation and chat-template
prefix mismatches fail explicitly. The artifact report records the selected pair
directions, unique prompts, completion token counts and zero truncated pairs.
Token IDs are read explicitly from `apply_chat_template(..., return_dict=True)`;
Transformers 5.15 returns a dictionary, not a bare token list. Prompt-only rendering
adds the assistant header; completed conversations explicitly do not add another.
`dpo-run.json` records config/run hashes, library versions and completion status.
Existing nonempty outputs and paths overlapping frozen inputs are rejected.

The first Windows artifact's manifest had an LF checksum while its data used CRLF.
Only its checksum metadata was corrected; preference rows/review labels were not
changed. Future builds explicitly write LF so the byte checksum is correct.

## Local Commands

From the repository root, install the optional extra in the existing CUDA venv:

```powershell
.venv\Scripts\python -m pip install -e ".[dpo]"
.venv\Scripts\python scripts\train_dpo_lora.py --dry-run
```

Review `data/dpo/v1/anchored-epochs2-preferences/pairs-review.jsonl` before
acknowledging `--reviewed`. Then run a separate memory/tokenization smoke test:

```powershell
.venv\Scripts\python scripts\train_dpo_lora.py `
  --reviewed --max-pairs 16 --max-steps 1 `
  --output-dir tmp\qwen-dpo-smoke-windows
```

After the smoke succeeds, a full run uses a new default output directory:

```powershell
.venv\Scripts\python scripts\train_dpo_lora.py --reviewed
```

Generated preference data and model outputs are ignored by Git. No packages were
installed, models loaded, or DPO training/evaluation started by the coding agent.
The user completed the Windows smoke and full original-checkpoint DPO run. The
smoke retained all 16 pairs; the full run retained all 705 pairs with no truncation.
The 2:1-starting-checkpoint experiment below is dry-run validated but not yet trained.

## 2:1 Starting Checkpoint Comparison

The owner approved repeating DPO from the two-epoch 2:1 SFT checkpoint. Use
`configs/dpo/qwen2_5_1_5b_2to1_epochs2_v1.json`. The exact 705 preference pairs,
prompt, learning rate, beta, epoch count, seed and generation/evaluation settings
stay unchanged. Only the SFT policy and its matching frozen reference change.
"2:1" refers to the SFT mixture, not a newly rebalanced preference dataset.

`reference_config` remains the preference dataset's original provenance.
`initial_sft_config` separately identifies the DPO policy/reference starting point,
pinned by file hash and adapter hashes. The alternate config must have a valid
content hash, the same base model/prompt, and identical held-out validation/test
file hashes. The original preference manifest and earlier checkpoints/results are
not rewritten. Regression tests check alternate checkpoint loading and drift.

```powershell
.venv\Scripts\python scripts\train_dpo_lora.py `
  --config configs\dpo\qwen2_5_1_5b_2to1_epochs2_v1.json `
  --reviewed
```

Output: `models/qwen2_5_1_5b_2to1_epochs2_dpo_v1/adapter`. Compare its results against
the 2:1 SFT checkpoint first, then against the original SFT/DPO pair. This is an
initialization/reference comparison, not proof that a 2:1 ratio is optimal.

## Completing Missing ACT Rejections

The first preference set lacks ACT rejections for 50 clarification and 45 refusal
anchors. The owner approved filling these before another optimization experiment.
`dpo_act_rejections.py` authors explicit wrong-action completions for existing
training families only. Unknown families remain uncovered rather than receiving
arbitrary tool calls. Existing training-only ACT counterparts take priority.

Negatives deliberately guess unresolved arguments, act before confirmation,
attempt a known but unavailable operation, or substitute an available mutation
for an unsupported request. Substituting a reminder for a booking/payment or
creating a draft instead of searching is wrong for those prompts, even if the
adapter could execute the call. No new capabilities or SFT examples are introduced.
Saved passwords/private contents are never supplied; invented access claims are
rejected answers, not chosen targets. Each added pair includes `rejection_reason`
in the review file, excluded from the model's training input/output columns.

All authored negatives use actual productive tool schemas, including unavailable
tools. Malformed arguments are not exempted merely because a tool is unavailable.
Chosen answers, prompts, original pairs and split protection remain unchanged.
No evaluation fixtures/results were read to author these negatives. Prior data,
checkpoints and result artifacts are preserved; weighting is not changed here.

```powershell
.venv\Scripts\python scripts\build_dpo_dataset.py `
  --complete-act-rejections `
  --output-dir data\dpo\v1\anchored-epochs2-act-rejections-v2
```

A short coverage check against the existing review source IDs found authored
negatives for all 95 gaps. Expected new dataset: 800 pairs, 400 unique prompts,
CLARIFY>ACT=100 and REFUSE>ACT=100 (both were partial). ACT>CLARIFY=200 and
ACT>REFUSE=200 remain unchanged: filling coverage is not full direction balancing.
Build/report the new artifact and review its rejection explanations before
creating a hash-pinned training config. No rebuild or retraining was run by the
coding agent for this change.

The user built the v2 artifact: 800 pairs, 400 unique prompts, 95 authored ACT
rejections, and no uncovered directions. The next run starts fresh from the
original anchored two-epoch SFT adapter (not the earlier DPO adapter or 2:1 SFT).
`configs/dpo/qwen2_5_1_5b_anchored_epochs2_act_rejections_v2.json` pins the new
manifest/review hashes and unchanged original SFT adapter hashes. DPO settings
remain one epoch, 5e-6 learning rate, beta 0.1 and seed 1. Results/checkpoints use
new paths. Adding 95 pairs also increases training exposure at the same epoch
count; this is a coverage/data-addition experiment, not a step-matched ablation.

After reviewing the 95 added `rejection_reason` entries:

```powershell
.venv\Scripts\python scripts\train_dpo_lora.py `
  --config configs\dpo\qwen2_5_1_5b_anchored_epochs2_act_rejections_v2.json `
  --reviewed
```

Output: `models/qwen2_5_1_5b_anchored_epochs2_dpo_act_rejections_v2/adapter`.

## Two-Epoch Duration Test

Prepared on 2026-10-12; training and evaluation remain owner-run steps.
Use `configs/dpo/qwen2_5_1_5b_anchored_epochs2_act_rejections_v2_dpo_epochs2.json`.
Only the DPO epoch count and output identity differ from the completed-rejections
one-epoch config. Keep all 800 pairs / 400 unique prompts, input hashes, original
two-epoch SFT policy/reference adapter, prompt, learning rate 5e-6, beta 0.1,
batch one, accumulation 16, context 1024, constant scheduler and seed 1 unchanged.
Do not rebuild preference data, add weighting, or resume a DPO adapter.

The planned run presents 1600 pairs over two passes and performs 100 optimizer
updates, versus 800 presentations / 50 updates for one epoch on one GPU. Unique
prompts remain 400. These are calculated exposures, not completed-run measurements
or evidence that two epochs is sufficient.

New adapter: `models/qwen2_5_1_5b_anchored_epochs2_dpo_act_rejections_v2_epochs2/adapter`.
New result: `results/phase2/qwen-lora/dpo/anchored-sft2-dpo2-act-rejections-v2-combined-150.json`.
Compare first against `anchored-sft2-dpo1-act-rejections-v2-combined-150.json`, with
the original anchored two-epoch SFT result as an additional reference. Use the
same combined 150 scenarios, greedy decoding and 256-token generation limit.
Inspect routing macro-F1, the decision matrix, premature actions, task success and
invalid outputs. No benchmark improvement is claimed until those runs complete.
