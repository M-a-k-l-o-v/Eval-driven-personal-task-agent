# BAYMAX Experiment Notebook

This folder explains the experiments, rather than merely listing result filenames.
Each numbered note covers one training experiment, inference experiment or clearly
identified measurement/audit investigation. Multiple evaluations of the same
checkpoint can live in its note; a 30-case diagnostic does not become a separate
trained model.

Start with a question or follow the stages below. Numbers within folders are
reading order, not invented run IDs or a guarantee of exact historical chronology.
Every evaluation table links to its original JSON and records its saved date.

Latest completed runs: [unique 2k training pool at two epochs](08-output-contract-learning/19-unique-2k-training-pool.md)
and [the same 2k pool at four epochs](08-output-contract-learning/20-unique-2k-four-epochs.md).
The four-epoch run records 73/150 successes, macro-F1 0.681 and zero invalid
outputs. The owner closed the project on 2026-10-17. These are single-seed
development results with remaining clarification errors, not a production claim.
The [grounding revision](08-output-contract-learning/18-controlled-grounding-data-revision.md)
did not improve the selected unweighted compact/name-first model; it is retained
as the corrected 1k parent for the expansion, not promoted as the reference.
The historical summary below describes earlier experiments.

## Where We Stand

The selected development reference is
[cleaned anchored SFT at two epochs](04-training-duration-and-contrasts/01-anchored-data-at-two-epochs.md).
It is the reference for balanced routing, not the winner of every metric.

| Candidate | Task successes / 150 | Routing macro-F1 | Invalid outputs / 150 | Premature ACT on boundary cases |
| --- | ---: | ---: | ---: | ---: |
| Anchored two-epoch SFT | 44 | 0.514 | 2 | 31 |
| DPO from that reference | 53 | 0.376 | 5 | 56 |
| 2:1 SFT | 29 | 0.477 | 8 | 22 |
| DPO from 2:1 SFT | 31 | 0.453 | 7 | 34 |
| DPO with completed ACT rejections | 46 | 0.482 | 7 | 45 |
| Rank-16 two-epoch SFT | 46 | 0.450 | 7 | 42 |
| Lower-learning-rate two-epoch SFT | 45 | 0.436 | 15 | 37 |
| QKVO two-epoch SFT | 52 | 0.437 | 14 | 40 |

Those are measured development results, not evidence of a robust overall win.
First DPO has the highest task completion but substantially worse abstention.
Completing its missing rejections recovers some balance, yet remains below original
SFT macro-F1. Rank 16 also worsened routing and validity despite two additional
task successes. Lower learning rate did not improve balance either. QKVO completes
more tasks but has worse routing and output validity: retain it as a task-completion
candidate, not the balanced reference. A's original rank-8 recipe remains the
reference. No model switch or new training method was implemented by this
documentation task.

The full recipes, matrices, sources and caveats are in the individual notes.
For the most recent sequence, read the two-epoch reference, four-epoch trial,
matched contrasts, 2:1 trial, and the three DPO notes in that order.

## How an Experiment Flows

1. Author or import training examples under the general policy and tool contracts.
2. Protect source train/validation/test membership and contrast/content groups.
3. Mix and sample a deterministic raw training cap.
4. Render the chat template, tokenize, mask non-assistant labels and truncate.
5. Inspect the usable artifact that actually reaches the trainer.
6. Train a new versioned adapter.
7. Run the chosen adapter on a separate evaluation suite through fake adapters.
8. Read task outcome, routing, output validity and wording metrics separately.

Pipeline fixes at steps 2-5 or 7-8 can change the validity of the evidence without
creating a better trained model. That distinction explains several apparent jumps
in the history.

## Vocabulary

| Term | Meaning in these notes |
| --- | --- |
| ACT | Select supported productive tools when the request is safe and complete. |
| CLARIFY | Supported safe action needs missing/ambiguous information or required confirmation. |
| REFUSE | Action/capability is unsafe, unavailable, unsupported or out of scope. |
| Meta-tool | request_clarification(question) or refuse_request(reason); not an external mutation. |
| xLAM | External Salesforce function-calling dataset hosted on Hugging Face. |
| BAYMAX-control | Authored training data; "control" is the dataset name, not necessarily a baseline model. |
| Mixed training | xLAM plus authored BAYMAX examples. |
| SFT | Learn supplied correct assistant answers using target-token cross-entropy. |
| LoRA | Train small adapter weights rather than updating every base-model weight. |
| DPO | Learn to prefer a chosen answer over a rejected answer for the same prompt. |
| Anchored sampling | Keep reference examples/positions and replace only explicitly eligible rows. |
| Matched contrast | Closely related requests/states/tools with a genuine decision-changing difference. |
| Raw cap | Number selected before tokenization and assistant-label filtering. |
| Usable rows | Examples retaining trainable assistant labels after preprocessing. |
| Epoch | One pass over the usable training artifact; not one optimizer step. |
| Exact duplicate | Identical example content; not a test of semantic or template similarity. |
| Reference freeze | Record hashes to detect drift; it does not lock or copy files. |

### Metrics Are Different Questions

- **Task success:** did all scenario checks pass, including arguments and message criteria?
- **Routing macro-F1:** average ACT, CLARIFY and REFUSE F1 equally. F1 combines
  precision (correctness of predictions of that class) and recall (coverage of
  expected cases of that class). Invalid/unknown outputs miss their expected class.
- **Decision accuracy:** correctness among recorded known decisions. Read it with
  coverage; unknown outputs are not included in its conditional denominator.
- **Clarification/refusal accuracy:** required question/reason keyword matches.
  A correct label can still have an incomplete or misleading explanation.
- **Invalid-output rate:** strict parser rejections, separate from valid but
  inappropriate decisions or tool-execution failures.
- **Hallucination rate:** currently unavailable tool names, not every invented
  person, date or success claim. Invalid parsing can leave no accepted calls to count.
- **Latency/cost:** local reported API cost zero is not zero electricity,
  hardware or training cost. Replayed results inherit original inference timing.

Tables use ratios from zero to one, rounded to three decimals, unless they show
explicit counts. `n/a` means unavailable or inapplicable, not zero.

## Which Evaluation Suite?

| Suite | Size | Purpose |
| --- | ---: | --- |
| v1 | 50 | Original benchmark, used by early trials. |
| Combined v1 + v2 | 150 | Established development comparison. Current decision support is ACT 81, CLARIFY 40, REFUSE 29. |
| Template shift | 30 | Separate wording/structure diagnostic: ten cases per decision. |
| Refusal subset | 7 | Small single-class diagnostic, not balanced overall evaluation. |
| Smoke subsets | 1-4 | Sanity checks; not estimates of full benchmark quality. |

Historical policy-era fixtures/scoring differ from current measurements.
Do not combine the template-shift and established-suite rates into one score.
Their repeatedly inspected status means neither is an untouched final benchmark.

## Early Training

The first external-only, synthetic-only and mixed trials. Historical provenance is incomplete.

- [External Tool Data Only: First Local Fine-Tune](01-early-training/01-external-tool-data-only.md)
- [BAYMAX Control Data Only](01-early-training/02-baymax-control-data-only.md)
- [Control Data With Contrast Pairs](01-early-training/03-control-data-with-contrast-pairs.md)
- [Control Data With Targeted Refusals](01-early-training/04-control-data-with-targeted-refusals.md)
- [Original External and BAYMAX Training Mix](01-early-training/05-original-external-and-baymax-mix.md)
- [Expanded Control Generator: Early Mixed Trial](01-early-training/06-expanded-control-generator.md)
- [Weighted Control Examples: Early Oversampling Trial](01-early-training/07-weighted-control-examples.md)
- [Clarification and Refusal as Explicit Meta-Tools](01-early-training/08-meta-tools-in-training.md)

## Format, Prompt and Sampling

Input alignment, weighting, unique-first caps and inference-only prompt changes.

- [Aligning Training and Evaluation Formats](02-format-and-sampling/01-training-and-evaluation-format-alignment.md)
- [Extra Clarification and Refusal Repeats](02-format-and-sampling/02-extra-clarification-and-refusal-repeats.md)
- [Legacy Boundary-Weighting Trial](02-format-and-sampling/03-legacy-boundary-weighting-trial.md)
- [Fixed Unique Boundary Pool at 1,000 Rows](02-format-and-sampling/04-fixed-unique-boundary-pool-at-1000-rows.md)
- [Increasing the Cap to 1,500 Without Repeating Boundaries](02-format-and-sampling/05-larger-cap-with-the-same-boundary-pool.md)
- [First Policy Prompt: Evaluation Without Retraining](02-format-and-sampling/06-first-policy-prompt-without-retraining.md)
- [Expanded Policy Prompt: Evaluation Without Retraining](02-format-and-sampling/07-expanded-policy-prompt-without-retraining.md)
- [Returning to the Recorded Training Prompt](02-format-and-sampling/08-evaluating-with-the-recorded-training-prompt.md)

## Data Quality and Class Mix

Changes in class quantities, partition integrity, policy, credentials and target compatibility.

- [One Hundred ACT, CLARIFY and REFUSE Examples](03-data-quality-and-class-mix/01-one-hundred-examples-per-decision.md)
- [Early Balanced BAYMAX Data at Two Epochs](03-data-quality-and-class-mix/02-early-balanced-data-at-two-epochs.md)
- [Schema-Diverse Control Examples](03-data-quality-and-class-mix/03-schema-diverse-control-examples.md)
- [Increasing Productive BAYMAX Examples to Two Hundred](03-data-quality-and-class-mix/04-more-productive-baymax-examples.md)
- [Productive Midpoint: One Hundred Fifty BAYMAX ACT Examples](03-data-quality-and-class-mix/05-productive-midpoint-at-one-hundred-fifty.md)
- [Content-Group Splits With Seventy Percent ACT](03-data-quality-and-class-mix/06-content-group-splits-and-more-boundary-data.md)
- [Corrected Policy, Credential Filtering and Representative Validation](03-data-quality-and-class-mix/07-corrected-policy-and-authentication-filter.md)
- [Corrected Policy With Two Hundred Productive Examples](03-data-quality-and-class-mix/08-corrected-policy-with-two-hundred-act-examples.md)
- [Target-Contract Cleaning With Resampled External Rows](03-data-quality-and-class-mix/09-contract-cleaning-with-resampled-external-rows.md)
- [Target-Contract Cleaning With Anchored External Rows](03-data-quality-and-class-mix/10-contract-cleaning-with-anchored-external-rows.md)

## Training Duration and Matched Contrasts

Fresh one/two/four-epoch comparisons, matched groups and the 2:1 SFT ratio.

- [Anchored Data at Two Epochs: Selected Reference](04-training-duration-and-contrasts/01-anchored-data-at-two-epochs.md)
- [Anchored Data at Four Epochs](04-training-duration-and-contrasts/02-anchored-data-at-four-epochs.md)
- [Matched Contrasts at Fixed Class Counts](04-training-duration-and-contrasts/03-matched-contrasts-at-fixed-class-counts.md)
- [Two-to-One Productive-to-Boundary SFT Ratio](04-training-duration-and-contrasts/04-two-to-one-productive-to-boundary-ratio.md)

## SFT Hyperparameters

Controlled adapter/training changes from the selected SFT reference. B, C and E
are completed. Increasing rank, lowering the learning rate and adding attention
projections did not beat A's balanced routing. E improved task completion, with
output-contract failures documented separately.

- [Experiment B: Rank 16 at the Fixed Two-Epoch SFT Recipe](07-sft-hyperparameters/01-rank16-at-fixed-two-epoch-recipe.md)
- [Experiment C: Lower Learning Rate at Two Epochs](07-sft-hyperparameters/02-lower-learning-rate-at-two-epochs.md)
- [Experiment E: LoRA on All Four Attention Projections](07-sft-hyperparameters/03-all-four-attention-projections.md)

## Output-Contract Learning

Separate structural expression from policy judgment. Preparation is implemented;
the next run changes only assistant call-key order using E's unchanged recipe.

- [Name-First Tool Calls: Serialization-Only SFT on E](08-output-contract-learning/01-name-first-targets-on-qkvo.md)
- [Held-Out SFT Clarification: Ten-Example Sanity Check](08-output-contract-learning/02-heldout-clarification-sanity-check.md)
- [SFT Versus Benchmark: Prompt Audit and JSON-Presentation Check](08-output-contract-learning/03-prompt-audit-and-json-presentation-check.md)
- [Compact JSON: Manual Review of Ten Clarification Questions](08-output-contract-learning/04-compact-json-clarification-question-review.md)
- [Held-Out SFT: Question Quality Versus Supplied Targets](08-output-contract-learning/05-heldout-sft-question-quality-review.md)
- [Capped Training Data: Ten Clarification Targets](08-output-contract-learning/06-capped-training-clarification-target-audit.md)
- [Generator 15: Urgency Without an Invented Deadline](08-output-contract-learning/07-generator15-optional-task-deadline-fix.md)
- [Capped Training Questions: Second Ten-Target Batch](08-output-contract-learning/08-capped-training-question-audit-second-batch.md)
- [Training Labels: Counts and Decoded-Target Check](08-output-contract-learning/09-training-label-supervision-check.md)
- [In-Sample Clarification: Frozen Training-Input Probe](08-output-contract-learning/10-insample-training-clarification-probe.md)
- [Existing Checkpoint: Training Loss by Decision Class](08-output-contract-learning/11-training-loss-by-decision-class.md)
- [Tool Names Versus Clarification and Refusal Text](08-output-contract-learning/12-tool-name-versus-boundary-text-loss.md)
- [First Tool-Name Loss Weight: Three Instead of One](08-output-contract-learning/13-first-tool-name-weight3.md)
- [First Tool-Name Weighting: Paired Lost and Gained Successes](08-output-contract-learning/14-tool-weight3-paired-failure-review.md)
- [First Tool-Name Loss Weight: Two Instead of Three](08-output-contract-learning/15-first-tool-name-weight2.md)
- [Held-Out SFT: Plan Completeness and Explanation Grounding](08-output-contract-learning/16-heldout-sft-plan-and-explanation-check.md)
- [Training Targets: Missing Fields and State Grounding](08-output-contract-learning/17-training-target-coverage-and-state-grounding.md)
- [Controlled Training-Data Revision: Policy and Grounding Variation](08-output-contract-learning/18-controlled-grounding-data-revision.md)

## Preference Training

The original DPO stage, alternative SFT initialization, and completed ACT rejections.

- [DPO From the Selected Two-Epoch SFT Reference](05-preference-training/01-dpo-from-the-selected-sft-reference.md)
- [DPO From the Two-to-One SFT Model](05-preference-training/02-dpo-from-the-two-to-one-sft-model.md)
- [Completing Missing Productive Rejections for Boundary Decisions](05-preference-training/03-completing-productive-rejections.md)

## Measurement and Data Audits

Baseline interpretation and engineering corrections. These are not new model-training wins.

- [Untrained Qwen: What the Baseline Actually Measures](06-measurement-and-data-audits/01-untrained-qwen-reference.md)
- [Effective Token Distribution and Training-Cap Audit](06-measurement-and-data-audits/02-effective-token-distribution-and-cap-audit.md)
- [Strict Parser Replay: Correcting Measurement, Not Training](06-measurement-and-data-audits/03-strict-parser-replay.md)
- [Training Data Contract and Policy Review](06-measurement-and-data-audits/04-training-data-contract-and-policy-review.md)
- [Decision Metrics Separate From Explanation Wording](06-measurement-and-data-audits/05-decision-metrics-separate-from-wording.md)
- [Clarification Output Contract: Target Audit and Failure Diagnostics](06-measurement-and-data-audits/06-clarification-output-contract-audit.md)

## Rules for Reading Comparisons

1. Compare the same scenario suite, prompt, generation settings, parser and scorer.
2. Distinguish a new trained checkpoint from an inference-only prompt test or
   strict replay of saved generations.
3. Check actual usable rows and tokens. A 1,000-row raw cap can deliver 974 rows.
4. A 200/100/100 BAYMAX quota plus 600 xLAM ACT examples is 800/100/100 overall,
   not a balanced three-class mixture. Multi-tool sequences still count as ACT.
5. Matching broad quantities does not guarantee matching domain/subtype coverage.
6. A fresh four-epoch run is not resuming a saved two-epoch adapter. A changed
   cosine horizon also changes per-step learning rates.
7. A 705-pair and an 800-pair DPO run at one epoch have different training exposure.
8. Single-seed changes are observations, not statistical proof or established causes.
9. Do not turn an inspected evaluation failure into a fixture-specific training target.
10. Do not assign today's config defaults to an old checkpoint with missing provenance.

## Reproducibility and Sources

The external source revision used by the pinned pipeline is
`26d14ebfe18b1f7b524bd39b404b50af5dc97866`.
The later matched training prompt is identified by
`a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468`.

Configs can declare context 2,048/batch two/accumulation eight while owner low-VRAM
commands actually use 1,024/one/16. Later notes state actual reported overrides.
Earlier notes explicitly retain missing information instead of filling it by guess.

Original evidence remains in [the result folder](../../results/phase2/README.md).
That older catalog predates some recent runs; use these notes for their explanations.
[FILE_RENAMES.json](../../results/phase2/FILE_RENAMES.json) resolves historical names
without changing result bytes. Config links identify recipes; generated model/data
links are local-only and may not exist on another checkout.

Methodology remains defined by [BENCHMARKING.md](../BENCHMARKING.md) and
[SCORING.md](../SCORING.md). Detailed investigations remain in
[TRAINING_DATA_AUDIT.md](../TRAINING_DATA_AUDIT.md) and
[DPO_EXPERIMENT.md](../DPO_EXPERIMENT.md). Some older status paragraphs describe
preparations that later completed; measured result artifacts take priority.

This notebook covers all 65 saved Phase 2 evaluation JSON files present on
2026-10-10, plus early external-only local results and supporting audits.
No raw result, training example, evaluation target or model weight was changed.
No long training/evaluation/audit command was run. These notes are not staged,
committed or pushed; the owner's restriction on uploading Phase 2 documents remains.
