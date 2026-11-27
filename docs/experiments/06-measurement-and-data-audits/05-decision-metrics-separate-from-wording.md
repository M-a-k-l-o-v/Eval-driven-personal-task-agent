# Decision Metrics Separate From Explanation Wording

[Experiment index](../README.md)

## Question and Context

A valid refusal could choose the right decision but match only one required reason keyword. Were keyword scores being mistaken for routing accuracy?

## Measurement Change

Preserve selected act/clarify/refuse in AgentResponse.decision_type, independently of the displayed clarification question or refusal reason.

Keep existing keyword and task-success rules intact. Add decision accuracy and decision coverage; later reporting derives macro-F1, invalid-output rate and the confusion matrix from saved decisions.

## Results and Meaning

The historical ACT-balanced legacy and decision-metric evaluations both recorded 47/150 task successes, tool accuracy 0.553 and argument accuracy 0.805. On template shift both recorded 6/30 successes.

The reporting change did not train a better model. It revealed a different aspect of the same task: correct routing can coexist with an incomplete or false explanation.

## Reading the Metrics

- Macro-F1 weights ACT, CLARIFY and REFUSE equally, instead of rewarding the majority class alone.
- Conditional decision accuracy excludes unknown decisions; always read it with coverage.
- Invalid outputs count as missed predictions for their expected class in macro-F1.
- Refusal/clarification accuracy measures the required message keywords, not just the selected label.
- Task success requires the broader scenario checks, not merely a correct decision.
- Hallucination rate currently measures unavailable tool names, not every invented fact.

## Limits

An invalid-output rate is not the rate of every runtime/schema error. A parser-valid call can still have wrong tool arguments or fail execution.

Core-agent runtime outcome versus selected decision is a separate concern documented in the bug report. Do not assume a final refusal message always means the model chose REFUSE. This note records measurement semantics, not an assertion that every runner's runtime paths have been fixed.

## Supporting Files

- [SCORING.md](../../../docs/SCORING.md)
- [BUG_REPORT.md](../../../docs/BUG_REPORT.md)
- [routing_metrics.py](../../../src/baymax/eval/routing_metrics.py)
- [act-balanced-train1k-combined-150-legacy-metrics.json](../../../results/phase2/qwen-lora/baymax-mix/act-balanced-train1k-combined-150-legacy-metrics.json)
- [act-balanced-train1k-combined-150-decision-metrics.json](../../../results/phase2/qwen-lora/baymax-mix/act-balanced-train1k-combined-150-decision-metrics.json)

Notes assembled on 2026-10-10 from saved artifacts and repository records. No training, inference or result modification was performed to write this note.
