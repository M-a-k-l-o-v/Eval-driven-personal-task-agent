"""Balanced routing metrics without changing task-success or parser semantics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from baymax.eval.runner import ScenarioRunResult

DECISIONS = ("act", "clarify", "refuse")
UNKNOWN = "invalid_or_unknown"


def routing_metrics(results: Sequence[ScenarioRunResult]) -> dict[str, Any]:
    """Count unknown outputs as false negatives, not omitted predictions."""
    matrix = {expected: dict.fromkeys((*DECISIONS, UNKNOWN), 0) for expected in DECISIONS}
    invalid = 0
    for result in results:
        score = result.score
        if score.clarification_accuracy is not None and score.refusal_accuracy is not None:
            raise ValueError("A result cannot have two applicable boundary metrics.")
        expected = (
            "clarify"
            if score.clarification_accuracy is not None
            else "refuse"
            if score.refusal_accuracy is not None
            else "act"
        )
        response = result.agent_response
        rejected = "invalid_model_output" in score.failure_reasons or (
            response is not None and response.parse_error is not None
        )
        invalid += int(rejected)
        actual = response.decision_type if response is not None and not rejected else None
        matrix[expected][actual or UNKNOWN] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in DECISIONS:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[expected][label] for expected in DECISIONS)
        per_class[label] = {
            "support": support,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / support if support else 0.0,
            "f1": 2 * true_positive / (support + predicted) if support + predicted else 0.0,
        }
    return {
        "macro_f1": sum(float(row["f1"]) for row in per_class.values()) / 3 if results else None,
        "invalid_output_rate": invalid / len(results) if results else None,
        "decision_confusion_matrix": matrix,
        "decision_class_metrics": per_class,
    }
