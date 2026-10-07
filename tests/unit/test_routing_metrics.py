from __future__ import annotations

import pytest

from baymax.eval.routing_metrics import routing_metrics
from baymax.eval.runner import ScenarioRunResult
from baymax.eval.scorer import AgentResponse, DecisionType, ScenarioScore


def _result(
    expected: DecisionType, actual: DecisionType | None, *, invalid: bool = False
) -> ScenarioRunResult:
    return ScenarioRunResult(
        scenario_id="test",
        latency_ms=0,
        tool_results=[],
        final_state={},
        agent_response=AgentResponse(
            decision_type=actual, parse_error="invalid" if invalid else None
        ),
        score=ScenarioScore(
            task_success=expected == actual and not invalid,
            tool_call_accuracy=0,
            argument_accuracy=0,
            unordered_tool_match=0,
            clarification_accuracy=0 if expected == "clarify" else None,
            refusal_accuracy=0 if expected == "refuse" else None,
            hallucination_rate=0,
            failure_reasons=["invalid_model_output"] if invalid else [],
        ),
    )


def test_routing_perfect_predictions_have_macro_f1_one() -> None:
    report = routing_metrics([_result(d, d) for d in ("act", "clarify", "refuse")])
    assert report["macro_f1"] == 1
    assert report["invalid_output_rate"] == 0
    assert report["decision_class_metrics"]["clarify"]["support"] == 1


def test_routing_invalid_and_unknown_outputs_remain_false_negatives() -> None:
    report = routing_metrics(
        [
            _result("act", "act"),
            _result("clarify", "clarify"),
            _result("refuse", "refuse"),
            _result("clarify", None, invalid=True),
            _result("refuse", None),
        ]
    )
    assert report["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)
    assert report["invalid_output_rate"] == 1 / 5
    matrix = report["decision_confusion_matrix"]
    assert matrix["clarify"]["invalid_or_unknown"] == 1
    assert matrix["refuse"]["invalid_or_unknown"] == 1
    assert sum(sum(row.values()) for row in matrix.values()) == 5


def test_always_act_is_not_rewarded_by_majority_support() -> None:
    results = [_result("act", "act") for _ in range(8)]
    results += [_result("clarify", "act"), _result("refuse", "act")]
    assert routing_metrics(results)["macro_f1"] == pytest.approx((16 / 18) / 3)


def test_empty_routing_metrics_are_not_zero_measurements() -> None:
    report = routing_metrics([])
    assert report["macro_f1"] is None
    assert report["invalid_output_rate"] is None


def test_rejected_output_cannot_count_as_correct_decision() -> None:
    report = routing_metrics([_result("act", "act", invalid=True)])
    assert report["macro_f1"] == 0
    assert report["decision_confusion_matrix"]["act"]["invalid_or_unknown"] == 1
