from __future__ import annotations

import json
from pathlib import Path

import pytest

from baymax.eval.runner import ScenarioRunResult
from baymax.eval.scorer import AgentResponse, ScenarioScore
from scripts.compare_eval_results import format_markdown_table, load_result, main


def test_comparison_derives_routing_without_changing_saved_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    row = ScenarioRunResult(
        scenario_id="clarify",
        latency_ms=0,
        tool_results=[],
        final_state={},
        agent_response=AgentResponse(decision_type="clarify"),
        score=ScenarioScore(
            task_success=False,
            tool_call_accuracy=0,
            argument_accuracy=0,
            unordered_tool_match=0,
            clarification_accuracy=0,
            refusal_accuracy=None,
            hallucination_rate=0,
        ),
    )
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "scenario_count": 1,
                "aggregate_metrics": {"task_success_rate": 0},
                "scenario_results": [row.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    original = path.read_bytes()
    summary = load_result(f"saved={path}")
    assert summary.metrics["macro_f1"] == 1 / 3
    assert summary.metrics["invalid_output_rate"] == 0
    assert summary.metrics["task_success_rate"] == 0
    monkeypatch.setattr("sys.argv", ["compare", "--routing", f"saved={path}"])
    assert main() == 0
    assert "clarify | 0 | 1 | 0 | 0" in capsys.readouterr().out
    assert path.read_bytes() == original


def test_comparison_keeps_legacy_routing_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"scenario_count": 1, "aggregate_metrics": {}}), encoding="utf-8")
    summary = load_result(str(path))
    assert summary.metrics["macro_f1"] is None
    assert summary.routing is None


def test_load_result_accepts_labelled_path(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "backend_name": "local:qwen",
                "scenario_count": 2,
                "aggregate_metrics": {
                    "task_success_rate": 0.5,
                    "average_tool_call_accuracy": 1.0,
                    "average_argument_accuracy": 0.75,
                    "average_clarification_accuracy": None,
                    "average_refusal_accuracy": 0.0,
                    "average_decision_accuracy": 0.5,
                    "decision_coverage_rate": 1.0,
                    "average_hallucination_rate": 0.25,
                    "average_latency_ms": 1234.56,
                    "total_cost_usd": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = load_result(f"qwen={result_path}")

    assert summary.label == "qwen"
    assert summary.scenario_count == 2
    assert summary.metrics["task_success_rate"] == 0.5


def test_format_markdown_table_contains_metric_values(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "scenario_count": 1,
                "aggregate_metrics": {
                    "task_success_rate": 1.0,
                    "average_tool_call_accuracy": 1.0,
                    "average_argument_accuracy": 1.0,
                    "average_clarification_accuracy": None,
                    "average_refusal_accuracy": None,
                    "average_decision_accuracy": 1.0,
                    "decision_coverage_rate": 0.5,
                    "average_hallucination_rate": 0.0,
                    "average_latency_ms": 42.0,
                    "total_cost_usd": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )

    table = format_markdown_table([load_result(f"scripted={result_path}")])

    assert "scripted" in table
    assert "1.000" in table
    assert "n/a" in table
    assert "average_decision_accuracy" in table
    assert "decision_coverage_rate" in table
    assert "$0.0100" in table
