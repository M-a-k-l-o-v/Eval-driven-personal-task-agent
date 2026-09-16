from __future__ import annotations

import json

from scripts.compare_eval_results import format_markdown_table, load_result


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
    assert "$0.0100" in table
