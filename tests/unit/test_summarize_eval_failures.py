from __future__ import annotations

import json

from scripts.summarize_eval_failures import format_failure_summary, summarize_failures


def test_summarize_failures_counts_reasons_and_scenarios(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "scenario_results": [
                    {
                        "scenario_id": "gmail_refusal_001",
                        "score": {"failure_reasons": ["premature_tool_call", "missing_refusal"]},
                    },
                    {
                        "scenario_id": "calendar_ambiguous_001",
                        "score": {
                            "failure_reasons": [
                                "premature_tool_call",
                                "missing_clarification",
                            ]
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_failures(result_path)

    assert rows[0].reason == "premature_tool_call"
    assert rows[0].count == 2
    assert rows[0].scenario_ids == ["gmail_refusal_001", "calendar_ambiguous_001"]


def test_format_failure_summary_handles_no_failures():
    assert format_failure_summary([]) == "No failure reasons found."
