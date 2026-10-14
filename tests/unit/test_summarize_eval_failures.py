from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.summarize_eval_failures import (
    format_failure_summary,
    main,
    summarize_failures,
    summarize_output_diagnostics,
)


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


def test_output_diagnostics_do_not_repair_or_rewrite_results(tmp_path):
    path = tmp_path / "result.json"
    rows = [
        {
            "scenario_id": "invalid_clarification",
            "raw_model_output": json.dumps(
                [{"name": "request_clarification", "arguments": {"reason": "Need details."}}]
            ),
            "agent_response": {"parse_error": "Invalid arguments", "decision_type": None},
            "score": {"failure_reasons": ["invalid_model_output"], "clarification_accuracy": 0},
        },
        {
            "scenario_id": "empty_array",
            "raw_model_output": "[]",
            "agent_response": {"parse_error": None, "decision_type": None},
            "score": {"failure_reasons": ["missing_refusal"], "refusal_accuracy": 0},
        },
        {
            "scenario_id": "valid_wrong_wording",
            "raw_model_output": json.dumps(
                [{"name": "request_clarification", "arguments": {"question": "Who?"}}]
            ),
            "agent_response": {"parse_error": None, "decision_type": "clarify"},
            "score": {"failure_reasons": ["wrong_clarification"], "clarification_accuracy": 0},
        },
        {
            "scenario_id": "recorded_parser_mismatch",
            "raw_model_output": json.dumps(
                [{"name": "refuse_request", "arguments": {"reason": "Unsupported operation."}}]
            ),
            "agent_response": {"parse_error": "Historical rejection", "decision_type": None},
            "score": {"failure_reasons": ["invalid_model_output"], "refusal_accuracy": 0},
        },
    ]
    path.write_text(json.dumps({"scenario_results": rows}), encoding="utf-8")
    before = path.read_bytes()
    original = summarize_failures(path)

    diagnostics = summarize_output_diagnostics(path)

    assert {row.reason: row.scenario_ids for row in diagnostics} == {
        "attempted_clarification_invalid_arguments": ["invalid_clarification"],
        "unknown_decision": ["empty_array"],
        "recorded_rejection_or_unknown_not_reproduced": ["recorded_parser_mismatch"],
    }
    assert summarize_failures(path) == original
    assert path.read_bytes() == before


def test_cli_can_print_separate_output_diagnostics(tmp_path, monkeypatch, capsys):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"scenario_results": []}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["summarize_eval_failures", str(path), "--output-diagnostics"])

    assert main() == 0
    assert "official failure reasons and scores are unchanged" in capsys.readouterr().out


def test_direct_script_invocation_supports_output_diagnostics(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"scenario_results": []}), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "summarize_eval_failures.py"

    result = subprocess.run(
        [sys.executable, str(script), str(path), "--output-diagnostics"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "official failure reasons and scores are unchanged" in result.stdout
