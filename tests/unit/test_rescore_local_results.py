from __future__ import annotations

import json
from pathlib import Path

import pytest

from baymax.eval.cli import build_result_payload
from baymax.eval.runner import run_scenario
from baymax.eval.scenario_loader import load_scenario
from baymax.eval.scorer import AgentResponse, AgentToolCall
from scripts.rescore_local_results import rescore_local_results

SCENARIOS = Path(__file__).resolve().parents[2] / "scenarios" / "v1"


def test_rescore_preserves_source_and_rejects_invalid_extra_call(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIOS / "calendar_create_001.json")
    call = AgentToolCall(
        tool="calendar.create_event",
        arguments={
            "title": "Linear algebra revision",
            "start_date": "2026-05-19",
            "start_time": "16:00",
            "duration_minutes": 120,
        },
    )
    original = run_scenario(scenario, AgentResponse(tool_calls=[call]))
    assert original.score.task_success
    raw = json.dumps(
        [
            {"name": call.tool, "arguments": call.arguments},
            {"name": "delete_all_events", "arguments": {}},
        ]
    )
    original = original.model_copy(update={"raw_model_output": raw, "latency_ms": 123})
    payload = build_result_payload(results=[original], backend_name="local-lora:test")
    source = tmp_path / "source.json"
    original_text = json.dumps(payload)
    source.write_text(original_text, encoding="utf-8")
    target = tmp_path / "rescored.json"
    output = rescore_local_results(source, target, [SCENARIOS])
    row = output["scenario_results"][0]
    assert not row["score"]["task_success"]
    assert row["score"]["failure_reasons"] == ["invalid_model_output"]
    assert row["raw_model_output"] == raw
    assert row["latency_ms"] == 123
    assert row["tool_results"] == []
    assert row["final_state"]["calendar_events"] == []
    assert source.read_text(encoding="utf-8") == original_text
    assert output["benchmark_config"]["generation_rerun"] is False
    assert output["benchmark_config"]["source_run_id"] == payload["run_id"]
    with pytest.raises(FileExistsError):
        rescore_local_results(source, target, [SCENARIOS])


def test_rescore_rejects_missing_raw_generation(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIOS / "calendar_create_001.json")
    result = run_scenario(scenario, AgentResponse())
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(build_result_payload(results=[result], backend_name="local-lora:test")),
        encoding="utf-8",
    )
    target = tmp_path / "rescored.json"
    with pytest.raises(ValueError, match="Missing raw generation"):
        rescore_local_results(source, target, [SCENARIOS])
    assert not target.exists()
