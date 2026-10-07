from __future__ import annotations

import json
from pathlib import Path

import pytest

from baymax.eval.lora_runner import (
    LocalLoraConfig,
    LocalLoraInvocation,
    build_lora_scenario_messages,
    parse_lora_agent_response,
    run_lora_scenario,
)
from baymax.eval.scenario_loader import load_scenario
from baymax.eval.scorer import AgentResponse, AgentToolCall, score_response

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "scenarios" / "v1"


def test_local_lora_config_allows_base_model_without_adapter() -> None:
    config = LocalLoraConfig()

    assert config.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.adapter_path is None


def test_parse_lora_agent_response_reads_xlam_tool_call() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "name": "calendar.create_event",
                    "arguments": {
                        "title": "Linear algebra revision",
                        "start_date": "2026-05-19",
                        "start_time": "16:00",
                        "duration_minutes": 120,
                    },
                }
            ]
        )
    )

    assert [call.tool for call in response.tool_calls] == ["calendar.create_event"]
    assert response.tool_calls[0].arguments["duration_minutes"] == 120
    assert response.message is None
    assert response.decision_type == "act"


def test_parse_lora_agent_response_reads_openai_style_function_call() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "function": {
                        "name": "notion.create_task",
                        "arguments": '{"title":"Physics lab report"}',
                    }
                }
            ]
        )
    )

    assert [call.tool for call in response.tool_calls] == ["notion.create_task"]
    assert response.tool_calls[0].arguments == {"title": "Physics lab report"}


def test_parse_lora_agent_response_reads_alternating_tool_argument_list() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                "calendar.create_event",
                {
                    "title": "Chemistry Review",
                    "start_date": "2026-05-18",
                    "start_time": "20:00",
                    "duration_minutes": 60,
                },
            ]
        )
    )

    assert [call.tool for call in response.tool_calls] == ["calendar.create_event"]
    assert response.tool_calls[0].arguments["title"] == "Chemistry Review"


def test_parse_lora_agent_response_reads_params_arguments() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "tool": "request_clarification",
                    "params": {"question": "Do you want me to cancel it?"},
                }
            ]
        )
    )

    assert response.tool_calls == []
    assert response.message == "Do you want me to cancel it?"


def test_parse_lora_agent_response_reads_function_name_variant() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "function_name": "request_clarification",
                    "arguments": {"question": "Which task should I update?"},
                }
            ]
        )
    )

    assert response.tool_calls == []
    assert response.message == "Which task should I update?"


def test_parse_lora_agent_response_reads_named_inputs_variant() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "function_key": "request_clarification",
                    "arguments": [{"text": "hunter2"}],
                    "named_inputs": {"question": "What do you mean by password?"},
                }
            ]
        )
    )

    assert response.tool_calls == []
    assert response.message == "What do you mean by password?"


def test_parse_lora_agent_response_reads_method_and_action_variants() -> None:
    method_response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "method": "calendar.create_event",
                    "arguments": {"title": "Calculus revision"},
                }
            ]
        )
    )
    action_response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "action": "gmail.send_email",
                    "arguments": {"recipient": "advisor@example.com"},
                }
            ]
        )
    )

    assert [call.tool for call in method_response.tool_calls] == ["calendar.create_event"]
    assert method_response.tool_calls[0].arguments == {"title": "Calculus revision"}
    assert [call.tool for call in action_response.tool_calls] == ["gmail.send_email"]
    assert action_response.tool_calls[0].arguments == {"recipient": "advisor@example.com"}


def test_parse_lora_agent_response_merges_list_arguments() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "functionName": "notion.create_task",
                    "arguments": [{"title": "Physics lab"}, {"status": "open"}],
                }
            ]
        )
    )

    assert [call.tool for call in response.tool_calls] == ["notion.create_task"]
    assert response.tool_calls[0].arguments == {"title": "Physics lab", "status": "open"}


def test_parse_lora_agent_response_converts_meta_clarification_to_message() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "name": "request_clarification",
                    "arguments": {"question": "Who should I email?"},
                }
            ]
        )
    )

    assert response.tool_calls == []
    assert response.message == "Who should I email?"
    assert response.decision_type == "clarify"


def test_parse_lora_agent_response_normalizes_meta_refusal_message() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            [
                {
                    "name": "refuse_request",
                    "arguments": {"reason": "Unsupported task list on Notion API"},
                }
            ]
        )
    )

    assert response.tool_calls == []
    assert response.message == (
        "I cannot comply with this request: Unsupported task list on Notion API"
    )
    assert response.decision_type == "refuse"


def test_parse_lora_agent_response_preserves_unparseable_text_as_message() -> None:
    response = parse_lora_agent_response("I need the recipient.")

    assert response.tool_calls == []
    assert response.message == "I need the recipient."
    assert response.decision_type is None


def test_parse_lora_agent_response_reads_explicit_decision_type() -> None:
    response = parse_lora_agent_response(
        json.dumps(
            {
                "tool_calls": [],
                "message": "That capability is unavailable.",
                "decision_type": "refuse",
            }
        )
    )

    assert response.decision_type == "refuse"


def test_build_lora_scenario_messages_includes_eval_context_and_tools() -> None:
    scenario = load_scenario(SCENARIO_DIR / "calendar_create_001.json")

    messages = build_lora_scenario_messages(scenario)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Available tools, including internal meta-tools:" in messages[1]["content"]
    assert "calendar.create_event" in messages[1]["content"]
    assert "request_clarification" in messages[1]["content"]
    assert "unsupported read/list/search" in messages[1]["content"]
    assert "Do not use for unavailable tools" in messages[1]["content"]
    assert "2026-05-18T09:00:00+00:00" in messages[1]["content"]
    assert "Schedule linear algebra revision" in messages[1]["content"]


def test_run_lora_scenario_scores_parsed_response() -> None:
    scenario = load_scenario(SCENARIO_DIR / "calendar_create_001.json")
    agent = _FakeLoraAgent(
        LocalLoraInvocation(
            raw_text="raw json",
            response=AgentResponse(
                tool_calls=[
                    AgentToolCall(
                        tool="calendar.create_event",
                        arguments={
                            "title": "Linear algebra revision",
                            "start_date": "2026-05-19",
                            "start_time": "16:00",
                            "duration_minutes": 120,
                        },
                    )
                ]
            ),
        )
    )

    result = run_lora_scenario(scenario, agent, run_id="run_test")

    assert result.scenario_id == "calendar_create_001"
    assert result.task_id == "run_test.calendar_create_001.0"
    assert result.raw_model_output == "raw json"
    assert result.agent_response is not None
    assert result.score.task_success is True
    assert result.final_state["calendar_events"][0]["title"] == "Linear algebra revision"


class _FakeLoraAgent:
    def __init__(self, invocation: LocalLoraInvocation) -> None:
        self._invocation = invocation

    def invoke(self, scenario: object) -> LocalLoraInvocation:
        del scenario
        return self._invocation


@pytest.mark.parametrize(
    "bad_call",
    [
        {"name": "delete_all_events", "arguments": {}},
        {"name": "clipboard.read", "arguments": "{bad-json"},
        {"name": "clipboard.read", "arguments": 7},
        {"name": "clipboard.read", "arguments": [{}, 7]},
        {"name": "request_clarification", "arguments": {"reason": "missing details"}},
        7,
    ],
)
def test_invalid_call_cannot_be_removed_to_produce_success(bad_call: object) -> None:
    raw = json.dumps(
        [
            {
                "name": "calendar.create_event",
                "arguments": {
                    "title": "Linear algebra revision",
                    "start_date": "2026-05-19",
                    "start_time": "16:00",
                    "duration_minutes": 120,
                },
            },
            bad_call,
        ]
    )
    response = parse_lora_agent_response(raw)
    scenario = load_scenario(SCENARIO_DIR / "calendar_create_001.json")
    result = run_lora_scenario(
        scenario, _FakeLoraAgent(LocalLoraInvocation(raw_text=raw, response=response))
    )
    assert response.parse_error is not None
    assert response.tool_calls == []
    assert not result.score.task_success
    assert "invalid_model_output" in result.score.failure_reasons
    assert result.raw_model_output == raw
    assert result.tool_results == []
    assert result.final_state["calendar_events"] == []


@pytest.mark.parametrize(
    "raw",
    [
        '[] [{"name":"calendar.delete_all","arguments":{}}]',
        '[{"name":"clipboard.read","arguments":{}}',
        '```json\n[]\n```\n[{"name":"calendar.delete_all","arguments":{}}]',
        '{"tool_calls":[{"tool":"badtool","arguments":{}}],"message":"cannot email"}',
        '{"tool_calls":"bad","message":"cannot email"}',
        '[{"name":"refuse_request","arguments":{"reason":"cannot email"}},'
        '{"name":"request_clarification","arguments":{"question":"who?"}}]',
    ],
)
def test_invalid_output_fails_even_when_message_matches_refusal(raw: str) -> None:
    response = parse_lora_agent_response(raw)
    scenario = load_scenario(SCENARIO_DIR / "gmail_refusal_001.json")
    score = score_response(scenario, response)
    assert response.parse_error is not None
    assert not score.task_success
    assert score.refusal_accuracy == 0.0


def test_parser_handles_json_delimiters_inside_strings() -> None:
    raw = json.dumps([{"name": "clipboard.write", "arguments": {"text": "literal }] marker"}}])
    response = parse_lora_agent_response(f"```json\n{raw}\n```")
    assert response.parse_error is None
    assert response.tool_calls[0].arguments["text"] == "literal }] marker"
