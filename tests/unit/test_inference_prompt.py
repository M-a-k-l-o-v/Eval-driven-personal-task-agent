from __future__ import annotations

from baymax.service.inference import (
    _META_TOOLS,
    _PLAN_SYSTEM_PROMPT,
    _build_openai_tool_definitions,
)


def test_openai_plan_prompt_prioritizes_refusal_before_clarification() -> None:
    refusal_index = _PLAN_SYSTEM_PROMPT.index("refuse_request")
    clarification_index = _PLAN_SYSTEM_PROMPT.index("request_clarification")

    assert refusal_index < clarification_index
    assert "Missing tool or missing capability means refusal" in _PLAN_SYSTEM_PROMPT
    assert "unsupported read/list/search" in _PLAN_SYSTEM_PROMPT
    assert "Do not invent missing people" in _PLAN_SYSTEM_PROMPT


def test_openai_meta_tools_distinguish_missing_details_from_missing_capability() -> None:
    descriptions = {
        tool["function"]["name"]: tool["function"]["description"] for tool in _META_TOOLS
    }

    assert "Do not use for unavailable tools" in descriptions["request_clarification"]
    assert "missing tools/capabilities" in descriptions["refuse_request"]
    assert "unsupported read/list/search" in descriptions["refuse_request"]


def test_openai_tool_definitions_always_include_meta_tools() -> None:
    definitions = _build_openai_tool_definitions(
        ["calendar.create_event"],
        {"calendar.create_event": {"type": "object", "additionalProperties": False}},
    )
    names = [definition["function"]["name"] for definition in definitions]

    assert names[:2] == ["request_clarification", "refuse_request"]
    assert "calendar__create_event" in names
