from __future__ import annotations

from collections import Counter
from pathlib import Path

from baymax.eval.scenario_loader import (
    ToolCallExpectedBehavior,
    ToolCallsExpectedBehavior,
    load_scenarios,
)

ROOT = Path(__file__).resolve().parents[2]


def test_v2_suite_matches_additive_coverage_blueprint() -> None:
    v1 = load_scenarios(ROOT / "scenarios" / "v1")
    v2 = load_scenarios(ROOT / "scenarios" / "v2")

    assert len(v1) == 50
    assert len(v2) == 100
    assert Counter(scenario.category for scenario in v2) == {
        "calendar": 20,
        "gmail": 20,
        "notion": 20,
        "clipboard": 20,
        "multi_tool": 20,
    }
    assert Counter(scenario.difficulty for scenario in v2) == {
        "explicit": 16,
        "implicit": 24,
        "contextual": 20,
        "ambiguous": 20,
        "multi_step": 20,
    }
    assert Counter(scenario.expected_behavior.type for scenario in v2) == {
        "tool_call": 36,
        "tool_calls": 20,
        "clarification": 24,
        "refusal": 20,
    }


def test_v2_suite_is_disjoint_from_v1_and_expected_tools_are_available() -> None:
    v1 = load_scenarios(ROOT / "scenarios" / "v1")
    v2 = load_scenarios(ROOT / "scenarios" / "v2")

    assert {scenario.id for scenario in v1}.isdisjoint(scenario.id for scenario in v2)
    assert {scenario.user_input.casefold().strip() for scenario in v1}.isdisjoint(
        scenario.user_input.casefold().strip() for scenario in v2
    )

    for scenario in v2:
        behavior = scenario.expected_behavior
        if isinstance(behavior, ToolCallExpectedBehavior):
            assert behavior.tool in scenario.available_tools
        elif isinstance(behavior, ToolCallsExpectedBehavior):
            assert all(call.tool in scenario.available_tools for call in behavior.calls)


def test_calendar_creation_clarifications_expose_create_tool() -> None:
    scenarios = {scenario.id: scenario for scenario in load_scenarios(ROOT / "scenarios" / "v2")}
    scenario_ids = {
        "calendar_v2_ambiguous_missing_date_001",
        "calendar_v2_ambiguous_missing_time_001",
        "calendar_v2_ambiguous_invalid_duration_001",
    }

    for scenario_id in scenario_ids:
        assert scenarios[scenario_id].available_tools == ["calendar.create_event"]


def test_clipboard_task_clarification_exposes_task_creation_tool() -> None:
    scenarios = {scenario.id: scenario for scenario in load_scenarios(ROOT / "scenarios" / "v2")}

    assert scenarios["clipboard_v2_contextual_whitespace_001"].available_tools == [
        "clipboard.read",
        "notion.create_task",
    ]
