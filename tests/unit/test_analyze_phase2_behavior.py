from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_phase2_behavior import (
    analyze_failures_by_expected_behavior,
    analyze_trainable_tokens_by_behavior,
    classify_boundary_generation,
    classify_sft_behavior,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        suffix = "assistant:" if add_generation_prompt else ""
        return "\n".join(f"{message['role']}:{message['content']}" for message in messages) + suffix

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        add_special_tokens: bool,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        token_count = len(text.split())
        input_ids = list(range(token_count))
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def test_classify_sft_behavior_detects_boundary_and_tool_call_types() -> None:
    assert classify_sft_behavior(_row([{"name": "request_clarification", "arguments": {}}])) == (
        "request_clarification"
    )
    assert classify_sft_behavior(_row([{"name": "refuse_request", "arguments": {}}])) == (
        "refuse_request"
    )
    assert classify_sft_behavior(_row([{"name": "calendar.create_event", "arguments": {}}])) == (
        "direct_tool_call"
    )
    assert (
        classify_sft_behavior(
            _row(
                [
                    {"name": "notion.create_task", "arguments": {}},
                    {"name": "gmail.create_draft", "arguments": {}},
                ]
            )
        )
        == "multi_tool_sequence"
    )


def test_analyze_trainable_tokens_groups_rows_by_behavior() -> None:
    rows = [
        _row([{"name": "request_clarification", "arguments": {"question": "Who?"}}]),
        _row([{"name": "refuse_request", "arguments": {"reason": "Unsupported."}}]),
        _row([{"name": "calendar.create_event", "arguments": {"title": "Study"}}]),
    ]

    distribution = analyze_trainable_tokens_by_behavior(
        rows=rows,
        tokenizer=FakeTokenizer(),
        max_sequence_length=128,
    )

    counts = {row.behavior_class: row.row_count for row in distribution}
    tokens = {row.behavior_class: row.trainable_tokens for row in distribution}

    assert counts == {
        "direct_tool_call": 1,
        "refuse_request": 1,
        "request_clarification": 1,
    }
    assert all(value > 0 for value in tokens.values())


def test_analyze_failures_by_expected_behavior_groups_by_scenario_type() -> None:
    result_payloads = [
        (
            "aligned",
            Path("aligned.json"),
            {
                "scenario_results": [
                    {
                        "scenario_id": "clarify_001",
                        "score": {"failure_reasons": ["missing_clarification"]},
                    },
                    {
                        "scenario_id": "refuse_001",
                        "score": {"failure_reasons": []},
                    },
                ]
            },
        )
    ]

    rows = analyze_failures_by_expected_behavior(
        result_payloads=result_payloads,
        expected_behavior_by_id={"clarify_001": "clarification", "refuse_001": "refusal"},
    )

    assert rows[0].expected_behavior == "clarification"
    assert rows[0].failed_scenarios == 1
    assert rows[0].failure_counts == {"missing_clarification": 1}
    assert rows[1].expected_behavior == "refusal"
    assert rows[1].failed_scenarios == 0
    assert rows[1].failure_counts == {}


def test_classify_boundary_generation_identifies_common_failure_modes() -> None:
    assert (
        classify_boundary_generation(
            raw_model_output=json.dumps(
                [{"name": "calendar.create_event", "arguments": {"title": "Study"}}]
            ),
            expected_behavior="clarification",
        )
        == "premature_tool"
    )
    assert (
        classify_boundary_generation(
            raw_model_output=json.dumps(
                [{"name": "request_clarification", "arguments": {"question": "Who?"}}]
            ),
            expected_behavior="refusal",
        )
        == "wrong_meta_tool"
    )
    assert (
        classify_boundary_generation(
            raw_model_output=json.dumps(
                [{"name": "request_clarification", "arguments": {"question": "Who?"}}]
            ),
            expected_behavior="clarification",
        )
        == "correct_decision_wrong_wording"
    )
    assert (
        classify_boundary_generation(raw_model_output="not-json", expected_behavior="refusal")
        == "malformed_call"
    )


def _row(calls: list[dict[str, object]]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(calls)},
        ]
    }
