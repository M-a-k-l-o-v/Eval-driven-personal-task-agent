from __future__ import annotations

import json

from baymax.models.sft_artifact_report import (
    build_sft_artifact_report,
    classify_sft_behavior,
)
from baymax.models.sft_dataset import SFTSourceRowRef


def test_classify_sft_behavior_detects_expected_classes() -> None:
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
    assert classify_sft_behavior(_row([{"name": "request_clarification", "arguments": {}}])) == (
        "request_clarification"
    )
    assert classify_sft_behavior(_row([{"name": "refuse_request", "arguments": {}}])) == (
        "refuse_request"
    )


def test_build_sft_artifact_report_counts_behavior_source_tokens_and_duplicates() -> None:
    duplicate_row = _row([{"name": "refuse_request", "arguments": {"reason": "No."}}])
    rows = [
        _row([{"name": "calendar.create_event", "arguments": {"title": "Study"}}]),
        duplicate_row,
        duplicate_row,
    ]
    source_rows = [
        _source("xlam:train:0:r0"),
        _source("baymax:train:1:r0"),
        _source("baymax:train:1:r1"),
    ]

    report = build_sft_artifact_report(
        rows=rows,
        tokenizer=_WhitespaceTokenizer(),
        max_sequence_length=128,
        source_rows=source_rows,
    )

    behavior = {row.behavior_class: row for row in report.behavior}
    source = {row.source: row for row in report.source}

    assert report.row_count == 3
    assert behavior["direct_tool_call"].rows == 1
    assert behavior["refuse_request"].rows == 2
    assert behavior["direct_tool_call"].trainable_tokens > 0
    assert behavior["refuse_request"].trainable_tokens > 0
    assert source["xlam"].rows == 1
    assert source["baymax"].rows == 2
    assert report.duplicates.exact_duplicate_rows == 1
    assert report.duplicates.unique_rows == 2
    duplicates_by_behavior = {row.behavior_class: row for row in report.duplicates_by_behavior}
    assert duplicates_by_behavior["direct_tool_call"].exact_duplicate_rows == 0
    assert duplicates_by_behavior["direct_tool_call"].unique_rows == 1
    assert duplicates_by_behavior["refuse_request"].exact_duplicate_rows == 1
    assert duplicates_by_behavior["refuse_request"].unique_rows == 1


def _row(calls: list[dict[str, object]]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(calls)},
        ]
    }


def _source(source_row_id: str) -> SFTSourceRowRef:
    return SFTSourceRowRef(source_index=0, source_row_id=source_row_id, row_hash="hash")


class _WhitespaceTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            text = f"{text}\nassistant:"
        return text

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        tokens = text.split()
        if truncation:
            tokens = tokens[:max_length]
        input_ids = list(range(len(tokens)))
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
