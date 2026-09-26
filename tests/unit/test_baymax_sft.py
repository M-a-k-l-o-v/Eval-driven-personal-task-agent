from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from baymax.models.baymax_sft import (
    ACT_CLARIFY_REFUSE_TRIPLET_COUNT,
    CLARIFICATION_CONTRAST_PAIR_COUNT,
    REFUSAL_CONTRAST_PAIR_COUNT,
    TARGETED_REFUSAL_CONTRAST_PAIR_COUNT,
    BaymaxSFTBuildConfig,
    SyntheticBaymaxExample,
    build_baymax_sft_dataset,
    generate_synthetic_baymax_examples,
    synthetic_baymax_example_to_sft_example,
)
from baymax.models.sft_dataset import DEFAULT_SFT_SYSTEM_PROMPT
from baymax.models.sft_validation import validate_sft_dataset


def test_generate_synthetic_baymax_examples_covers_required_behavior_types() -> None:
    examples = generate_synthetic_baymax_examples()
    behavior_counts = Counter(example.behavior_type for example in examples)

    assert 300 <= len(examples) <= 1000
    assert len({example.id for example in examples}) == len(examples)
    assert behavior_counts["request_clarification"] >= 1
    assert behavior_counts["refuse_request"] >= 1
    assert behavior_counts["direct_tool_call"] >= 1
    assert behavior_counts["multi_tool_sequence"] >= 1
    assert behavior_counts["request_clarification"] >= 1
    assert behavior_counts["refuse_request"] > behavior_counts["direct_tool_call"]


def test_generate_synthetic_baymax_examples_includes_contrast_pairs() -> None:
    contrast_groups: dict[str, list[SyntheticBaymaxExample]] = defaultdict(list)
    for example in generate_synthetic_baymax_examples():
        if example.contrast_group is not None:
            contrast_groups[example.contrast_group].append(example)

    assert len(contrast_groups) == (
        CLARIFICATION_CONTRAST_PAIR_COUNT
        + REFUSAL_CONTRAST_PAIR_COUNT
        + TARGETED_REFUSAL_CONTRAST_PAIR_COUNT
        + ACT_CLARIFY_REFUSE_TRIPLET_COUNT
    )

    clarification_pairs = [
        group_examples
        for contrast_group, group_examples in contrast_groups.items()
        if contrast_group.startswith("clarification_pair_")
    ]
    refusal_pairs = [
        group_examples
        for contrast_group, group_examples in contrast_groups.items()
        if contrast_group.startswith("refusal_pair_")
    ]
    triplets = [
        group_examples
        for contrast_group, group_examples in contrast_groups.items()
        if contrast_group.startswith("act_clarify_refuse_triplet_")
    ]

    pair_groups = clarification_pairs + refusal_pairs

    assert all(len(group_examples) == 2 for group_examples in pair_groups)
    assert all(len(group_examples) == 3 for group_examples in triplets)

    assert all(
        {example.contrast_role for example in group_examples} == {"missing_info", "explicit_info"}
        for group_examples in clarification_pairs
    )
    assert all(
        {example.behavior_type for example in group_examples}
        == {"request_clarification", "direct_tool_call"}
        for group_examples in clarification_pairs
    )
    assert all(
        {example.contrast_role for example in group_examples}
        == {"unsafe_or_unsupported", "safe_allowed"}
        for group_examples in refusal_pairs
    )
    assert all(
        {example.behavior_type for example in group_examples}
        == {"refuse_request", "direct_tool_call"}
        for group_examples in refusal_pairs
    )
    assert all(
        {example.contrast_role for example in group_examples} == {"act", "clarify", "refuse"}
        for group_examples in triplets
    )
    assert all(
        {example.behavior_type for example in group_examples}
        == {"direct_tool_call", "request_clarification", "refuse_request"}
        for group_examples in triplets
    )


def test_generate_synthetic_baymax_examples_covers_targeted_refusal_gaps() -> None:
    refusal_examples = [
        example
        for example in generate_synthetic_baymax_examples()
        if example.behavior_type == "refuse_request"
    ]
    refusal_text = "\n".join(
        f"{example.user_input} {example.assistant_calls}" for example in refusal_examples
    ).lower()

    assert "cannot list calendar events" in refusal_text
    assert "cannot list notion tasks" in refusal_text
    assert "cannot search email" in refusal_text
    assert "email is unavailable" in refusal_text
    assert "cannot order food" in refusal_text
    assert "cannot search notion tasks" in refusal_text
    assert "cannot access or copy saved passwords" in refusal_text
    assert "cannot send bulk unsolicited email" in refusal_text
    assert "cannot search calendar events" in refusal_text


def test_generate_synthetic_baymax_examples_covers_boundary_gaps() -> None:
    examples = generate_synthetic_baymax_examples()
    clarification_text = "\n".join(
        f"{example.user_input} {example.assistant_calls}"
        for example in examples
        if example.behavior_type == "request_clarification"
    ).lower()
    refusal_text = "\n".join(
        f"{example.user_input} {example.assistant_calls}"
        for example in examples
        if example.behavior_type == "refuse_request"
    ).lower()

    assert "confirm you want me to send this contract acceptance" in clarification_text
    assert "confirm you want me to send this resignation message" in clarification_text
    assert "what valid date should i use" in clarification_text
    assert "what valid email address should i use" in clarification_text
    assert "cannot make payments" in refusal_text
    assert "cannot access another person's account" in refusal_text


def test_default_sft_system_prompt_describes_clarification_and_refusal_policy() -> None:
    assert "request_clarification" in DEFAULT_SFT_SYSTEM_PROMPT
    assert "refuse_request" in DEFAULT_SFT_SYSTEM_PROMPT
    assert "missing tools or unsupported capabilities; refuse instead" in DEFAULT_SFT_SYSTEM_PROMPT
    assert "unsupported read/list/search" in DEFAULT_SFT_SYSTEM_PROMPT
    assert "routine explicit send" in DEFAULT_SFT_SYSTEM_PROMPT
    assert "unavailable destructive operation must be refused" in DEFAULT_SFT_SYSTEM_PROMPT
    assert "Do not invent" in DEFAULT_SFT_SYSTEM_PROMPT


def test_synthetic_baymax_example_to_sft_example_uses_baymax_prompt_shape() -> None:
    source = _example_by_id("baymax_direct_calendar_create_001")

    example = synthetic_baymax_example_to_sft_example(source, system_prompt="system")

    assert [message.role for message in example.messages] == ["system", "user", "assistant"]
    assert example.messages[0].content == "system"
    assert "Available tools, including internal meta-tools:" in example.messages[1].content
    assert "User request:" in example.messages[1].content
    assert "request_clarification" in example.messages[1].content
    assert "refuse_request" in example.messages[1].content

    assistant_payload = json.loads(example.messages[2].content)
    assert assistant_payload == [
        {
            "name": "calendar.create_event",
            "arguments": {
                "duration_minutes": 90,
                "start_date": "2026-06-02",
                "start_time": "15:30",
                "title": "calculus review",
            },
        }
    ]


def test_synthetic_baymax_examples_emit_meta_tool_calls_for_clarification_and_refusal() -> None:
    clarification = synthetic_baymax_example_to_sft_example(
        _example_by_id("baymax_clarify_gmail_recipient_001"),
        system_prompt="system",
    )
    refusal = synthetic_baymax_example_to_sft_example(
        _example_by_id("baymax_refuse_food_order_001"),
        system_prompt="system",
    )

    clarification_payload = json.loads(clarification.messages[2].content)
    refusal_payload = json.loads(refusal.messages[2].content)

    assert clarification_payload[0]["name"] == "request_clarification"
    assert "question" in clarification_payload[0]["arguments"]
    assert refusal_payload[0]["name"] == "refuse_request"
    assert "reason" in refusal_payload[0]["arguments"]


def test_synthetic_baymax_examples_emit_ordered_multi_tool_sequence() -> None:
    source = _example_by_id("baymax_multi_notion_gmail_001")

    example = synthetic_baymax_example_to_sft_example(source, system_prompt="system")

    assistant_payload = json.loads(example.messages[2].content)
    assert [call["name"] for call in assistant_payload] == [
        "notion.create_task",
        "gmail.create_draft",
    ]


def test_build_baymax_sft_dataset_writes_valid_jsonl_and_manifest(tmp_path: Path) -> None:
    manifest = build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=tmp_path, seed=123))

    report = validate_sft_dataset(tmp_path)

    assert manifest.dataset_id == "baymax/synthetic-control-v1"
    assert manifest.revision is not None
    assert manifest.loaded_rows == len(generate_synthetic_baymax_examples())
    assert report.valid is True
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "validation.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "split-manifest.json").exists()


def _example_by_id(example_id: str) -> SyntheticBaymaxExample:
    for example in generate_synthetic_baymax_examples():
        if example.id == example_id:
            return example

    raise AssertionError(f"missing synthetic BAYMAX example: {example_id}")
