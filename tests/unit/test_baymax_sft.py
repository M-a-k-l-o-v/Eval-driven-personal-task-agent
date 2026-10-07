from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from baymax.core.meta_tools import meta_tool_specs
from baymax.eval.agent_runner import FAKE_TOOL_SCHEMAS
from baymax.models import baymax_sft
from baymax.models.baymax_sft import (
    ACT_CLARIFY_REFUSE_TRIPLET_COUNT,
    CLARIFICATION_CONTRAST_PAIR_COUNT,
    REFUSAL_CONTRAST_PAIR_COUNT,
    TARGET_SYNTHETIC_EXAMPLE_COUNT,
    TARGETED_REFUSAL_CONTRAST_PAIR_COUNT,
    BaymaxSFTBuildConfig,
    SyntheticBaymaxExample,
    build_baymax_sft_dataset,
    generate_synthetic_baymax_examples,
    synthetic_baymax_example_to_sft_example,
    validate_synthetic_baymax_example,
)
from baymax.models.boundary_cases import BOUNDARY_CASES
from baymax.models.sft_dataset import DEFAULT_SFT_SYSTEM_PROMPT
from baymax.models.sft_validation import validate_sft_dataset, validate_sft_target_contract


def test_all_synthetic_targets_match_schemas_and_execute_against_adapters() -> None:
    schemas = {
        **FAKE_TOOL_SCHEMAS,
        **{spec["name"]: spec["parameters"] for spec in meta_tool_specs()},
    }
    for source in generate_synthetic_baymax_examples():
        validate_synthetic_baymax_example(source)
        for call in source.assistant_calls:
            errors = list(
                Draft202012Validator(schemas[call["name"]]).iter_errors(call["arguments"])
            )
            assert not errors, f"{source.id}: {[error.message for error in errors]}"


def test_matched_triplets_change_state_missing_field_or_capability_only() -> None:
    examples = generate_synthetic_baymax_examples(matched_contrasts=True)
    groups = defaultdict(list)
    for source in examples:
        if source.contrast_group and source.contrast_group.startswith(
            "act_clarify_refuse_triplet_"
        ):
            groups[source.contrast_group].append(source)
    assert len(groups) == ACT_CLARIFY_REFUSE_TRIPLET_COUNT
    for index, group in enumerate(groups.values()):
        act, clarify, refuse = group
        assert [e.contrast_role for e in group] == ["act", "clarify", "refuse"]
        for source in group:
            validate_synthetic_baymax_example(source)
            example = synthetic_baymax_example_to_sft_example(
                source, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT
            )

            assert not validate_sft_target_contract(example, location=source.id)
        assert act.available_tools == clarify.available_tools
        if index % 8 in (0, 4, 5):
            assert act.user_input == clarify.user_input
            assert act.initial_state != clarify.initial_state
        if index % 8 in (1, 2, 4, 5, 6):
            assert refuse.user_input == act.user_input
            assert refuse.initial_state == act.initial_state
            assert not set(act.available_tools) & set(refuse.available_tools)


def test_matched_build_is_versioned_exports_groups_and_does_not_overwrite(tmp_path: Path) -> None:
    config = BaymaxSFTBuildConfig(output_dir=tmp_path / "matched", matched_contrasts=True)
    manifest = build_baymax_sft_dataset(config)
    metadata = json.loads((config.output_dir / "contrast-groups.json").read_text(encoding="utf-8"))
    assert metadata["revision"] == manifest.revision
    assert manifest.revision != baymax_sft.baymax_synthetic_revision(
        generate_synthetic_baymax_examples()
    )
    assert baymax_sft.baymax_sft_build_config_hash(
        config
    ) != baymax_sft.baymax_sft_build_config_hash(
        config.model_copy(update={"matched_contrasts": False})
    )
    partitions = {
        source.source_row_id: split
        for split, entry in manifest.splits.items()
        for source in entry.source_rows
    }
    for members in metadata["groups"].values():
        assert len({partitions[source_id] for source_id in members}) == 1
    with pytest.raises(ValueError, match="new, empty"):
        build_baymax_sft_dataset(config)


@pytest.mark.parametrize(
    "initial_state",
    [
        {"clipboard": "legacy text"},
        {"calendar_events": [{"id": "evt_missing_fields", "title": "Meeting"}]},
        {"notion_tasks": [{"id": "task_bad", "title": "Task", "status": "invalid"}]},
        {"gmail_contacts": [{"name": "Person", "email": "invalid"}]},
    ],
)
def test_builder_rejects_bad_adapter_state_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial_state: dict[str, Any]
) -> None:
    source = _example_by_id("baymax_clarify_clipboard_text_001").model_copy(
        update={"initial_state": initial_state}
    )
    monkeypatch.setattr(baymax_sft, "generate_synthetic_baymax_examples", lambda: [source])
    output = tmp_path / "invalid-build"
    with pytest.raises(ValueError, match=f"{source.id}: invalid initial_state"):
        build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=output))
    assert not output.exists()


@pytest.mark.parametrize("arguments", [{"title": ""}, {}, {"title": "Task", "extra": True}])
def test_synthetic_validator_rejects_unexecutable_act_target(arguments: dict[str, Any]) -> None:
    source = _example_by_id("baymax_direct_notion_create_002").model_copy(
        update={"assistant_calls": [{"name": "notion.create_task", "arguments": arguments}]}
    )
    with pytest.raises(ValueError, match=f"{source.id}: target execution failed"):
        validate_synthetic_baymax_example(source)


def test_synthetic_validator_rejects_unavailable_target_tool() -> None:
    source = _example_by_id("baymax_direct_calendar_create_001").model_copy(
        update={"available_tools": ["calendar.update_event"]}
    )
    with pytest.raises(ValueError, match=f"{source.id}: unavailable target tool"):
        validate_synthetic_baymax_example(source)


def test_synthetic_validator_rejects_mixed_meta_and_productive_targets() -> None:
    source = _example_by_id("baymax_clarify_calendar_time_001")
    source = source.model_copy(
        update={
            "assistant_calls": source.assistant_calls
            + [{"name": "calendar.create_event", "arguments": {}}]
        }
    )
    with pytest.raises(ValueError, match="exactly one call"):
        validate_synthetic_baymax_example(source)


def test_corrected_clarifications_preserve_invalid_requests_and_ask_missing_fields() -> None:
    examples = generate_synthetic_baymax_examples()
    incomplete_schedule_examples = [
        e
        for e in examples
        if e.id == "baymax_clarify_calendar_time_001"
        or e.id.startswith("baymax_contrast_clarify_missing_time_")
        or (
            e.id.startswith("baymax_triplet_clarify_calendar_")
            and e.available_tools == ["calendar.create_event"]
        )
        or (e.id.startswith("baymax_generated_clarify_") and "Schedule " in e.user_input)
    ]
    assert incomplete_schedule_examples
    for source in incomplete_schedule_examples:
        assert source.behavior_type == "request_clarification"
        question = source.assistant_calls[0]["arguments"]["question"]
        assert "time" in question and "minutes" in question
        assert "minutes" not in source.user_input
        validate_synthetic_baymax_example(source)
    invalid_date = next(e for e in examples if "2026-02-30" in e.user_input)
    assert invalid_date.behavior_type == "request_clarification"
    assert "valid date" in invalid_date.assistant_calls[0]["arguments"]["question"]
    negative_duration = _example_by_id("capability_boundary_negative_event_length_clarify")
    assert "minus 20" in negative_duration.user_input
    validate_synthetic_baymax_example(negative_duration)


def test_indiscriminate_outreach_refusals_do_not_assume_consent() -> None:
    examples = generate_synthetic_baymax_examples()
    bulk_examples = [
        source
        for source in examples
        if source.behavior_type == "refuse_request"
        and "bulk" in source.assistant_calls[0]["arguments"]["reason"]
    ]
    assert len(bulk_examples) == 41
    for source in bulk_examples:
        assert "every" in source.user_input.lower()
        reason = source.assistant_calls[0]["arguments"]["reason"]
        assert "indiscriminate" in reason
        assert "unsolicited" not in reason
    explicit_send = _example_by_id("baymax_direct_gmail_send_001")
    assert explicit_send.behavior_type == "direct_tool_call"
    assert explicit_send.assistant_calls[0]["name"] == "gmail.send_email"


def test_send_request_with_draft_only_tools_refuses_before_clarifying() -> None:
    source = _example_by_id("baymax_refuse_multi_step_send_unavailable_001")
    assert source.user_input == "Create the task and email Marv about it."
    assert "gmail.create_draft" in source.available_tools
    assert "gmail.send_email" not in source.available_tools
    assert source.behavior_type == "refuse_request"
    assert len(source.assistant_calls) == 1
    assert source.assistant_calls[0]["name"] == "refuse_request"
    assert "gmail.send_email is unavailable" in source.assistant_calls[0]["arguments"]["reason"]
    validate_synthetic_baymax_example(source)


def test_draft_triplets_compose_known_facts_but_clarify_missing_deadline() -> None:
    examples = generate_synthetic_baymax_examples()
    drafts = [e for e in examples if e.id.startswith("baymax_triplet_act_gmail_draft_")]
    assert len(drafts) == 12
    for source in drafts:
        assert "about the" in source.user_input and "saying" not in source.user_input
        assert source.behavior_type == "direct_tool_call"
        call = source.assistant_calls[0]
        assert call["name"] == "gmail.create_draft"
        date = re.search(r"\d{4}-\d{2}-\d{2}", source.user_input)
        assert date is not None
        assert date.group() in call["arguments"]["body"]
        topic = call["arguments"]["body"].removeprefix("The ").split(" is due on ")[0]
        assert topic in source.user_input
        assert not {"reviewed", "finished", "submitted"} & set(call["arguments"]["body"].split())
        clarify = next(
            e
            for e in examples
            if e.contrast_group == source.contrast_group and e.contrast_role == "clarify"
        )
        assert "That deadline is not provided here" in clarify.user_input
        assert clarify.behavior_type == "request_clarification"
        assert "deadline" in clarify.assistant_calls[0]["arguments"]["question"]
        assert re.search(r"\d{4}-\d{2}-\d{2}", clarify.user_input) is None


def test_generate_synthetic_baymax_examples_covers_required_behavior_types() -> None:
    examples = generate_synthetic_baymax_examples()
    behavior_counts = Counter(example.behavior_type for example in examples)

    assert len(examples) == TARGET_SYNTHETIC_EXAMPLE_COUNT + 2 * len(BOUNDARY_CASES)
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
        + len(BOUNDARY_CASES)
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

    assert len(triplets) == ACT_CLARIFY_REFUSE_TRIPLET_COUNT == 100
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

    triplet_rows = [
        synthetic_baymax_example_to_sft_example(example, system_prompt="system").model_dump_json()
        for group_examples in triplets
        for example in group_examples
    ]
    assert len(triplet_rows) == 300
    assert len(set(triplet_rows)) == len(triplet_rows)
    assert all(
        {example.behavior_type for example in group_examples}
        == {"direct_tool_call", "request_clarification", "refuse_request"}
        for group_examples in triplets
    )


def test_authored_capability_pairs_are_unique_and_change_only_tool_availability() -> None:
    assert len(BOUNDARY_CASES) == len({case.key for case in BOUNDARY_CASES}) == 64
    assert len({case.request for case in BOUNDARY_CASES}) == 64
    assert Counter(case.tool.split(".")[0] for case in BOUNDARY_CASES) == {
        "calendar": 16,
        "gmail": 16,
        "notion": 16,
        "clipboard": 16,
    }
    examples = {example.id: example for example in generate_synthetic_baymax_examples()}
    unique_rows: set[str] = set()
    for case in BOUNDARY_CASES:
        group = f"capability_boundary_{case.key}"
        clarify = examples[f"{group}_clarify"]
        refuse = examples[f"{group}_refuse"]
        assert clarify.contrast_group == refuse.contrast_group == group
        assert clarify.user_input == refuse.user_input == case.request
        assert clarify.initial_state == refuse.initial_state == case.initial_state
        assert case.tool in clarify.available_tools
        assert case.tool not in refuse.available_tools
        assert clarify.assistant_calls == [
            {"name": "request_clarification", "arguments": {"question": case.question}}
        ]
        assert refuse.assistant_calls[0]["name"] == "refuse_request"
        assert case.tool in refuse.assistant_calls[0]["arguments"]["reason"]
        for example in (clarify, refuse):
            unique_rows.add(
                synthetic_baymax_example_to_sft_example(
                    example, system_prompt="system"
                ).model_dump_json()
            )
    assert len(unique_rows) == 128


def test_triplets_cover_contextual_act_and_disjoint_meta_tool_schemas() -> None:
    triplet_examples = [
        example
        for example in generate_synthetic_baymax_examples()
        if example.contrast_group is not None
        and example.contrast_group.startswith("act_clarify_refuse_triplet_")
    ]
    act_examples = [
        example for example in triplet_examples if example.behavior_type == "direct_tool_call"
    ]
    clarification_examples = [
        example for example in triplet_examples if example.behavior_type == "request_clarification"
    ]
    refusal_examples = [
        example for example in triplet_examples if example.behavior_type == "refuse_request"
    ]
    calendar_updates = [
        example
        for example in act_examples
        if example.assistant_calls[0]["name"] == "calendar.update_event"
    ]
    notion_updates = [
        example
        for example in act_examples
        if example.assistant_calls[0]["name"] == "notion.update_task"
    ]

    assert len(calendar_updates) >= 10
    assert len(notion_updates) >= 10
    assert all(
        set(example.assistant_calls[0]["arguments"]) == {"event_id", "start_time"}
        for example in calendar_updates
    )
    assert all(
        set(example.assistant_calls[0]["arguments"]) == {"task_id", "status"}
        for example in notion_updates
    )
    assert all(
        set(example.assistant_calls[0]["arguments"]) == {"question"}
        for example in clarification_examples
    )
    assert all(
        set(example.assistant_calls[0]["arguments"]) == {"reason"} for example in refusal_examples
    )
    assert all(
        not {"question", "reason"} & set(example.assistant_calls[0]["arguments"])
        for example in act_examples
    )
    assert any(
        "password" in example.user_input.casefold()
        and example.assistant_calls[0]["name"] == "clipboard.write"
        for example in act_examples
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
    assert "cannot send indiscriminate bulk email" in refusal_text
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


def test_build_baymax_sft_dataset_keeps_contrast_groups_in_one_split(tmp_path: Path) -> None:
    examples = generate_synthetic_baymax_examples()
    example_by_id = {example.id: example for example in examples}
    manifest = build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=tmp_path, seed=123))
    group_splits: dict[str, set[str]] = defaultdict(set)

    for split_name, split in manifest.splits.items():
        for source_row in split.source_rows:
            source_row_id = source_row.source_row_id
            assert isinstance(source_row_id, str)
            contrast_group = example_by_id[source_row_id].contrast_group
            if contrast_group is not None:
                group_splits[contrast_group].add(split_name)

    assert group_splits
    assert all(len(split_names) == 1 for split_names in group_splits.values())


def _example_by_id(example_id: str) -> SyntheticBaymaxExample:
    for example in generate_synthetic_baymax_examples():
        if example.id == example_id:
            return example

    raise AssertionError(f"missing synthetic BAYMAX example: {example_id}")


@pytest.mark.parametrize("seed", [1, 123])
def test_synthetic_splits_have_disjoint_content(tmp_path: Path, seed: int) -> None:
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=tmp_path, seed=seed))
    contents = [
        set((tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
        for split in ("train", "validation", "test")
    ]
    assert not contents[0] & contents[1]
    assert not contents[0] & contents[2]
    assert not contents[1] & contents[2]


def test_duplicate_content_connects_contrast_groups_transitively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = generate_synthetic_baymax_examples()[0]
    examples = [
        source.model_copy(update={"id": str(index), "contrast_group": group, "user_input": text})
        for index, (group, text) in enumerate(
            [
                ("a", "same-one"),
                ("b", "same-one"),
                ("b", "same-two"),
                ("c", "same-two"),
                ("c", "unique"),
                ("d", "different"),
            ]
        )
    ]
    monkeypatch.setattr(baymax_sft, "generate_synthetic_baymax_examples", lambda: examples)
    manifest = build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=tmp_path))
    placement = {
        row.source_row_id: split
        for split, entry in manifest.splits.items()
        for row in entry.source_rows
    }
    assert len({placement[str(index)] for index in range(5)}) == 1


def test_validator_detects_content_overlap_with_different_source_ids(tmp_path: Path) -> None:
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=tmp_path))
    train_line = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    validation_path = tmp_path / "validation.jsonl"
    lines = validation_path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(json.loads(train_line), indent=None)
    validation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_sft_dataset(tmp_path)
    assert not report.valid
    assert any(
        "identical SFT content" in issue.message for issue in report.splits["validation"].issues
    )
