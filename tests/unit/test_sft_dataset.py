from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from baymax.models import sft_dataset
from baymax.models.sft_dataset import (
    DEFAULT_SFT_SYSTEM_PROMPT,
    SFTBuildConfig,
    SFTConversionError,
    SFTSplitRatios,
    build_sft_dataset_from_rows,
    convert_xlam_row_to_sft_example,
    sft_build_config_hash,
)
from baymax.models.sft_validation import validate_sft_dataset


def test_sft_split_ratios_reject_non_unit_total() -> None:
    with pytest.raises(ValidationError):
        SFTSplitRatios(train=0.8, validation=0.2, test=0.2)


def test_convert_xlam_row_to_sft_example_uses_chat_messages_format() -> None:
    row = {
        "query": "Email Maya the notes.",
        "answers": '[{"name": "send_email", "arguments": {"recipient": "maya@example.com"}}]',
        "tools": '[{"name": "send_email", "parameters": {"recipient": {"type": "str"}}}]',
    }

    example = convert_xlam_row_to_sft_example(row, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT)

    assert [message.role for message in example.messages] == ["system", "user", "assistant"]
    assert example.messages[0].content == DEFAULT_SFT_SYSTEM_PROMPT
    assert "Available tools, including internal meta-tools:" in example.messages[1].content
    assert "Email Maya the notes." in example.messages[1].content
    assert "current_time" in example.messages[1].content
    assert "initial_state" in example.messages[1].content
    assert "request_clarification" in example.messages[1].content
    assert "refuse_request" in example.messages[1].content
    assert json.loads(example.messages[2].content) == [
        {"name": "send_email", "arguments": {"recipient": "maya@example.com"}}
    ]


def test_convert_xlam_row_to_sft_example_uses_baymax_user_request_shape() -> None:
    row = {
        "query": "Look up item 1.",
        "answers": '[{"name": "lookup_item", "arguments": {"item_id": 1}}]',
        "tools": '[{"name": "lookup_item", "parameters": {"item_id": {"type": "int"}}}]',
    }

    example = convert_xlam_row_to_sft_example(row, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT)
    content = example.messages[1].content
    tools_text, request_text = content.split("\n\nUser request:\n", 1)

    tools = json.loads(tools_text.removeprefix("Available tools, including internal meta-tools:\n"))
    request = json.loads(request_text)

    assert [tool["name"] for tool in tools[:2]] == ["request_clarification", "refuse_request"]
    assert tools[2]["name"] == "lookup_item"
    assert request == {
        "current_time": "2026-06-01T09:00:00Z",
        "initial_state": {},
        "request": "Look up item 1.",
    }


def test_convert_xlam_row_to_sft_example_rejects_malformed_answers() -> None:
    row = {
        "query": "Email Maya the notes.",
        "answers": "[not json]",
        "tools": "[]",
    }

    with pytest.raises(SFTConversionError, match="answers: invalid JSON"):
        convert_xlam_row_to_sft_example(row, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT)


def test_build_sft_dataset_from_rows_writes_splits_and_manifest(tmp_path: Path) -> None:
    rows = [_xlam_row(index) for index in range(10)]
    config = SFTBuildConfig(
        dataset_id="example/tool-dataset",
        revision="abc123",
        output_dir=tmp_path,
        seed=123,
    )

    manifest = build_sft_dataset_from_rows(rows, config=config)

    assert manifest.loaded_rows == 10
    assert manifest.converted_rows == 10
    assert manifest.skipped_rows == []
    assert manifest.splits["train"].count == 7
    assert manifest.splits["validation"].count == 2
    assert manifest.splits["test"].count == 1
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "validation.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert (tmp_path / "split-manifest.json").exists()

    train_rows = _read_jsonl(tmp_path / "train.jsonl")
    assert list(train_rows[0].keys()) == ["messages"]
    messages = train_rows[0]["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 3

    split_indexes = {
        split_name: {row.source_index for row in split.source_rows}
        for split_name, split in manifest.splits.items()
    }
    assert split_indexes["train"].isdisjoint(split_indexes["validation"])
    assert split_indexes["train"].isdisjoint(split_indexes["test"])
    assert split_indexes["validation"].isdisjoint(split_indexes["test"])
    assert set.union(*split_indexes.values()) == set(range(10))


def test_build_sft_dataset_from_rows_skips_malformed_rows(tmp_path: Path) -> None:
    rows = [
        _xlam_row(0),
        {"id": 1, "query": "bad", "answers": "[not json]", "tools": "[]"},
        "not a mapping",
    ]
    config = SFTBuildConfig(
        dataset_id="example/tool-dataset",
        revision="abc123",
        output_dir=tmp_path,
        seed=1,
    )

    manifest = build_sft_dataset_from_rows(rows, config=config)

    assert manifest.loaded_rows == 3
    assert manifest.converted_rows == 1
    assert [row.source_index for row in manifest.skipped_rows] == [1, 2]
    assert "answers: invalid JSON" in manifest.skipped_rows[0].reason
    assert manifest.skipped_rows[1].reason == "expected mapping, got str"


def test_sft_build_config_hash_is_stable_and_changes_with_content() -> None:
    first_config = SFTBuildConfig(
        dataset_id="example/tool-dataset",
        revision="abc123",
        seed=1,
    )
    second_config = SFTBuildConfig(
        dataset_id="example/tool-dataset",
        revision="abc123",
        seed=1,
    )
    changed_config = SFTBuildConfig(
        dataset_id="example/tool-dataset",
        revision="abc123",
        seed=2,
    )

    assert sft_build_config_hash(first_config) == sft_build_config_hash(second_config)
    assert sft_build_config_hash(first_config) != sft_build_config_hash(changed_config)


@pytest.mark.parametrize("seed", [1, 123])
def test_split_keeps_duplicate_content_together_without_dropping_source_rows(
    tmp_path: Path,
    seed: int,
) -> None:
    rows: list[dict[str, object]] = []
    for index in range(60):
        row = _xlam_row(index % 10)
        row["id"] = index
        # Equivalent input JSON can have different whitespace and key order.
        if (index // 10) % 2:
            row["answers"] = json.dumps(
                [{"arguments": {"item_id": index % 10}, "name": "lookup_item"}], indent=2
            )
        rows.append(row)
    config = SFTBuildConfig(output_dir=tmp_path, seed=seed, revision="fixture-v1")
    manifest = build_sft_dataset_from_rows(rows, config=config)

    content_splits: dict[str, set[str]] = {}
    source_indexes: list[int] = []
    for name, split in manifest.splits.items():
        contents = _read_jsonl(tmp_path / f"{name}.jsonl")
        assert len(contents) == split.count == len(split.source_rows)
        for row in contents:
            content_splits.setdefault(json.dumps(row, sort_keys=True), set()).add(name)
        source_indexes.extend(row.source_index for row in split.source_rows)
    assert len(content_splits) == 10
    assert all(len(names) == 1 for names in content_splits.values())
    assert sorted(source_indexes) == list(range(60))
    assert manifest.loaded_rows == manifest.converted_rows == 60
    assert not manifest.skipped_rows
    assert validate_sft_dataset(tmp_path).valid

    rebuilt_dir = tmp_path / "rebuilt"
    rebuilt = build_sft_dataset_from_rows(
        rows,
        config=config.model_copy(
            update={
                "output_dir": rebuilt_dir,
            }
        ),
    )
    assert rebuilt.config_hash == manifest.config_hash
    for split in manifest.splits:
        assert (tmp_path / f"{split}.jsonl").read_bytes() == (
            rebuilt_dir / f"{split}.jsonl"
        ).read_bytes()


def test_split_changes_with_seed(tmp_path: Path) -> None:
    rows = [_xlam_row(index) for index in range(30)]
    first = build_sft_dataset_from_rows(
        rows, config=SFTBuildConfig(output_dir=tmp_path / "a", seed=1)
    )
    second = build_sft_dataset_from_rows(
        rows, config=SFTBuildConfig(output_dir=tmp_path / "b", seed=2)
    )
    assert {row.source_index for row in first.splits["train"].source_rows} != {
        row.source_index for row in second.splits["train"].source_rows
    }


@pytest.mark.parametrize("count", [0, 10])
def test_empty_or_all_identical_data_never_splits_a_content_group(
    tmp_path: Path, count: int
) -> None:
    rows = [{**_xlam_row(0), "id": index} for index in range(count)]
    manifest = build_sft_dataset_from_rows(rows, config=SFTBuildConfig(output_dir=tmp_path))
    assert sum(split.count for split in manifest.splits.values()) == count
    assert sum(split.count > 0 for split in manifest.splits.values()) == (1 if count else 0)


def test_split_strategy_is_part_of_build_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SFTBuildConfig()
    current_hash = sft_build_config_hash(config)
    monkeypatch.setattr(sft_dataset, "SFT_SPLIT_STRATEGY", "legacy-row-split")
    assert sft_build_config_hash(config) != current_hash


def _xlam_row(index: int) -> dict[str, object]:
    return {
        "id": index,
        "query": f"Look up item {index}.",
        "answers": json.dumps(
            [{"name": "lookup_item", "arguments": {"item_id": index}}],
            separators=(",", ":"),
        ),
        "tools": json.dumps(
            [{"name": "lookup_item", "parameters": {"item_id": {"type": "int"}}}],
            separators=(",", ":"),
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
