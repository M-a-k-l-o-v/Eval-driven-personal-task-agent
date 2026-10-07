from __future__ import annotations

import json
from pathlib import Path

import pytest

from baymax.models.sft_dataset import (
    SFTBuildConfig,
    build_sft_dataset_from_rows,
    convert_xlam_row_to_sft_example,
)
from baymax.models.sft_validation import (
    validate_sft_dataset,
    validate_sft_target_contract,
    validate_training_targets,
)


def test_validate_sft_dataset_accepts_valid_generated_dataset(tmp_path: Path) -> None:
    _build_dataset(tmp_path, row_count=10)

    report = validate_sft_dataset(tmp_path)

    assert report.valid is True
    assert report.manifest_loaded is True
    assert report.revision == "abc123"
    assert report.config_hash is not None
    assert report.converted_rows == 10
    assert report.loaded_rows == 10
    assert report.skipped_row_count == 0
    assert report.splits["train"].actual_count == 7
    assert report.splits["validation"].actual_count == 2
    assert report.splits["test"].actual_count == 1


def test_validate_sft_dataset_reports_missing_manifest(tmp_path: Path) -> None:
    report = validate_sft_dataset(tmp_path)

    assert report.valid is False
    assert report.manifest_loaded is False
    assert report.issues[0].message == "split manifest not found"


def test_validate_sft_dataset_reports_invalid_jsonl_line(tmp_path: Path) -> None:
    _build_dataset(tmp_path, row_count=10)
    _append_text(tmp_path / "train.jsonl", "not json\n")

    report = validate_sft_dataset(tmp_path)

    assert report.valid is False
    assert report.splits["train"].invalid_json_lines == 1
    assert any("invalid JSON" in issue.message for issue in report.splits["train"].issues)
    assert any("manifest count is 7" in issue.message for issue in report.splits["train"].issues)


def test_validate_sft_dataset_reports_wrong_message_roles(tmp_path: Path) -> None:
    _build_dataset(tmp_path, row_count=10)
    train_path = tmp_path / "train.jsonl"
    rows = _read_jsonl(train_path)
    rows[0]["messages"] = _message_list(rows[0])[:2]
    _write_jsonl(train_path, rows)

    report = validate_sft_dataset(tmp_path)

    assert report.valid is False
    assert any(
        "messages must have roles" in issue.message for issue in report.splits["train"].issues
    )


def test_validate_sft_dataset_reports_invalid_assistant_json(tmp_path: Path) -> None:
    _build_dataset(tmp_path, row_count=10)
    train_path = tmp_path / "train.jsonl"
    rows = _read_jsonl(train_path)
    _message_list(rows[0])[-1]["content"] = "[not json]"
    _write_jsonl(train_path, rows)

    report = validate_sft_dataset(tmp_path)

    assert report.valid is False
    assert any(
        "assistant content is invalid JSON" in issue.message
        for issue in report.splits["train"].issues
    )


def test_validate_sft_dataset_reports_split_overlap(tmp_path: Path) -> None:
    _build_dataset(tmp_path, row_count=10)
    manifest_path = tmp_path / "split-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["validation"]["source_rows"].append(
        manifest["splits"]["train"]["source_rows"][0]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_sft_dataset(tmp_path)

    assert report.valid is False
    assert any("appears in both train and validation" in issue.message for issue in report.issues)


def test_validate_sft_dataset_reports_missing_revision(tmp_path: Path) -> None:
    _build_dataset(tmp_path, row_count=10)
    manifest_path = tmp_path / "split-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_sft_dataset(tmp_path)

    assert report.valid is False
    assert any("dataset revision must be pinned" in issue.message for issue in report.issues)


@pytest.mark.parametrize(
    "calls",
    [
        [{"arguments": {"item_id": 1}}],
        [{"name": "missing", "arguments": {}}],
        [{"name": "lookup_item", "arguments": {"item_id": "wrong"}}],
        [{"name": "lookup_item", "arguments": {"unknown": 1}}],
        [{"name": "request_clarification", "arguments": {"reason": "Need details"}}],
        [{"name": "refuse_request", "arguments": {"question": "Which one?"}}],
        [{"name": "request_clarification", "arguments": {"question": " "}}],
        [
            {"name": "request_clarification", "arguments": {"question": "Which?"}},
            {"name": "lookup_item", "arguments": {"item_id": 1}},
        ],
        [],
    ],
)
def test_target_contract_rejects_bad_names_arguments_and_mixed_decisions(calls) -> None:
    row = _xlam_row(1)
    row["answers"] = json.dumps(calls)
    example = convert_xlam_row_to_sft_example(row, system_prompt="system")
    assert validate_sft_target_contract(example, location="train.jsonl:1")


@pytest.mark.parametrize(
    "name,field",
    [
        ("request_clarification", "question"),
        ("refuse_request", "reason"),
    ],
)
def test_target_contract_accepts_canonical_meta_tools(name: str, field: str) -> None:
    row = _xlam_row(1)
    row["answers"] = json.dumps([{"name": name, "arguments": {field: "Necessary text"}}])
    example = convert_xlam_row_to_sft_example(row, system_prompt="system")
    assert validate_sft_target_contract(example, location="test:1") == []


def test_target_contract_checks_declared_json_schema_without_exposing_values() -> None:
    row = _xlam_row(1)
    row["tools"] = json.dumps(
        [
            {
                "name": "lookup_item",
                "parameters": {
                    "type": "object",
                    "properties": {"item_id": {"type": "integer"}},
                    "required": ["item_id"],
                    "additionalProperties": False,
                },
            }
        ]
    )
    row["answers"] = json.dumps([{"name": "lookup_item", "arguments": {"item_id": "PRIVATE"}}])
    example = convert_xlam_row_to_sft_example(row, system_prompt="system")
    issues = validate_sft_target_contract(example, location="test:1")
    assert issues
    assert all("PRIVATE" not in issue.message for issue in issues)
    row["answers"] = json.dumps([{"name": "lookup_item", "arguments": {}}])
    assert validate_sft_target_contract(
        convert_xlam_row_to_sft_example(row, system_prompt="system"), location="test:1"
    )


def test_target_contract_fails_closed_for_unverifiable_types_and_references() -> None:
    for parameters in (
        {"item_id": {"type": "MysteryType"}},
        {"type": "object", "$ref": "https://example.invalid/schema"},
        {"type": "object", "$dynamicRef": "https://example.invalid/schema"},
    ):
        row = _xlam_row(1)
        row["tools"] = json.dumps([{"name": "lookup_item", "parameters": parameters}])
        example = convert_xlam_row_to_sft_example(row, system_prompt="system")
        assert validate_sft_target_contract(example, location="test:1")


@pytest.mark.parametrize(
    "definition,arguments",
    [
        ({"type": "int", "required": True}, {}),
        ({"type": "int", "minimum": 2}, {"item_id": 1}),
        ({"type": "int", "enum": [2, 3]}, {"item_id": 1}),
        ({"type": "list", "items": {"type": "integer"}}, {"item_id": ["wrong"]}),
        ({"type": "str", "format": "date"}, {"item_id": "2026-02-30"}),
    ],
)
def test_target_contract_checks_native_required_fields_and_nested_constraints(
    definition, arguments
) -> None:
    row = _xlam_row(1)
    row["tools"] = json.dumps([{"name": "lookup_item", "parameters": {"item_id": definition}}])
    row["answers"] = json.dumps([{"name": "lookup_item", "arguments": arguments}])
    example = convert_xlam_row_to_sft_example(row, system_prompt="system")
    assert validate_sft_target_contract(example, location="test:1")


def test_target_contract_rejects_missing_or_drifted_meta_tool_definitions() -> None:
    example = convert_xlam_row_to_sft_example(_xlam_row(1), system_prompt="system")
    header, context = example.messages[1].content.split("\n\nUser request:\n", 1)
    prefix, tools_json = header.split("\n", 1)
    tools = json.loads(tools_json)
    tools = [tool for tool in tools if tool["name"] != "refuse_request"]
    tools[0]["parameters"] = {"type": "object", "properties": {"reason": {"type": "string"}}}
    example.messages[1].content = (
        prefix + "\n" + json.dumps(tools) + "\n\nUser request:\n" + context
    )
    issues = validate_sft_target_contract(example, location="test:1")
    assert any("missing" in issue.message for issue in issues)
    assert any("differs" in issue.message for issue in issues)


def test_training_target_gate_reports_path_and_line_without_repairing() -> None:
    row = _xlam_row(1)
    row["answers"] = json.dumps([{"name": "request_clarification", "arguments": {"reason": "x"}}])
    example = convert_xlam_row_to_sft_example(row, system_prompt="system")
    original = example.model_dump(mode="json")
    with pytest.raises(ValueError, match="train.jsonl:1"):
        validate_training_targets([original], path=Path("train.jsonl"))
    assert original == example.model_dump(mode="json")


def _build_dataset(output_dir: Path, *, row_count: int) -> None:
    rows = [_xlam_row(index) for index in range(row_count)]
    config = SFTBuildConfig(
        dataset_id="example/tool-dataset",
        revision="abc123",
        output_dir=output_dir,
        seed=1,
    )
    build_sft_dataset_from_rows(rows, config=config)


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


def _message_list(row: dict[str, object]) -> list[dict[str, object]]:
    messages = row["messages"]
    assert isinstance(messages, list)

    typed_messages: list[dict[str, object]] = []
    for message in messages:
        assert isinstance(message, dict)
        typed_messages.append(message)

    return typed_messages


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _append_text(path: Path, text: str) -> None:
    existing_text = path.read_text(encoding="utf-8")
    path.write_text(existing_text + text, encoding="utf-8")
