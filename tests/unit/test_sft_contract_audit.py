from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from baymax.models.sft_contract_audit import find_contract_findings, matches_declared_type
from baymax.models.sft_dataset import SFTBuildConfig, build_sft_dataset_from_rows
from scripts.audit_sft_numeric_arguments import audit_dataset, main


def _audit(schema: dict[str, Any], value: Any, *, argument: str = "value") -> list[dict[str, Any]]:
    _, findings = find_contract_findings(
        [{"name": "example", "parameters": {argument: schema}}],
        [{"name": "example", "arguments": {argument: value}}],
    )
    return findings


@pytest.mark.parametrize(
    "value,declared,expected",
    [
        (1, "int", True),
        (1.0, "integer", True),
        (1.2, "int", False),
        (True, "int", False),
        ("1", "int", False),
        (1, "float", True),
        (True, "number", False),
        (False, "bool", True),
        (0, "bool", False),
        (None, "int, optional", False),
        (None, "Optional[int]", True),
        (None, ["string", "null"], True),
        ("a", "Union[int, str]", True),
        ([1, 2], "List[float]", True),
        ([1, "2"], "List[int]", False),
        ([[1, 2]], "List[List[int]]", True),
        ([[1, "2"]], "List[List[int]]", False),
        ([1, 2], "Tuple[float, float]", True),
        ([1], "Tuple[int, int]", False),
        ([1, 2, 3], "Tuple[int, ...]", True),
        ({"a": 1}, "Dict[str, int]", True),
        ({"a": "1"}, "Dict[str, int]", False),
        (1, "Unknown", None),
        ([1], "List[Optional[Unknown]]", None),
        (1, "__import__('os')", None),
        (float("inf"), "number", False),
        (float("nan"), "float", False),
    ],
)
def test_types_are_checked_without_coercion_or_expression_execution(
    value: Any,
    declared: Any,
    expected: bool | None,
) -> None:
    assert matches_declared_type(value, declared) is expected


def test_defaults_and_prose_are_separate_review_flags() -> None:
    findings = _audit({"type": "int", "default": "Darwin", "description": "Defaults to 10."}, 10)
    assert {f["issue"] for f in findings} == {"default_type_mismatch", "prose_default_conflict"}
    assert all(f["disposition"] == "review" for f in findings)


def test_numeric_equivalent_defaults_do_not_conflict_but_bool_and_int_do() -> None:
    assert _audit({"type": "float", "default": 1, "description": "Default is 1.0."}, 2) == []
    assert any(
        f["issue"] == "prose_default_conflict"
        for f in _audit(
            {"type": "bool", "default": True, "description": "Default is 1."},
            False,
        )
    )


def test_units_in_prose_are_not_guessed() -> None:
    assert _audit({"type": "int", "default": 60, "description": "Defaults to 1 minute."}, 90) == []


@pytest.mark.parametrize(
    "value", ["random.randint(1, 20)", "prior_call()[0]", "<cursor_from_previous_request>"]
)
def test_numeric_expression_placeholders_need_review(value: str) -> None:
    findings = _audit({"type": "int"}, value)
    assert {f["issue"] for f in findings} == {
        "target_type_mismatch",
        "unevaluated_argument_candidate",
    }
    assert all(f["disposition"] == "review" for f in findings)


def test_prose_code_and_placeholder_strings_are_not_treated_as_executable_arguments() -> None:
    assert _audit({"type": "str"}, "random.randint(1, 20)") == []
    assert _audit({"type": "str"}, "<cursor_from_previous_request>") == []


@pytest.mark.parametrize("argument", ["password", "api_key", "accessToken", "client_secret"])
def test_explicit_secret_arguments_propose_exclusion_without_exposing_values(argument: str) -> None:
    findings = _audit(
        {"type": "str", "default": "SECRET_SENTINEL"}, "SECRET_SENTINEL", argument=argument
    )
    assert [f["issue"] for f in findings] == ["credential_emitting_target"]
    assert findings[0]["disposition"] == "proposed_exclusion"
    assert "SECRET_SENTINEL" not in json.dumps(findings)


def test_description_can_identify_nonstandard_auth_field() -> None:
    findings = _audit(
        {"type": "str", "description": "Application token for API access."},
        "SENTINEL",
        argument="apptoken",
    )
    assert findings[0]["issue"] == "credential_emitting_target"


def test_nested_credentials_are_found_but_not_rendered() -> None:
    findings = _audit({"type": "object"}, {"password": "SENTINEL"}, argument="auth_payload")
    assert findings[0]["argument"] == "auth_payload.password"
    assert findings[0]["disposition"] == "proposed_exclusion"
    assert "SENTINEL" not in json.dumps(findings)


@pytest.mark.parametrize(
    "argument,description",
    [
        ("token", "The ID of a content token."),
        ("key", "Public record identifier."),
        ("text", "Password reset guidance."),
        ("text", "API key guidance text."),
        ("text", "Text explaining password for login guidance."),
        ("auth_id", "Authentication ID for the user."),
    ],
)
def test_identifiers_and_harmless_guidance_are_not_exclusions(
    argument: str, description: str
) -> None:
    assert (
        _audit({"type": "str", "description": description}, "password guidance", argument=argument)
        == []
    )


def test_ambiguous_token_and_unknown_types_are_review_only() -> None:
    findings = _audit({"type": "CustomToken"}, "abc", argument="token")
    assert {f["issue"] for f in findings} == {
        "unsupported_target_type",
        "ambiguous_credential_field",
    }
    assert all(f["disposition"] == "review" for f in findings)


def test_secret_named_identifier_is_review_only() -> None:
    findings = _audit(
        {"type": "str", "description": "Public API key identifier."},
        "record-123",
        argument="api_key",
    )
    assert [f["issue"] for f in findings] == ["ambiguous_credential_field"]
    assert findings[0]["disposition"] == "review"


def test_explicit_optional_null_is_not_an_exclusion() -> None:
    findings = _audit({"type": "int, optional", "default": None}, None)
    assert {f["issue"] for f in findings} == {"default_type_mismatch", "null_type_ambiguity"}
    assert all(f["disposition"] == "review" for f in findings)


def test_empty_secret_and_default_only_secret_do_not_propose_exclusion() -> None:
    assert _audit({"type": "str", "default": "SENTINEL"}, "", argument="password") == []
    assert find_contract_findings(
        [{"name": "login", "parameters": {"password": {"type": "str", "default": "SENTINEL"}}}], []
    ) == (0, [])


def test_duplicate_tools_are_reported_without_arbitrarily_selecting_a_schema() -> None:
    tools = [
        {"name": "x", "parameters": {"value": {"type": "int"}}},
        {"name": "x", "parameters": {"value": {"type": "str"}}},
        {"name": "login", "parameters": {"password": {"type": "str"}}},
    ]
    inspected, findings = find_contract_findings(
        tools,
        [
            {"name": "x", "arguments": {"value": "123"}},
            {"name": "login", "arguments": {"password": "SENTINEL"}},
        ],
    )
    assert inspected == 1
    assert {f["issue"] for f in findings} == {
        "duplicate_tool_name",
        "ambiguous_target_tool",
        "credential_emitting_target",
    }
    assert "SENTINEL" not in json.dumps(findings)


def test_json_schema_properties_work_and_undeclared_args_need_review() -> None:
    tools = [
        {"name": "x", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}}
    ]
    _, findings = find_contract_findings(
        tools, [{"name": "x", "arguments": {"n": "1", "extra": "x"}}]
    )
    assert {f["issue"] for f in findings} == {
        "target_type_mismatch",
        "undeclared_target_argument",
        "unsupported_target_type",
    }


@pytest.mark.parametrize("duplicate", [False, True])
def test_unresolved_tool_schemas_do_not_hide_credentials(duplicate: bool) -> None:
    tools = (
        [
            {"name": "login", "parameters": {"password": {"type": "str"}}},
            {"name": "login", "parameters": {"password": {"type": "int"}}},
        ]
        if duplicate
        else []
    )
    inspected, findings = find_contract_findings(
        tools,
        [
            {
                "name": "login",
                "arguments": {
                    "password": "SECRET_SENTINEL",
                    "credentials": [{"accessToken": "SECRET_SENTINEL"}],
                    "username": "ordinary-user",
                },
            }
        ],
    )
    expected = (
        {"duplicate_tool_name", "ambiguous_target_tool"}
        if duplicate
        else {"unavailable_target_tool"}
    )
    assert inspected == 0
    assert {f["issue"] for f in findings} == expected | {"credential_emitting_target"}
    exclusions = [f for f in findings if f["disposition"] == "proposed_exclusion"]
    assert {f["argument"] for f in exclusions} == {
        "password",
        "credentials[0].accessToken",
    }
    assert all(f["call_index"] == 0 for f in exclusions)
    assert "SECRET_SENTINEL" not in json.dumps(findings)


def test_contract_report_is_read_only_traceable_and_contains_no_raw_secrets(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    build_sft_dataset_from_rows(
        [
            {
                "id": 71,
                "query": "Log in with SECRET_SENTINEL.",
                "tools": [
                    {
                        "name": "login",
                        "parameters": {"password": {"type": "str"}, "api_key": {"type": "str"}},
                    }
                ],
                "answers": [
                    {
                        "name": "login",
                        "arguments": {"password": "SECRET_SENTINEL", "api_key": "SECRET_SENTINEL"},
                    }
                ],
            }
        ],
        config=SFTBuildConfig(output_dir=dataset, revision="pinned"),
    )
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    report = audit_dataset(dataset, checks="contracts")
    assert "SECRET_SENTINEL" not in json.dumps(report)
    assert report["checks"] == "contracts" and report["arguments_inspected"] == 2
    assert len(report["proposed_exclusions"]) == 1
    assert report["proposed_exclusions"][0]["source_id"] == 71
    assert len(report["proposed_exclusions"][0]["example_sha256"]) == 64
    output = tmp_path / "report.json"
    assert (
        main(["--dataset-dir", str(dataset), "--checks", "contracts", "--output", str(output)]) == 0
    )
    assert "SECRET_SENTINEL" not in output.read_text(encoding="utf-8")
    path = next(p for p in dataset.glob("*.jsonl") if p.stat().st_size)
    path.write_text('{"SECRET_SENTINEL":1}\n', encoding="utf-8")
    malformed = audit_dataset(dataset, checks="contracts")
    assert len(malformed["row_parse_errors"]) == 1
    assert "SECRET_SENTINEL" not in json.dumps(malformed)
    path.write_bytes(before[path.name])
    assert before == {p.name: p.read_bytes() for p in dataset.iterdir()}
