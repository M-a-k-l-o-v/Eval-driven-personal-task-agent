from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from baymax.models.sft_dataset import SFTBuildConfig, build_sft_dataset_from_rows
from scripts.audit_sft_numeric_arguments import audit_dataset, find_numeric_mismatches, main


def _tools(declared_type: str = "int", *, json_schema: bool = False) -> list[dict[str, Any]]:
    properties = {"frequency": {"type": declared_type, "description": "Frequency in Hz."}}
    return [
        {
            "name": "measure",
            "parameters": (
                {"type": "object", "properties": properties, "required": ["frequency"]}
                if json_schema
                else properties
            ),
        }
    ]


def _calls(value: Any) -> list[dict[str, Any]]:
    return [{"name": "measure", "arguments": {"frequency": value}}]


def _dataset(tmp_path: Path, *, count: int = 1) -> Path:
    output = tmp_path / "input"
    rows = [
        {
            "id": 100 + index,
            "query": "Use a frequency of 261.63 Hz.",
            "tools": json.dumps(_tools()),
            "answers": json.dumps(_calls(261)),
        }
        for index in range(count)
    ]
    build_sft_dataset_from_rows(
        rows,
        config=SFTBuildConfig(output_dir=output, revision="pinned-test-revision"),
    )
    return output


def test_fractional_request_truncated_to_integer_is_a_review_candidate() -> None:
    inspected, findings = find_numeric_mismatches("Use 261.63 Hz.", _tools(), _calls(261))
    assert inspected == 1
    assert len(findings) == 1
    finding = findings[0]
    assert finding["issue"] == "possible_decimal_to_integer_conversion"
    assert finding["argument"] == "frequency"
    assert finding["target_value"] == 261
    assert finding["matched_literals"][0]["possible_conversions"] == ["truncate", "floor"]
    assert finding["review_priority"] == "medium"


@pytest.mark.parametrize(
    "literal,target", [("-261.63", -261), ("2.6163e2", 261), ("1,261.63", 1261)]
)
def test_decimal_detection_supports_signs_exponents_and_grouped_digits(
    literal: str, target: int
) -> None:
    _, findings = find_numeric_mismatches(f"Use {literal} Hz.", _tools(), _calls(target))
    assert findings[0]["matched_literals"][0]["literal"] == literal


def test_decimal_tokens_ignore_identifiers_and_versions() -> None:
    _, findings = find_numeric_mismatches("Version v261.63, ID item261.63.", _tools(), _calls(261))
    assert findings == []


@pytest.mark.parametrize("value", [True, "261", 261.63, None])
def test_integer_contract_mismatches_are_separate_from_conversion_candidates(value: Any) -> None:
    _, findings = find_numeric_mismatches("Use 261.63 Hz.", _tools(), _calls(value))
    assert [f["issue"] for f in findings] == ["integer_value_contract_mismatch"]


@pytest.mark.parametrize("value", [261, 261.0])
def test_integral_values_do_not_fail_integer_contract(value: int | float) -> None:
    _, findings = find_numeric_mismatches("Use 261 Hz.", _tools(), _calls(value))
    assert findings == []


def test_float_arguments_are_not_treated_as_integer_contracts() -> None:
    inspected, findings = find_numeric_mismatches("Use 261.63 Hz.", _tools("float"), _calls(261))
    assert inspected == 0 and findings == []


@pytest.mark.parametrize("declared_type,json_schema", [("int, optional", False), ("integer", True)])
def test_external_and_json_schema_integer_types(declared_type: str, json_schema: bool) -> None:
    _, findings = find_numeric_mismatches(
        "Use 261.63 Hz.", _tools(declared_type, json_schema=json_schema), _calls(261)
    )
    assert len(findings) == 1


def test_rounding_language_is_a_hint_not_an_automatic_exemption() -> None:
    _, findings = find_numeric_mismatches(
        "Round 261.63 Hz to the nearest integer.", _tools(), _calls(262)
    )
    assert findings[0]["rounding_language"] == "mentioned"
    assert findings[0]["review_priority"] == "low"
    _, forbidden = find_numeric_mismatches("Use 261.63 Hz; do not round.", _tools(), _calls(261))
    assert forbidden[0]["rounding_language"] == "prohibited"
    assert forbidden[0]["review_priority"] == "medium"


def test_unrelated_explicit_integer_does_not_hide_a_possible_conversion() -> None:
    _, findings = find_numeric_mismatches("Use 261.63 Hz for item 261.", _tools(), _calls(261))
    assert findings[0]["target_also_explicit_in_request"] is True
    assert findings[0]["review_priority"] == "low"


def test_audit_records_provenance_grouping_and_preserves_inputs(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    report = audit_dataset(dataset)
    finding = report["findings"][0]
    assert report["rows_scanned"] == report["flagged_rows"] == 1
    assert report["partial"] is False
    assert report["revision"] == "pinned-test-revision"
    assert finding["source_id"] == 100 and finding["source_index"] == 0
    assert finding["line"] == 1 and len(finding["example_sha256"]) == 64
    assert finding["request"] == "Use a frequency of 261.63 Hz."
    assert report["groups"][0]["findings"] == 1
    assert {p.name: p.read_bytes() for p in dataset.iterdir()} == before


def test_bad_row_is_reported_with_source_and_file_not_silently_skipped(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    path = next(p for p in dataset.glob("*.jsonl") if p.stat().st_size)
    path.write_text("{malformed\n", encoding="utf-8")
    report = audit_dataset(dataset)
    assert report["flagged_rows"] == 0
    assert len(report["row_parse_errors"]) == 1
    assert report["row_parse_errors"][0]["path"] == str(path)
    assert report["row_parse_errors"][0]["source_id"] == 100


def test_max_rows_cap_is_global_and_reported_as_partial(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=3)
    report = audit_dataset(dataset, max_rows=1)
    assert report["rows_scanned"] == 1 and report["partial"] is True
    with pytest.raises(ValueError, match="positive"):
        audit_dataset(dataset, max_rows=0)


def test_count_mismatch_prevents_false_source_provenance(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    path = next(p for p in dataset.glob("*.jsonl") if p.stat().st_size)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from manifest"):
        audit_dataset(dataset)


def test_cap_reaching_manifest_total_does_not_claim_full_file_verification(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    report = audit_dataset(dataset, max_rows=1)
    assert report["rows_scanned"] == 1 and report["partial"] is True
    assert audit_dataset(dataset, max_rows=2)["partial"] is False


def test_cli_writes_only_new_report_outside_input_dataset(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "review" / "report.json"
    args = ["--dataset-dir", str(dataset), "--output", str(output)]
    assert main(args) == 0
    content = output.read_bytes()
    assert json.loads(content)["flagged_rows"] == 1
    assert main(args) == 1 and output.read_bytes() == content
    assert main(["--dataset-dir", str(dataset), "--output", str(dataset / "report.json")]) == 1
    assert not (dataset / "report.json").exists()


def test_cli_missing_input_prints_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert (
        main(["--dataset-dir", str(tmp_path / "missing"), "--output", str(tmp_path / "report")])
        == 1
    )
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "report").exists()


def test_targets_audit_proposes_only_target_failures_and_credentials(tmp_path: Path) -> None:
    dataset = tmp_path / "targets"
    cases = [
        # A contradictory default does not make this explicitly supplied target invalid.
        (
            [{"name": "x", "parameters": {"n": {"type": "int", "default": None}}}],
            [{"name": "x", "arguments": {"n": 3}}],
        ),
        (
            [{"name": "x", "parameters": {"n": {"type": "int"}}}],
            [{"name": "x", "arguments": {"n": "SENSITIVE_SENTINEL"}}],
        ),
        ([{"name": "x", "parameters": {}}], [{"name": "missing", "arguments": {}}]),
        (
            [{"name": "x", "parameters": {}}, {"name": "x", "parameters": {}}],
            [{"name": "x", "arguments": {}}],
        ),
        (
            [{"name": "x", "parameters": {"f": {"type": "Callable[[float], float]"}}}],
            [{"name": "x", "arguments": {"f": "SENSITIVE_SENTINEL"}}],
        ),
        (
            [{"name": "loginuser", "parameters": {"password": {"type": "str"}}}] * 2,
            [{"name": "loginuser", "arguments": {"password": "SENSITIVE_SENTINEL"}}],
        ),
    ]
    build_sft_dataset_from_rows(
        [
            {
                "id": i,
                "query": f"Private request SENSITIVE_SENTINEL {i}",
                "tools": tools,
                "answers": calls,
            }
            for i, (tools, calls) in enumerate(cases)
        ],
        config=SFTBuildConfig(output_dir=dataset, revision="pinned"),
    )
    before = {p.name: p.read_bytes() for p in dataset.iterdir()}
    report = audit_dataset(dataset, checks="targets")
    assert report["target_policy"] == "sft-target-compatible-v1"
    assert report["rows_scanned"] == 6 and report["partial"] is False
    assert report["split_sha256"] == {
        split: hashlib.sha256((dataset / f"{split}.jsonl").read_bytes()).hexdigest()
        for split in ("train", "validation", "test")
    }
    proposals = {p["source_id"]: p for p in report["proposed_exclusions"]}
    assert set(proposals) == {1, 2, 3, 4, 5}
    assert proposals[5]["reasons"] == ["target_contract_incompatible", "credential_emitting_target"]
    assert all(len(p["example_sha256"]) == 64 for p in proposals.values())
    assert "SENSITIVE_SENTINEL" not in json.dumps(report)
    assert {p.name: p.read_bytes() for p in dataset.iterdir()} == before
    output = tmp_path / "target-report.json"
    assert (
        main(["--dataset-dir", str(dataset), "--checks", "targets", "--output", str(output)]) == 0
    )
    assert json.loads(output.read_text()) == report


def test_targets_audit_checks_meta_contracts_and_leaves_valid_text_alone(tmp_path: Path) -> None:
    dataset = tmp_path / "meta"
    build_sft_dataset_from_rows(
        [
            {
                "id": 1,
                "query": "Missing details",
                "tools": [],
                "answers": [{"name": "request_clarification", "arguments": {"reason": "Why?"}}],
            },
            {
                "id": 2,
                "query": "Missing details",
                "tools": [],
                "answers": [{"name": "request_clarification", "arguments": {"question": "Which?"}}],
            },
            {
                "id": 3,
                "query": "Unsupported",
                "tools": [],
                "answers": [{"name": "refuse_request", "arguments": {"reason": "Unavailable."}}],
            },
        ],
        config=SFTBuildConfig(output_dir=dataset, revision="pinned"),
    )
    report = audit_dataset(dataset, checks="targets")
    assert [p["source_id"] for p in report["proposed_exclusions"]] == [1]


def test_targets_audit_keeps_credential_inspection_errors_separate(tmp_path: Path) -> None:
    dataset = tmp_path / "nested"
    nested: Any = 1
    for _ in range(11):
        nested = [nested]
    build_sft_dataset_from_rows(
        [
            {
                "id": 7,
                "query": "Flatten nested values",
                "tools": [{"name": "flatten", "parameters": {"values": {"type": "list"}}}],
                "answers": [{"name": "flatten", "arguments": {"values": nested}}],
            }
        ],
        config=SFTBuildConfig(output_dir=dataset, revision="pinned"),
    )
    report = audit_dataset(dataset, checks="targets")
    assert report["proposed_exclusions"] == []
    assert len(report["row_parse_errors"]) == 1
    assert report["row_parse_errors"][0]["source_id"] == 7


def test_targets_audit_marks_capped_reports_partial(tmp_path: Path) -> None:
    report = audit_dataset(_dataset(tmp_path, count=3), checks="targets", max_rows=1)
    assert report["partial"] is True and report["rows_scanned"] == 1
