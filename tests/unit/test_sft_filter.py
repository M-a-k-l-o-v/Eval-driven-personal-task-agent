from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from baymax.models.sft_dataset import (
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SFTSplitRatios,
    convert_xlam_row_to_sft_example,
)
from baymax.models.sft_filter import filter_sft_dataset
from baymax.models.sft_validation import validate_sft_dataset


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source, output = tmp_path / "source", tmp_path / "filtered"
    source.mkdir()
    entries = {}
    selected = []
    gaps = []
    index = 0
    for split in ("train", "validation", "test"):
        rows, refs = [], []
        for n in range(3):
            example = convert_xlam_row_to_sft_example(
                {
                    "query": f"{split} request {n}",
                    "tools": [{"name": "test.call", "parameters": {}}],
                    "answers": [{"name": "test.call", "arguments": {}}],
                },
                system_prompt="Test prompt",
            )
            payload = example.model_dump(mode="json")
            rows.append(json.dumps(payload).encode() + b"\r\n")
            refs.append(
                SFTSourceRowRef(source_index=index, source_row_id=index, row_hash=str(index))
            )
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            record = {
                "split": split,
                "path": str(source / f"{split}.jsonl"),
                "line": n + 1,
                "source_index": index,
                "source_id": index,
                "example_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "reasons": ["credential_emitting_target"],
            }
            if n == 1:
                selected.append(record)
            if split == "train" and n == 0:
                gaps.append({k: v for k, v in record.items() if k not in ("path", "reasons")})
            index += 1
        (source / f"{split}.jsonl").write_bytes(b"".join(rows))
        entries[split] = SFTSplitManifestEntry(
            path=str(source / f"{split}.jsonl"), count=3, source_rows=refs
        )
    manifest = SFTSplitManifest(
        format_version="qwen-chat-messages-v2",
        config_hash="test-config",
        dataset_id="external-test",
        revision="pinned-test",
        source_split="train",
        seed=42,
        max_rows=None,
        split_ratios=SFTSplitRatios(),
        system_prompt_sha256=hashlib.sha256(b"Test prompt").hexdigest(),
        output_dir=str(source),
        splits=entries,
        loaded_rows=9,
        converted_rows=9,
        skipped_rows=[],
    )
    (source / "split-manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    audit_path, review_path = tmp_path / "audit.json", tmp_path / "review.json"
    errors = [
        {**{k: v for k, v in g.items() if k != "example_sha256"}, "error": "ValueError"}
        for g in gaps
    ]
    audit = {
        "audit_version": 1,
        "checks": "contracts",
        "partial": False,
        "credential_policy": "adapter-managed-auth-v1",
        "dataset_id": manifest.dataset_id,
        "revision": manifest.revision,
        "rows_scanned": 9,
        "manifest_sha256": _digest(source / "split-manifest.json"),
        "proposed_exclusions": selected,
        "row_parse_errors": errors,
    }
    _write(audit_path, audit)
    _write(
        review_path,
        {
            "format_version": "credential-filter-review-v1",
            "policy": "adapter-managed-auth-v1",
            "audit_sha256": _digest(audit_path),
            "manifest_sha256": audit["manifest_sha256"],
            "approved_rows": selected,
            "schema_reviews": [],
            "retained_coverage_gaps": [
                {**g, "error": "ValueError", "resolution": "Reviewed nesting-only gap."}
                for g in gaps
            ],
        },
    )
    return source, audit_path, review_path, output


def _run(paths: tuple[Path, Path, Path, Path]) -> SFTSplitManifest:
    source, audit, review, output = paths
    return filter_sft_dataset(
        dataset_dir=source, audit_path=audit, review_path=review, output_dir=output
    )


def test_filter_preserves_bytes_order_splits_and_parent(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    source, _, review, output = paths
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    manifest = _run(paths)

    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    for split, entry in manifest.splits.items():
        lines = before[f"{split}.jsonl"].splitlines(keepends=True)
        assert (output / f"{split}.jsonl").read_bytes() == lines[0] + lines[2]
        assert entry.count == 2
        assert [ref.source_index for ref in entry.source_rows] == [
            ref.source_index
            for ref in SFTSplitManifest.model_validate_json(before["split-manifest.json"])
            .splits[split]
            .source_rows
            if ref.source_index % 3 != 1
        ]
    assert manifest.loaded_rows == 9
    assert manifest.converted_rows == 6
    assert len(manifest.skipped_rows) == 3
    assert (output / "filter-review.json").read_bytes() == review.read_bytes()
    provenance = json.loads((output / "filter-provenance.json").read_text())
    assert provenance["parent_revision"] == "pinned-test"
    assert provenance["retained_coverage_gap_rows"] == 1
    assert validate_sft_dataset(output).valid


def test_filter_removes_only_explicitly_approved_proposals(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    review = json.loads(paths[2].read_text())
    review["approved_rows"] = review["approved_rows"][:1]
    _write(paths[2], review)
    manifest = _run(paths)
    assert manifest.splits["train"].count == 2
    assert manifest.splits["validation"].count == manifest.splits["test"].count == 3


@pytest.mark.parametrize(
    "defect",
    [
        "audit_hash",
        "manifest_hash",
        "partial",
        "wrong_policy",
        "scan_count",
        "duplicate_approval",
        "unproposed",
        "missing_gap",
        "gap_provenance",
        "bad_gap_hash",
        "selected_hash",
        "selected_provenance",
        "row_count",
        "outside_line",
        "escaped_source",
    ],
)
def test_filter_rejects_stale_or_incomplete_evidence(tmp_path: Path, defect: str) -> None:
    paths = _setup(tmp_path)
    source, audit_path, review_path, output = paths
    audit, review = json.loads(audit_path.read_text()), json.loads(review_path.read_text())
    manifest_path = source / "split-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if defect == "partial":
        audit["partial"] = True
    elif defect == "wrong_policy":
        audit["credential_policy"] = "other"
    elif defect == "scan_count":
        audit["rows_scanned"] = 8
    elif defect == "duplicate_approval":
        review["approved_rows"].append(review["approved_rows"][0])
    elif defect == "unproposed":
        review["approved_rows"][0]["source_id"] = 100
    elif defect == "missing_gap":
        review["retained_coverage_gaps"] = []
    elif defect == "gap_provenance":
        review["retained_coverage_gaps"][0]["source_id"] = 100
    elif defect == "bad_gap_hash":
        review["retained_coverage_gaps"][0]["example_sha256"] = "0" * 64
    elif defect == "selected_hash":
        path = source / "train.jsonl"
        path.write_bytes(path.read_bytes().replace(b"train request 1", b"changed request"))
    elif defect == "selected_provenance":
        manifest["splits"]["train"]["source_rows"][1]["source_row_id"] = 100
    elif defect == "row_count":
        path = source / "train.jsonl"
        path.write_bytes(path.read_bytes() + path.read_bytes().splitlines(keepends=True)[0])
    elif defect == "outside_line":
        audit["proposed_exclusions"][0]["line"] = 99
        review["approved_rows"][0]["line"] = 99
    elif defect == "escaped_source":
        escaped = tmp_path / "outside.jsonl"
        escaped.write_bytes((source / "train.jsonl").read_bytes())
        manifest["splits"]["train"]["path"] = str(escaped)
    _write(manifest_path, manifest)
    audit["manifest_sha256"] = review["manifest_sha256"] = _digest(manifest_path)
    _write(audit_path, audit)
    review["audit_sha256"] = _digest(audit_path)
    if defect == "audit_hash":
        review["audit_sha256"] = "0" * 64
    if defect == "manifest_hash":
        review["manifest_sha256"] = "0" * 64
    _write(review_path, review)
    with pytest.raises(ValueError):
        _run(paths)
    assert not output.exists()
    assert not list(tmp_path.glob(".filtered-*"))


def test_filter_identity_is_reproducible_at_different_output_paths(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    first = _run(paths)
    second = _run((*paths[:3], tmp_path / "second"))
    assert first.revision == second.revision
    assert first.config_hash == second.config_hash


def _target_setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    paths = _setup(tmp_path)
    source, audit_path, review_path, _ = paths
    audit, review = json.loads(audit_path.read_text()), json.loads(review_path.read_text())
    audit["checks"] = "targets"
    audit["target_policy"] = "sft-target-compatible-v1"
    review["format_version"] = "target-filter-review-v1"
    review["policy"] = "sft-target-compatible-v1"
    for row in audit["proposed_exclusions"]:
        path = source / (row["split"] + ".jsonl")
        lines = path.read_bytes().splitlines(keepends=True)
        example = json.loads(lines[row["line"] - 1])
        example["messages"][2]["content"] = '[{"name":"test.call","arguments":{"extra":1}}]'
        lines[row["line"] - 1] = json.dumps(example).encode() + b"\r\n"
        path.write_bytes(b"".join(lines))
        row["example_sha256"] = hashlib.sha256(
            json.dumps(example, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        row["reasons"] = ["target_contract_incompatible"]
    review["approved_rows"] = audit["proposed_exclusions"]
    audit["split_sha256"] = {
        split: _digest(source / f"{split}.jsonl") for split in ("train", "validation", "test")
    }
    _write(audit_path, audit)
    review["audit_sha256"] = _digest(audit_path)
    _write(review_path, review)
    return paths


def test_target_filter_removes_reviewed_failures_without_changing_retained_rows(
    tmp_path: Path,
) -> None:
    paths = _target_setup(tmp_path)
    before = {p.name: p.read_bytes() for p in paths[0].iterdir()}
    assert not validate_sft_dataset(paths[0]).valid
    result = _run(paths)
    assert result.converted_rows == 6
    assert validate_sft_dataset(paths[3]).valid
    assert {p.name: p.read_bytes() for p in paths[0].iterdir()} == before
    for split in result.splits:
        lines = before[f"{split}.jsonl"].splitlines(keepends=True)
        assert (paths[3] / f"{split}.jsonl").read_bytes() == lines[0] + lines[2]
    provenance = json.loads((paths[3] / "filter-provenance.json").read_text())
    assert provenance["policy"] == "sft-target-compatible-v1"
    assert provenance["filter_version"] == 2
    assert all(
        row.reason == "target-filter-v1:sft-target-compatible-v1" for row in result.skipped_rows
    )


@pytest.mark.parametrize(
    "defect",
    ["checks", "policy", "partial", "unapproved", "selected_hash", "retained_hash", "missing_hash"],
)
def test_target_filter_rejects_wrong_or_stale_evidence(tmp_path: Path, defect: str) -> None:
    paths = _target_setup(tmp_path)
    audit, review = json.loads(paths[1].read_text()), json.loads(paths[2].read_text())
    if defect == "checks":
        audit["checks"] = "contracts"
    elif defect == "policy":
        audit["target_policy"] = "wrong-policy"
    elif defect == "partial":
        audit["partial"] = True
    elif defect == "unapproved":
        review["approved_rows"][0]["reasons"] = ["credential_emitting_target"]
    elif defect == "retained_hash":
        path = paths[0] / "train.jsonl"
        path.write_bytes(path.read_bytes().replace(b"train request 2", b"changed request"))
    elif defect == "missing_hash":
        audit.pop("split_sha256")
    else:
        path = paths[0] / "train.jsonl"
        path.write_bytes(path.read_bytes().replace(b"train request 1", b"changed request"))
    _write(paths[1], audit)
    review["audit_sha256"] = _digest(paths[1])
    _write(paths[2], review)
    with pytest.raises(ValueError):
        _run(paths)
    assert not paths[3].exists()
    assert not list(tmp_path.glob(".filtered-*"))


def test_credential_review_cannot_approve_target_exclusions(tmp_path: Path) -> None:
    paths = _target_setup(tmp_path)
    audit, review = json.loads(paths[1].read_text()), json.loads(paths[2].read_text())
    audit["checks"] = "contracts"
    audit.pop("target_policy")
    review["format_version"] = "credential-filter-review-v1"
    review["policy"] = "adapter-managed-auth-v1"
    _write(paths[1], audit)
    review["audit_sha256"] = _digest(paths[1])
    _write(paths[2], review)
    with pytest.raises(ValueError, match="Credential-only"):
        _run(paths)
    assert not paths[3].exists()


def test_target_audit_to_review_to_filter_integration(tmp_path: Path) -> None:
    from baymax.models.sft_dataset import SFTBuildConfig, build_sft_dataset_from_rows
    from scripts.audit_sft_numeric_arguments import audit_dataset

    source, output = tmp_path / "source", tmp_path / "filtered"
    tool = {"name": "measure", "parameters": {"n": {"type": "int", "default": None}}}
    login = {"name": "login", "parameters": {"password": {"type": "str"}}}
    build_sft_dataset_from_rows(
        [
            {
                "id": 1,
                "query": "Measure 3",
                "tools": [tool],
                "answers": [{"name": "measure", "arguments": {"n": 3}}],
            },
            {
                "id": 2,
                "query": "Measure 3",
                "tools": [tool],
                "answers": [{"name": "measure", "arguments": {"n": "3"}}],
            },
            {
                "id": 3,
                "query": "Authenticate",
                "tools": [login, login],
                "answers": [{"name": "login", "arguments": {"password": "SENTINEL"}}],
            },
            {
                "id": 4,
                "query": "Unsupported",
                "tools": [],
                "answers": [{"name": "refuse_request", "arguments": {"reason": "Unavailable."}}],
            },
        ],
        config=SFTBuildConfig(output_dir=source, revision="pinned"),
    )
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    audit = audit_dataset(source, checks="targets")
    assert audit["row_parse_errors"] == []
    assert {p["source_id"] for p in audit["proposed_exclusions"]} == {2, 3}
    audit_path, review_path = tmp_path / "audit.json", tmp_path / "review.json"
    _write(audit_path, audit)
    _write(
        review_path,
        {
            "format_version": "target-filter-review-v1",
            "policy": "sft-target-compatible-v1",
            "audit_sha256": _digest(audit_path),
            "manifest_sha256": audit["manifest_sha256"],
            "approved_rows": audit["proposed_exclusions"],
            "schema_reviews": [],
            "retained_coverage_gaps": [],
        },
    )
    manifest = filter_sft_dataset(
        dataset_dir=source, audit_path=audit_path, review_path=review_path, output_dir=output
    )
    assert manifest.converted_rows == 2
    assert {
        ref.source_row_id for split in manifest.splits.values() for ref in split.source_rows
    } == {1, 4}
    assert validate_sft_dataset(output).valid
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


@pytest.mark.parametrize("output_name", ["source", "source/nested", "filtered"])
def test_filter_never_overwrites_existing_or_nested_output(
    tmp_path: Path, output_name: str
) -> None:
    paths = _setup(tmp_path)
    if output_name == "filtered":
        paths[3].mkdir()
    with pytest.raises(ValueError):
        _run((*paths[:3], tmp_path / output_name))


def test_cli_handles_invalid_review_without_echoing_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _setup(tmp_path)
    review = json.loads(paths[2].read_text())
    review["approved_rows"][0]["line"] = "SENSITIVE-SENTINEL"
    _write(paths[2], review)
    script = Path(__file__).resolve().parents[2] / "scripts" / "filter_sft_dataset.py"
    spec = importlib.util.spec_from_file_location("filter_cli", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.main(
            [
                "--dataset-dir",
                str(paths[0]),
                "--audit",
                str(paths[1]),
                "--review",
                str(paths[2]),
                "--output-dir",
                str(paths[3]),
            ]
        )
        == 1
    )
    assert "SENSITIVE-SENTINEL" not in capsys.readouterr().err


def test_filter_rejects_non_object_audit(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _write(paths[1], [])
    with pytest.raises(ValueError, match="Audit must be a JSON object"):
        _run(paths)
    assert not paths[3].exists()


@pytest.mark.parametrize("section", ["proposed_exclusions", "row_parse_errors"])
def test_filter_rejects_duplicate_audit_locations(tmp_path: Path, section: str) -> None:
    paths = _setup(tmp_path)
    audit = json.loads(paths[1].read_text())
    audit[section].append(audit[section][0])
    _write(paths[1], audit)
    review = json.loads(paths[2].read_text())
    review["audit_sha256"] = _digest(paths[1])
    _write(paths[2], review)
    with pytest.raises(ValueError, match="duplicate"):
        _run(paths)
    assert not paths[3].exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL inheritance regression")
def test_published_artifact_inherits_windows_permissions(tmp_path: Path) -> None:
    paths = _setup(tmp_path)
    _run(paths)
    for path in (paths[3], paths[3] / "split-manifest.json"):
        escaped = str(path).replace("'", "''")
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"(Get-Acl -LiteralPath '{escaped}').AreAccessRulesProtected",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout.strip() == "False"
