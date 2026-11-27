"""Apply reviewed exclusions to a new artifact without rewriting retained SFT rows."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from baymax.models.sft_dataset import (
    SFTSkippedRow,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SplitName,
)

SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
POLICY = "adapter-managed-auth-v1"
TARGET_POLICY = "sft-target-compatible-v1"


class ReviewedRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: SplitName
    path: str
    line: int = Field(ge=1)
    source_id: int | str | None
    source_index: int
    example_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reasons: list[Literal["credential_emitting_target", "target_contract_incompatible"]] = Field(
        min_length=1
    )


class RetainedGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: SplitName
    line: int = Field(ge=1)
    source_id: int | str | None
    source_index: int
    example_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    error: str
    resolution: str = Field(min_length=1)


class SchemaReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    argument: str
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(ge=1)
    reason: str = Field(min_length=1)


class CredentialFilterReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: Literal["credential-filter-review-v1"]
    policy: Literal["adapter-managed-auth-v1"]
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_rows: list[ReviewedRow] = Field(min_length=1)
    schema_reviews: list[SchemaReview]
    retained_coverage_gaps: list[RetainedGap]


class TargetFilterReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: Literal["target-filter-review-v1"]
    policy: Literal["sft-target-compatible-v1"]
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_rows: list[ReviewedRow] = Field(min_length=1)
    schema_reviews: list[SchemaReview]
    retained_coverage_gaps: list[RetainedGap]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


@contextmanager
def _staging_directory(destination: Path) -> Iterator[Path]:
    # TemporaryDirectory uses owner-only ACLs on Windows, which survive publication.
    staging = destination.parent / f".{destination.name}-{uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
    finally:
        if staging.exists():
            if not staging.resolve().is_relative_to(destination.parent):
                raise ValueError("Staging directory escaped the output parent.")
            shutil.rmtree(staging)


def filter_sft_dataset(
    *, dataset_dir: Path, audit_path: Path, review_path: Path, output_dir: Path
) -> SFTSplitManifest:
    """Require pinned review evidence; publish only after all selected rows verify."""
    source_dir, destination = dataset_dir.resolve(), output_dir.resolve()
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new artifact version.")
    if destination.is_relative_to(source_dir) or source_dir.is_relative_to(destination):
        raise ValueError("Input and output directories must not overlap.")

    manifest_bytes = (source_dir / "split-manifest.json").read_bytes()
    manifest = SFTSplitManifest.model_validate_json(manifest_bytes)
    audit_bytes, review_bytes = audit_path.read_bytes(), review_path.read_bytes()
    audit = json.loads(audit_bytes)
    if not isinstance(audit, dict):
        raise ValueError("Audit must be a JSON object.")
    review = TypeAdapter(CredentialFilterReview | TargetFilterReview).validate_json(review_bytes)
    target_filter = isinstance(review, TargetFilterReview)
    policy = review.policy
    filter_version = 2 if target_filter else 1
    if review.audit_sha256 != _sha256(audit_bytes):
        raise ValueError("Audit hash does not match the reviewed audit.")
    if review.manifest_sha256 != _sha256(manifest_bytes):
        raise ValueError("Source manifest hash does not match the review.")
    if (
        audit.get("checks") != ("targets" if target_filter else "contracts")
        or audit.get("audit_version") != 1
        or audit.get("partial") is not False
        or audit.get("credential_policy") != POLICY
        or audit.get("manifest_sha256") != review.manifest_sha256
        or audit.get("dataset_id") != manifest.dataset_id
        or audit.get("revision") != manifest.revision
        or audit.get("rows_scanned") != manifest.converted_rows
        or target_filter
        and audit.get("target_policy") != TARGET_POLICY
    ):
        raise ValueError("Require a complete audit matching the source manifest and review policy.")
    if set(manifest.splits) != set(SPLITS) or (
        sum(s.count for s in manifest.splits.values()) != manifest.converted_rows
        or manifest.loaded_rows - len(manifest.skipped_rows) != manifest.converted_rows
    ):
        raise ValueError("Source manifest counts or splits are inconsistent.")
    if target_filter and (
        not isinstance(audit.get("split_sha256"), dict) or set(audit["split_sha256"]) != set(SPLITS)
    ):
        raise ValueError("Target audit must pin every source split file hash.")

    proposals = {
        (row.split, row.line): row
        for row in (ReviewedRow.model_validate(p) for p in audit["proposed_exclusions"])
    }
    if len(proposals) != len(audit["proposed_exclusions"]):
        raise ValueError("Audit contains duplicate exclusion locations.")
    if not target_filter and any(
        row.reasons != ["credential_emitting_target"] for row in proposals.values()
    ):
        raise ValueError("Credential-only reviews cannot approve target-contract exclusions.")
    exclusions = {(row.split, row.line): row for row in review.approved_rows}
    gaps = {(row.split, row.line): row for row in review.retained_coverage_gaps}
    if len(exclusions) != len(review.approved_rows) or len(gaps) != len(
        review.retained_coverage_gaps
    ):
        raise ValueError("Review contains duplicate row locations.")
    if any(proposals.get(key) != row for key, row in exclusions.items()):
        raise ValueError("An approved row is not an exact audit exclusion proposal.")
    errors = {(row["split"], row["line"]): row for row in audit["row_parse_errors"]}
    if len(errors) != len(audit["row_parse_errors"]):
        raise ValueError("Audit contains duplicate error locations.")
    if set(gaps) != set(errors) or set(gaps) & set(exclusions):
        raise ValueError("Every audit error must be accounted for separately as a retained gap.")
    for key, gap in gaps.items():
        error = errors[key]
        if any(
            error[field] != getattr(gap, field) for field in ("source_id", "source_index", "error")
        ):
            raise ValueError("Retained gap provenance does not match the audit error.")

    config_hash = _json_hash(
        {"filter_version": filter_version, "policy": policy, "review_sha256": _sha256(review_bytes)}
    )
    entries: dict[SplitName, SFTSplitManifestEntry] = {}
    skipped = list(manifest.skipped_rows)
    file_hashes: dict[str, dict[str, str]] = {}
    verified: set[tuple[SplitName, int]] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Only this newly created staging directory is eligible for automatic cleanup.
    with _staging_directory(destination) as staging:
        if not staging.is_relative_to(destination.parent):
            raise ValueError("Staging directory escaped the output parent.")
        for split_name in SPLITS:
            entry = manifest.splits[split_name]
            source_path = Path(entry.path)
            if not source_path.is_absolute() and not source_path.exists():
                source_path = source_dir / source_path.name
            source_path = source_path.resolve()
            if not source_path.is_relative_to(source_dir) or len(entry.source_rows) != entry.count:
                raise ValueError(f"{split_name}: source path or provenance count is invalid.")
            retained = []
            input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
            count = 0
            with (
                source_path.open("rb") as source,
                (staging / f"{split_name}.jsonl").open("xb") as out,
            ):
                for count, line in enumerate(source, 1):
                    if count > entry.count:
                        raise ValueError(f"{split_name}: row count exceeds the manifest.")
                    input_hash.update(line)
                    key = (split_name, count)
                    selected = exclusions.get(key) or gaps.get(key)
                    ref = entry.source_rows[count - 1]
                    if selected:
                        if (
                            ref.source_index != selected.source_index
                            or ref.source_row_id != selected.source_id
                            or _json_hash(json.loads(line)) != selected.example_sha256
                        ):
                            raise ValueError(
                                f"{split_name} line {count}: reviewed row has changed."
                            )
                        if (
                            isinstance(selected, ReviewedRow)
                            and Path(selected.path).resolve() != source_path
                        ):
                            raise ValueError(
                                f"{split_name} line {count}: reviewed path has changed."
                            )
                        verified.add(key)
                    if key in exclusions:
                        skipped.append(
                            SFTSkippedRow(
                                source_index=ref.source_index,
                                source_row_id=ref.source_row_id,
                                reason=(
                                    f"target-filter-v1:{policy}"
                                    if target_filter
                                    else f"credential-filter-v1:{policy}"
                                ),
                            )
                        )
                    else:
                        out.write(line)
                        output_hash.update(line)
                        retained.append(ref)
            if count != entry.count:
                raise ValueError(f"{split_name}: row count is below the manifest.")
            if target_filter and input_hash.hexdigest() != audit["split_sha256"][split_name]:
                raise ValueError(f"{split_name}: source file changed after the target audit.")
            entries[split_name] = SFTSplitManifestEntry(
                path=str(output_dir / f"{split_name}.jsonl"),
                count=len(retained),
                source_rows=retained,
            )
            file_hashes[split_name] = {
                "parent_sha256": input_hash.hexdigest(),
                "retained_sha256": output_hash.hexdigest(),
            }
        if verified != set(exclusions) | set(gaps):
            raise ValueError("Review references a row outside the source files.")
        derived = manifest.model_copy(
            update={
                "config_hash": config_hash,
                "revision": _json_hash({"config_hash": config_hash, "files": file_hashes}),
                "output_dir": str(output_dir),
                "splits": entries,
                "converted_rows": sum(entry.count for entry in entries.values()),
                "skipped_rows": skipped,
            }
        )
        (staging / "split-manifest.json").write_text(
            derived.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (staging / "filter-review.json").write_bytes(review_bytes)
        (staging / "filter-provenance.json").write_text(
            json.dumps(
                {
                    "filter_version": filter_version,
                    "policy": policy,
                    "parent_dataset_id": manifest.dataset_id,
                    "parent_revision": manifest.revision,
                    "parent_manifest_sha256": review.manifest_sha256,
                    "audit_sha256": review.audit_sha256,
                    "review_sha256": _sha256(review_bytes),
                    "excluded_rows": len(exclusions),
                    "retained_coverage_gap_rows": len(gaps),
                    "files": file_hashes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    return derived
