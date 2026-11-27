"""Validation helpers for generated Phase 2 SFT datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, ValidationError

from baymax.core.meta_tools import META_TOOL_NAMES, meta_tool_specs
from baymax.models.sft_contract_audit import matches_declared_type
from baymax.models.sft_dataset import SFTExample, SFTSplitManifest, SplitName

IssueSeverity = Literal["error", "warning"]
EXPECTED_MESSAGE_ROLES = ["system", "user", "assistant"]
EXPECTED_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


class SFTValidationIssue(BaseModel):
    """One validation issue in an SFT dataset."""

    model_config = ConfigDict(extra="forbid")

    severity: IssueSeverity
    location: str
    message: str


class SFTSplitValidationReport(BaseModel):
    """Validation result for one JSONL split file."""

    model_config = ConfigDict(extra="forbid")

    name: SplitName
    path: str
    manifest_count: int | None
    actual_count: int
    valid_json_lines: int
    invalid_json_lines: int
    issues: list[SFTValidationIssue]


class SFTDatasetValidationReport(BaseModel):
    """Validation result for a generated SFT dataset directory."""

    model_config = ConfigDict(extra="forbid")

    dataset_dir: str
    manifest_path: str
    valid: bool
    manifest_loaded: bool
    dataset_id: str | None = None
    revision: str | None = None
    config_hash: str | None = None
    converted_rows: int | None = None
    loaded_rows: int | None = None
    skipped_row_count: int | None = None
    splits: dict[SplitName, SFTSplitValidationReport]
    issues: list[SFTValidationIssue]


def validate_sft_dataset(dataset_dir: Path) -> SFTDatasetValidationReport:
    """Validate generated SFT JSONL files and their split manifest."""

    manifest_path = dataset_dir / "split-manifest.json"
    issues: list[SFTValidationIssue] = []
    split_reports: dict[SplitName, SFTSplitValidationReport] = {}
    seen_content: dict[str, tuple[SplitName, str]] = {}

    manifest = _load_manifest(manifest_path, issues)
    if manifest is None:
        for split_name in EXPECTED_SPLITS:
            split_reports[split_name] = _validate_split_jsonl(
                split_name=split_name,
                path=dataset_dir / f"{split_name}.jsonl",
                manifest_count=None,
                seen_content=seen_content,
            )

        return _report(
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            manifest=None,
            split_reports=split_reports,
            issues=issues,
        )

    issues.extend(_validate_manifest_metadata(manifest))
    issues.extend(_validate_manifest_counts(manifest))
    issues.extend(_validate_manifest_split_overlap(manifest))

    for split_name in EXPECTED_SPLITS:
        split = manifest.splits.get(split_name)
        if split is None:
            issues.append(
                SFTValidationIssue(
                    severity="error",
                    location="split-manifest.json",
                    message=f"missing split metadata for {split_name}",
                )
            )
            split_reports[split_name] = _validate_split_jsonl(
                split_name=split_name,
                path=dataset_dir / f"{split_name}.jsonl",
                manifest_count=None,
                seen_content=seen_content,
            )
            continue

        split_reports[split_name] = _validate_split_jsonl(
            split_name=split_name,
            path=_resolve_split_path(dataset_dir, split.path),
            manifest_count=split.count,
            seen_content=seen_content,
        )

    return _report(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        split_reports=split_reports,
        issues=issues,
    )


def _load_manifest(
    manifest_path: Path,
    issues: list[SFTValidationIssue],
) -> SFTSplitManifest | None:
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location=str(manifest_path),
                message="split manifest not found",
            )
        )
        return None

    try:
        return SFTSplitManifest.model_validate_json(manifest_text)
    except ValidationError as error:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location=str(manifest_path),
                message=f"invalid split manifest: {error}",
            )
        )
        return None


def _validate_manifest_metadata(manifest: SFTSplitManifest) -> list[SFTValidationIssue]:
    issues: list[SFTValidationIssue] = []
    if not manifest.config_hash:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location="split-manifest.json:config_hash",
                message="config_hash is required",
            )
        )
    if not manifest.revision:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location="split-manifest.json:revision",
                message="dataset revision must be pinned",
            )
        )
    if len(manifest.skipped_rows) > 0:
        issues.append(
            SFTValidationIssue(
                severity="warning",
                location="split-manifest.json:skipped_rows",
                message=f"{len(manifest.skipped_rows)} source rows were skipped during conversion",
            )
        )

    return issues


def _validate_manifest_counts(manifest: SFTSplitManifest) -> list[SFTValidationIssue]:
    split_count_total = sum(split.count for split in manifest.splits.values())
    expected_converted_rows = manifest.loaded_rows - len(manifest.skipped_rows)
    issues: list[SFTValidationIssue] = []

    if split_count_total != manifest.converted_rows:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location="split-manifest.json:splits",
                message=(
                    f"split counts total {split_count_total}, "
                    f"but converted_rows is {manifest.converted_rows}"
                ),
            )
        )
    if expected_converted_rows != manifest.converted_rows:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location="split-manifest.json:converted_rows",
                message=(
                    f"loaded_rows - skipped_rows is {expected_converted_rows}, "
                    f"but converted_rows is {manifest.converted_rows}"
                ),
            )
        )

    return issues


def _validate_manifest_split_overlap(manifest: SFTSplitManifest) -> list[SFTValidationIssue]:
    seen_source_rows: dict[tuple[int, int | str | None], SplitName] = {}
    issues: list[SFTValidationIssue] = []

    for split_name, split in manifest.splits.items():
        for source_row in split.source_rows:
            key = (source_row.source_index, source_row.source_row_id)
            previous_split = seen_source_rows.get(key)
            if previous_split is not None:
                issues.append(
                    SFTValidationIssue(
                        severity="error",
                        location="split-manifest.json:splits",
                        message=(
                            f"source row {key} appears in both {previous_split} and {split_name}"
                        ),
                    )
                )
            seen_source_rows[key] = split_name

    return issues


def _validate_split_jsonl(
    *,
    split_name: SplitName,
    path: Path,
    manifest_count: int | None,
    seen_content: dict[str, tuple[SplitName, str]],
) -> SFTSplitValidationReport:
    issues: list[SFTValidationIssue] = []
    actual_count = 0
    valid_json_lines = 0
    invalid_json_lines = 0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location=str(path),
                message="split JSONL file not found",
            )
        )
        lines = []

    for line_number, line in enumerate(lines, start=1):
        actual_count += 1
        line_location = f"{path}:{line_number}"
        if not line.strip():
            invalid_json_lines += 1
            issues.append(
                SFTValidationIssue(
                    severity="error",
                    location=line_location,
                    message="blank JSONL line",
                )
            )
            continue

        try:
            raw_example = json.loads(line)
            valid_json_lines += 1
        except json.JSONDecodeError as error:
            invalid_json_lines += 1
            issues.append(
                SFTValidationIssue(
                    severity="error",
                    location=line_location,
                    message=f"invalid JSON at char {error.pos}: {error.msg}",
                )
            )
            continue

        issues.extend(_validate_example_shape(raw_example, location=line_location))
        content_hash = hashlib.sha256(
            json.dumps(raw_example, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        previous = seen_content.setdefault(content_hash, (split_name, line_location))
        if previous[0] != split_name:
            issues.append(
                SFTValidationIssue(
                    severity="error",
                    location=line_location,
                    message=f"identical SFT content also appears in {previous[0]} at {previous[1]}",
                )
            )

    if manifest_count is not None and actual_count != manifest_count:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location=str(path),
                message=f"manifest count is {manifest_count}, but JSONL has {actual_count} lines",
            )
        )

    return SFTSplitValidationReport(
        name=split_name,
        path=str(path),
        manifest_count=manifest_count,
        actual_count=actual_count,
        valid_json_lines=valid_json_lines,
        invalid_json_lines=invalid_json_lines,
        issues=issues,
    )


def _validate_example_shape(raw_example: Any, *, location: str) -> list[SFTValidationIssue]:
    try:
        example = SFTExample.model_validate(raw_example)
    except ValidationError:
        return [
            SFTValidationIssue(
                severity="error",
                location=location,
                message="invalid SFT example shape: check message roles and content types",
            )
        ]

    issues: list[SFTValidationIssue] = []
    roles = [message.role for message in example.messages]
    if roles != EXPECTED_MESSAGE_ROLES:
        issues.append(
            SFTValidationIssue(
                severity="error",
                location=location,
                message=f"messages must have roles {EXPECTED_MESSAGE_ROLES}, got {roles}",
            )
        )

    if example.messages:
        assistant_message = example.messages[-1]
        try:
            assistant_payload = json.loads(assistant_message.content)
        except json.JSONDecodeError as error:
            issues.append(
                SFTValidationIssue(
                    severity="error",
                    location=location,
                    message=f"assistant content is invalid JSON at char {error.pos}: {error.msg}",
                )
            )
        else:
            if not isinstance(assistant_payload, list):
                issues.append(
                    SFTValidationIssue(
                        severity="error",
                        location=location,
                        message="assistant content must decode to a JSON array",
                    )
                )

    if roles == EXPECTED_MESSAGE_ROLES:
        issues.extend(validate_sft_target_contract(example, location=location))
    return issues


def _reject_constant(value: str) -> None:
    raise ValueError("Non-standard JSON numeric constant.")


def validate_sft_target_contract(example: SFTExample, *, location: str) -> list[SFTValidationIssue]:
    """Validate targets against supplied schemas; never print argument values."""
    issues: list[SFTValidationIssue] = []

    def error(message: str) -> None:
        issues.append(SFTValidationIssue(severity="error", location=location, message=message))

    if [m.role for m in example.messages] != EXPECTED_MESSAGE_ROLES:
        error("Target contract requires system/user/assistant messages.")
        return issues
    header, separator, context = example.messages[1].content.partition("\n\nUser request:\n")
    try:
        if not separator or "\n" not in header:
            raise ValueError("Missing tool list or request wrapper.")
        tools = json.loads(header.split("\n", 1)[1], parse_constant=_reject_constant)
        wrapper = json.loads(context, parse_constant=_reject_constant)
        calls = json.loads(example.messages[2].content, parse_constant=_reject_constant)
        if not isinstance(wrapper, dict) or not isinstance(wrapper.get("request"), str):
            raise ValueError("Invalid request wrapper.")
        if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
            raise ValueError("Invalid tool list.")
        if not isinstance(calls, list) or not calls or not all(isinstance(c, dict) for c in calls):
            raise ValueError("Targets must be a nonempty array of calls.")
    except (ValueError, TypeError):
        error("Cannot parse tool list, request wrapper, or target calls.")
        return issues

    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name, parameters = tool.get("name"), tool.get("parameters")
        if not isinstance(name, str) or not name.strip() or not isinstance(parameters, dict):
            error("Tool definition needs a name and parameter schema.")
        elif name in schemas:
            error("Duplicate tool definitions make the target ambiguous.")
        else:
            schemas[name] = parameters
    canonical_meta = {t["name"]: t["parameters"] for t in meta_tool_specs()}
    for name in META_TOOL_NAMES:
        if name not in schemas:
            error("Shared meta-tool definition is missing from the supplied tool list.")
        elif schemas[name] != canonical_meta[name]:
            error("Meta-tool definition differs from the shared contract.")

    for index, call in enumerate(calls):
        name, arguments = call.get("name"), call.get("arguments")
        prefix = f"Call {index}: "
        if (
            set(call) != {"name", "arguments"}
            or not isinstance(name, str)
            or not isinstance(arguments, dict)
        ):
            error(prefix + "requires exactly name and an arguments object.")
            continue
        if name not in schemas:
            error(prefix + "tool is not in the supplied tool list.")
            continue
        if name in META_TOOL_NAMES:
            field = "question" if name == "request_clarification" else "reason"
            if (
                len(calls) != 1
                or set(arguments) != {field}
                or not isinstance(arguments.get(field), str)
                or not arguments[field].strip()
            ):
                error(prefix + "meta-tool must be exclusive with its required nonempty text field.")
            continue
        schema = schemas[name]
        if _has_reference(schema):
            error(prefix + "schema references are unsupported; no external retrieval is allowed.")
            continue
        if schema.get("type") == "object":
            try:
                Draft202012Validator.check_schema(schema)
                validator = Draft202012Validator(
                    schema, format_checker=Draft202012Validator.FORMAT_CHECKER
                )
                for failure in validator.iter_errors(arguments):
                    error(prefix + f"argument schema violation ({failure.validator}).")
            except SchemaError:
                error(prefix + "parameter schema is not valid JSON Schema.")
        else:
            # xLAM also uses a parameter-name map with Python-style type expressions.
            for key, definition in schema.items():
                if not isinstance(definition, dict):
                    error(prefix + "parameter definition must be an object.")
                elif "required" in definition and not isinstance(definition["required"], bool):
                    error(prefix + "native parameter requiredness must be boolean.")
                elif definition.get("required") is True and key not in arguments:
                    error(prefix + "an explicitly required parameter is missing.")
            for key, value in arguments.items():
                definition = schema.get(key)
                if not isinstance(definition, dict):
                    error(prefix + "argument is not declared by the tool.")
                    continue
                match = matches_declared_type(value, definition.get("type"))
                if match is not True:
                    error(
                        prefix
                        + (
                            "argument type mismatch."
                            if match is False
                            else "argument type cannot be verified."
                        )
                    )
                constraints = {
                    key: value
                    for key, value in definition.items()
                    if key not in {"type", "description", "required", "default"}
                }
                try:
                    Draft202012Validator.check_schema(constraints)
                    validator = Draft202012Validator(
                        constraints, format_checker=Draft202012Validator.FORMAT_CHECKER
                    )
                    for failure in validator.iter_errors(value):
                        error(prefix + f"argument schema violation ({failure.validator}).")
                except SchemaError:
                    error(prefix + "parameter constraints cannot be verified.")
    return issues


def _has_reference(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in value for key in ("$ref", "$dynamicRef", "$recursiveRef")) or any(
            _has_reference(child) for child in value.values()
        )
    return isinstance(value, list) and any(_has_reference(child) for child in value)


def validate_training_targets(rows: list[dict[str, Any]], *, path: Path) -> None:
    """Fail before loading a model if any selected SFT target violates its contract."""
    issues = [
        issue
        for index, row in enumerate(rows, 1)
        for issue in _validate_example_shape(row, location=f"{path}:{index}")
    ]
    if issues:
        details = "\n".join(f"- {issue.location}: {issue.message}" for issue in issues[:20])
        raise ValueError(f"Invalid SFT targets ({len(issues)} issues):\n{details}")


def _resolve_split_path(dataset_dir: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    if path.is_absolute() or path.exists():
        return path

    fallback_path = dataset_dir / path.name
    if fallback_path.exists():
        return fallback_path

    return path


def _report(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    manifest: SFTSplitManifest | None,
    split_reports: dict[SplitName, SFTSplitValidationReport],
    issues: list[SFTValidationIssue],
) -> SFTDatasetValidationReport:
    all_issues = issues + [
        issue for split_report in split_reports.values() for issue in split_report.issues
    ]

    return SFTDatasetValidationReport(
        dataset_dir=str(dataset_dir),
        manifest_path=str(manifest_path),
        valid=not any(issue.severity == "error" for issue in all_issues),
        manifest_loaded=manifest is not None,
        dataset_id=manifest.dataset_id if manifest is not None else None,
        revision=manifest.revision if manifest is not None else None,
        config_hash=manifest.config_hash if manifest is not None else None,
        converted_rows=manifest.converted_rows if manifest is not None else None,
        loaded_rows=manifest.loaded_rows if manifest is not None else None,
        skipped_row_count=len(manifest.skipped_rows) if manifest is not None else None,
        splits=split_reports,
        issues=issues,
    )
