"""Read-only numeric or contract review candidates in local SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
)
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from baymax.models.sft_contract_audit import find_contract_findings
from baymax.models.sft_dataset import SFTExample, SFTSplitManifest
from baymax.models.sft_validation import validate_sft_target_contract

NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w]|\.\d)"
)
ROUNDING_WORD = re.compile(r"\b(round(?:ing)?|truncat\w*|floor|ceil(?:ing)?|nearest)\b", re.I)
NO_ROUNDING = re.compile(
    r"\b(?:do not|don't|never|without|no)\s+(?:round\w*|truncat\w*|floor|ceil\w*)\b", re.I
)
CONVERSIONS = {
    "truncate": ROUND_DOWN,
    "floor": ROUND_FLOOR,
    "ceiling": ROUND_CEILING,
    "round_half_even": ROUND_HALF_EVEN,
    "round_half_up": ROUND_HALF_UP,
}


def find_numeric_mismatches(
    request: str, tools: list[dict[str, Any]], calls: list[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    """Flag scalar integer contracts and possible lossy conversions, not semantic errors."""

    numeric_literals = [(m, Decimal(m.group().replace(",", ""))) for m in NUMBER.finditer(request)]
    rounding = (
        "prohibited"
        if NO_ROUNDING.search(request)
        else "mentioned"
        if ROUNDING_WORD.search(request)
        else "not_mentioned"
    )
    schemas = {tool["name"]: tool["parameters"] for tool in tools}
    inspected = 0
    findings: list[dict[str, Any]] = []
    for call_index, call in enumerate(calls):
        name = call["name"]
        if name not in schemas:
            raise ValueError(f"target tool {name!r} has no supplied schema")
        parameters = schemas[name]
        properties = (
            parameters.get("properties", {}) if parameters.get("type") == "object" else parameters
        )
        arguments = call["arguments"]
        if not isinstance(properties, dict) or not isinstance(arguments, dict):
            raise ValueError(f"{name}: expected parameter and argument objects")
        for argument, value in arguments.items():
            schema = properties.get(argument, {})
            declared_type = schema.get("type", "") if isinstance(schema, dict) else ""
            if not isinstance(declared_type, str) or declared_type.split(",")[0].strip() not in (
                "int",
                "integer",
            ):
                continue
            inspected += 1
            target = Decimal(str(value)) if type(value) in (int, float) else None
            evidence = {
                "tool": name,
                "argument": argument,
                "call_index": call_index,
                "argument_schema": schema,
                "target_value": value,
            }
            if target is None or not target.is_finite() or target != target.to_integral_value():
                findings.append({**evidence, "issue": "integer_value_contract_mismatch"})
                continue
            matches = []
            for match, number in numeric_literals:
                if number == number.to_integral_value():
                    continue
                methods = [
                    label
                    for label, mode in CONVERSIONS.items()
                    if number.to_integral_value(rounding=mode) == target
                ]
                if methods:
                    matches.append(
                        {
                            "literal": match.group(),
                            "request_span": [match.start(), match.end()],
                            "possible_conversions": methods,
                        }
                    )
            if matches:
                also_explicit = any(number == target for _, number in numeric_literals)
                findings.append(
                    {
                        **evidence,
                        "issue": "possible_decimal_to_integer_conversion",
                        "matched_literals": matches,
                        "rounding_language": rounding,
                        "target_also_explicit_in_request": also_explicit,
                        "review_priority": (
                            "low" if rounding == "mentioned" or also_explicit else "medium"
                        ),
                    }
                )
    return inspected, findings


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _payload(line: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], str]:
    try:
        example = SFTExample.model_validate_json(line)
    except ValidationError as error:
        details = [e["msg"] for e in error.errors(include_input=False)]
        raise ValueError(f"invalid SFT shape: {details}") from error
    if [m.role for m in example.messages] != ["system", "user", "assistant"]:
        raise ValueError("expected system/user/assistant messages")
    header, separator, context = example.messages[1].content.partition("\n\nUser request:\n")
    if not separator or "\n" not in header:
        raise ValueError("missing tool list or structured request wrapper")
    tools = json.loads(header.split("\n", 1)[1], parse_constant=_reject_constant)
    wrapper = json.loads(context, parse_constant=_reject_constant)
    calls = json.loads(example.messages[2].content, parse_constant=_reject_constant)
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("request"), str):
        raise ValueError("request wrapper must contain a request string")
    if not isinstance(tools, list) or not isinstance(calls, list):
        raise ValueError("tools and targets must be JSON arrays")
    if not all(isinstance(item, dict) for item in [*tools, *calls]):
        raise ValueError("tools and targets must contain objects")
    canonical = json.dumps(
        json.loads(line), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return wrapper["request"], tools, calls, hashlib.sha256(canonical.encode()).hexdigest()


def audit_dataset(
    dataset_dir: Path,
    *,
    max_rows: int | None = None,
    checks: Literal["numeric", "contracts", "targets"] = "numeric",
) -> dict[str, Any]:
    """Stream saved splits with manifest provenance; never modify the source files."""

    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    if checks not in ("numeric", "contracts", "targets"):
        raise ValueError("checks must be numeric, contracts or targets")
    manifest_path = dataset_dir / "split-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = SFTSplitManifest.model_validate_json(manifest_bytes)
    scanned = inspected = 0
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    split_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        entry = manifest.splits[split]
        if entry.count != len(entry.source_rows):
            raise ValueError(f"{split}: manifest count and source references disagree")
        if max_rows is not None and scanned >= max_rows:
            break
        path = dataset_dir / f"{split}.jsonl"
        line_count = 0
        file_hash = hashlib.sha256()
        with path.open(encoding="utf-8", newline="") as source:
            for line_count, line in enumerate(source, 1):
                if line_count > entry.count:
                    raise ValueError(f"{path}:{line_count}: row has no manifest source reference")
                ref = entry.source_rows[line_count - 1]
                location = {
                    "split": split,
                    "path": str(path),
                    "line": line_count,
                    "source_id": ref.source_row_id,
                    "source_index": ref.source_index,
                }
                scanned += 1
                file_hash.update(line.encode("utf-8"))
                try:
                    request, tools, calls, digest = _payload(line)
                    if checks == "targets":
                        target_issues = validate_sft_target_contract(
                            SFTExample.model_validate_json(line), location=f"{path}:{line_count}"
                        )
                        if target_issues:
                            findings.append(
                                {
                                    **location,
                                    "example_sha256": digest,
                                    "issue": "target_contract_incompatible",
                                    "tool": "",
                                    "argument": "",
                                    "disposition": "proposed_exclusion",
                                    "details": [issue.message for issue in target_issues],
                                }
                            )
                    count, row_findings = (
                        find_numeric_mismatches(request, tools, calls)
                        if checks == "numeric"
                        else find_contract_findings(tools, calls)
                    )
                    if checks == "targets":
                        row_findings = [
                            finding
                            for finding in row_findings
                            if finding["issue"] == "credential_emitting_target"
                        ]
                    inspected += count
                    findings.extend(
                        {
                            **location,
                            "example_sha256": digest,
                            **({"request": request} if checks == "numeric" else {}),
                            **finding,
                        }
                        for finding in row_findings
                    )
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    errors.append(
                        {
                            **location,
                            "error": str(error) if checks == "numeric" else type(error).__name__,
                        }
                    )
                if max_rows is not None and scanned >= max_rows:
                    break
        split_hashes[split] = file_hash.hexdigest()
        if not (max_rows is not None and scanned >= max_rows) and line_count != entry.count:
            raise ValueError(f"{path}: file count {line_count} differs from manifest {entry.count}")
    groups = Counter((f["issue"], f["tool"], f["argument"]) for f in findings)
    report: dict[str, Any] = {
        "audit_version": 1,
        "checks": checks,
        "dataset_id": manifest.dataset_id,
        "revision": manifest.revision,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "max_rows": max_rows,
        "partial": (max_rows is not None and scanned >= max_rows)
        or scanned < sum(e.count for e in manifest.splits.values()),
        "rows_scanned": scanned,
        "integer_arguments_inspected" if checks == "numeric" else "arguments_inspected": inspected,
        "flagged_rows": len({(f["split"], f["line"]) for f in findings}),
        "groups": [
            {"issue": key[0], "tool": key[1], "argument": key[2], "findings": count}
            for key, count in sorted(groups.items())
        ],
        "findings": findings,
        "row_parse_errors": errors,
        "limitations": (
            "Only supplied scalar int/integer arguments are checked. Conversion candidates "
            "do not establish which request number belongs to an argument. Rounding language "
            "is a review hint, not permission. Calculations, unit conversions, identifiers, "
            "nested values, other types, and missing arguments are not semantically validated. "
            "A capped scan uses a train/validation/test prefix, not a representative sample. "
            "Nothing is automatically excluded or corrected."
        ),
    }
    if checks in ("contracts", "targets"):
        exclusions: dict[tuple[str, int], dict[str, Any]] = {}
        for finding in findings:
            if finding.get("disposition") != "proposed_exclusion":
                continue
            key = (finding["split"], finding["line"])
            if key not in exclusions:
                exclusions[key] = {
                    field: finding[field]
                    for field in (
                        "split",
                        "path",
                        "line",
                        "source_id",
                        "source_index",
                        "example_sha256",
                    )
                }
                exclusions[key]["reasons"] = []
            if finding["issue"] not in exclusions[key]["reasons"]:
                exclusions[key]["reasons"].append(finding["issue"])
        report["credential_policy"] = "adapter-managed-auth-v1"
        report["proposed_exclusions"] = list(exclusions.values())
        report["limitations"] = (
            "Read-only proposals, not approved removals. Basic declared types/defaults, literal "
            "prose defaults and non-string argument expression candidates only; not full JSON "
            "Schema or semantic validation. Unknown types, null ambiguity and generic token/key "
            "fields require review. Nested credential leaves are checked, but nested JSON Schema "
            "constraints, invented input, consent, and all dependency formats are not validated. "
            "Contract findings omit requests, schemas, defaults and argument values. Prose-default "
            "matching is limited to explicit literal 'default is/to' phrases. Capped coverage is "
            "a prefix, not a representative sample. No source data is changed."
        )
        report["limitations"] += (
            " Duplicate-name target schemas are ambiguous: argument type checks for those "
            "calls are skipped, but explicit credential keys remain checked. Unknown tools "
            "are also reported without guessing their argument contracts."
        )
        if checks == "targets":
            report["target_policy"] = "sft-target-compatible-v1"
            report["split_sha256"] = split_hashes
            report["limitations"] = (
                "Uses the training preflight target validator plus explicit credential checks. "
                "Target incompatibilities and unverifiable contracts are proposals, not proof "
                "that upstream labels are wrong. Default-only metadata findings do not propose "
                "exclusions. No coercion, schema guessing, target repair or semantic review. "
                "Row-processing errors require separate review, even if target findings were "
                "recorded before the credential inspection failed. Reports omit raw values. "
                "Capped coverage is a prefix, not a representative sample. Nothing is removed."
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None, help="Total cap across saved splits.")
    parser.add_argument("--checks", choices=("numeric", "contracts", "targets"), default="numeric")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve().is_relative_to(args.dataset_dir.resolve()):
            raise ValueError("write the report outside the input dataset directory")
        if args.output.exists():
            raise ValueError("output already exists; choose a new report path")
        report = audit_dataset(args.dataset_dir, max_rows=args.max_rows, checks=args.checks)
        serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(serialized)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {args.checks} review candidates to {args.output}")
    print(f"Rows scanned: {report['rows_scanned']}; flagged rows: {report['flagged_rows']}")
    print(f"Row parse errors: {len(report['row_parse_errors'])}; partial: {report['partial']}")
    if args.checks == "contracts":
        print(f"Proposed credential exclusions: {len(report['proposed_exclusions'])}")
    elif args.checks == "targets":
        print(f"Proposed target/credential exclusions: {len(report['proposed_exclusions'])}")
    print("Candidates require review; no source data was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
