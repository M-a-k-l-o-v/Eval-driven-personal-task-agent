"""Analyze Phase 2 SFT behavior mix and eval failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baymax.eval.output_diagnostics import classify_boundary_generation
from baymax.eval.scenario_loader import (
    ClarificationExpectedBehavior,
    RefusalExpectedBehavior,
    ToolCallsExpectedBehavior,
    load_scenarios,
)
from baymax.models.sft_training import _diagnose_sft_tokenization_row, load_sft_jsonl

BehaviorClass = str
FailureClass = str

DEFAULT_RESULTS = [
    "qwen-base=results/phase1/qwen-base/full.json",
    "v8=results/phase2/qwen-lora/baymax-mix/synthetic-v8-meta-tools-weighted-v1-50.json",
    "aligned=results/phase2/qwen-lora/baymax-mix/format-aligned-reference-v1-50.json",
]


@dataclass(frozen=True)
class TokenDistributionRow:
    behavior_class: BehaviorClass
    row_count: int
    trainable_tokens: int


@dataclass(frozen=True)
class FailureByBehaviorRow:
    run_label: str
    expected_behavior: str
    scenario_count: int
    failed_scenarios: int
    failure_counts: dict[str, int]


@dataclass(frozen=True)
class BoundaryFailureRow:
    run_label: str
    scenario_id: str
    expected_behavior: str
    failure_class: FailureClass
    failure_reasons: list[str]
    raw_model_output: str | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True, help="SFT JSONL split to analyze.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("scenarios/v1"),
        help="Scenario fixture directory.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=None,
        help="Eval result path or label=path. Defaults to base/v8/aligned.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tokenizer model for trainable-token counts.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Cap SFT rows for quick checks.")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=1024,
        help="Training sequence length used for label-token diagnostics.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("error: missing transformers. Install training dependencies first.", file=sys.stderr)
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        scenarios = load_scenarios(args.scenarios)
        sft_rows = load_sft_jsonl(args.sft, max_rows=args.max_rows)
        result_args = args.result or DEFAULT_RESULTS
        result_payloads = [_load_result_arg(value) for value in result_args]
        report = build_behavior_analysis_report(
            sft_rows=sft_rows,
            tokenizer=tokenizer,
            max_sequence_length=args.max_sequence_length,
            scenarios=scenarios,
            result_payloads=result_payloads,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(format_behavior_analysis_report(report))
    return 0


def build_behavior_analysis_report(
    *,
    sft_rows: list[dict[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
    scenarios: list[Any],
    result_payloads: list[tuple[str, Path, dict[str, Any]]],
) -> dict[str, Any]:
    expected_behavior_by_id = {
        scenario.id: _expected_behavior_type(scenario.expected_behavior) for scenario in scenarios
    }

    return {
        "trainable_token_distribution": [
            row.__dict__
            for row in analyze_trainable_tokens_by_behavior(
                rows=sft_rows,
                tokenizer=tokenizer,
                max_sequence_length=max_sequence_length,
            )
        ],
        "failure_counts_by_expected_behavior": [
            row.__dict__
            for row in analyze_failures_by_expected_behavior(
                result_payloads=result_payloads,
                expected_behavior_by_id=expected_behavior_by_id,
            )
        ],
        "boundary_failure_generations": [
            row.__dict__
            for row in classify_boundary_failures(
                result_payloads=result_payloads,
                expected_behavior_by_id=expected_behavior_by_id,
            )
        ],
    }


def analyze_trainable_tokens_by_behavior(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
) -> list[TokenDistributionRow]:
    row_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        behavior_class = classify_sft_behavior(row)
        diagnostic = _diagnose_sft_tokenization_row(
            row=row,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            row_index=row_index,
        )
        row_counts[behavior_class] += 1
        token_counts[behavior_class] += diagnostic.trainable_label_tokens

    return [
        TokenDistributionRow(
            behavior_class=behavior_class,
            row_count=row_counts[behavior_class],
            trainable_tokens=token_counts[behavior_class],
        )
        for behavior_class in sorted(row_counts)
    ]


def classify_sft_behavior(row: dict[str, Any]) -> BehaviorClass:
    payload = _assistant_call_payload(row)
    if payload is None:
        return "malformed"
    if len(payload) > 1:
        return "multi_tool_sequence"

    name = str(payload[0].get("name", ""))
    if name == "request_clarification":
        return "request_clarification"
    if name == "refuse_request":
        return "refuse_request"
    return "direct_tool_call"


def analyze_failures_by_expected_behavior(
    *,
    result_payloads: list[tuple[str, Path, dict[str, Any]]],
    expected_behavior_by_id: dict[str, str],
) -> list[FailureByBehaviorRow]:
    rows: list[FailureByBehaviorRow] = []
    for run_label, _path, payload in result_payloads:
        grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in _scenario_results(payload):
            scenario_id = str(result.get("scenario_id", ""))
            expected_behavior = expected_behavior_by_id.get(scenario_id, "unknown")
            grouped_results[expected_behavior].append(result)

        for expected_behavior in sorted(grouped_results):
            results = grouped_results[expected_behavior]
            failure_counts: Counter[str] = Counter()
            failed_scenarios = 0
            for result in results:
                reasons = _failure_reasons(result)
                if reasons:
                    failed_scenarios += 1
                failure_counts.update(reasons)

            rows.append(
                FailureByBehaviorRow(
                    run_label=run_label,
                    expected_behavior=expected_behavior,
                    scenario_count=len(results),
                    failed_scenarios=failed_scenarios,
                    failure_counts=dict(sorted(failure_counts.items())),
                )
            )

    return rows


def classify_boundary_failures(
    *,
    result_payloads: list[tuple[str, Path, dict[str, Any]]],
    expected_behavior_by_id: dict[str, str],
) -> list[BoundaryFailureRow]:
    rows: list[BoundaryFailureRow] = []
    for run_label, _path, payload in result_payloads:
        for result in _scenario_results(payload):
            scenario_id = str(result.get("scenario_id", ""))
            expected_behavior = expected_behavior_by_id.get(scenario_id, "unknown")
            if expected_behavior not in {"clarification", "refusal"}:
                continue

            failure_reasons = _failure_reasons(result)
            if not failure_reasons:
                continue

            raw_model_output = result.get("raw_model_output")
            rows.append(
                BoundaryFailureRow(
                    run_label=run_label,
                    scenario_id=scenario_id,
                    expected_behavior=expected_behavior,
                    failure_class=classify_boundary_generation(
                        raw_model_output=raw_model_output,
                        expected_behavior=expected_behavior,
                    ),
                    failure_reasons=failure_reasons,
                    raw_model_output=raw_model_output
                    if isinstance(raw_model_output, str)
                    else None,
                )
            )

    return rows


def format_behavior_analysis_report(report: dict[str, Any]) -> str:
    sections = [
        "## Trainable Tokens by Behavior",
        _format_table(
            ["behavior_class", "row_count", "trainable_tokens"],
            [
                [
                    row["behavior_class"],
                    str(row["row_count"]),
                    str(row["trainable_tokens"]),
                ]
                for row in report["trainable_token_distribution"]
            ],
        ),
        "",
        "## Failure Counts by Expected Behavior",
        _format_table(
            ["run", "expected_behavior", "scenarios", "failed", "failure_counts"],
            [
                [
                    row["run_label"],
                    row["expected_behavior"],
                    str(row["scenario_count"]),
                    str(row["failed_scenarios"]),
                    _format_failure_counts(row["failure_counts"]),
                ]
                for row in report["failure_counts_by_expected_behavior"]
            ],
        ),
        "",
        "## Clarification/Refusal Failure Generations",
        _format_table(
            ["run", "scenario_id", "expected_behavior", "failure_class", "failure_reasons"],
            [
                [
                    row["run_label"],
                    row["scenario_id"],
                    row["expected_behavior"],
                    row["failure_class"],
                    ", ".join(row["failure_reasons"]),
                ]
                for row in report["boundary_failure_generations"]
            ],
        ),
    ]
    return "\n".join(sections)


def _assistant_call_payload(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    assistant_message = messages[-1]
    if not isinstance(assistant_message, dict):
        return None
    content = assistant_message.get("content")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(call, dict) for call in payload):
        return None
    return payload


def _expected_behavior_type(expected_behavior: object) -> str:
    if isinstance(expected_behavior, ClarificationExpectedBehavior):
        return "clarification"
    if isinstance(expected_behavior, RefusalExpectedBehavior):
        return "refusal"
    if isinstance(expected_behavior, ToolCallsExpectedBehavior):
        return "tool_calls"
    return "tool_call"


def _load_result_arg(value: str) -> tuple[str, Path, dict[str, Any]]:
    label, path = _parse_result_arg(value)
    with path.open(encoding="utf-8") as result_file:
        payload: dict[str, Any] = json.load(result_file)
    return label, path, payload


def _parse_result_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label:
            raise ValueError(f"missing label in result argument: {value}")
        return label, Path(raw_path)
    path = Path(value)
    return path.stem, path


def _scenario_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_results = payload.get("scenario_results")
    if not isinstance(scenario_results, list):
        raise ValueError("result payload does not contain scenario_results.")
    return [result for result in scenario_results if isinstance(result, dict)]


def _failure_reasons(result: dict[str, Any]) -> list[str]:
    score = result.get("score")
    if not isinstance(score, dict):
        return []
    failure_reasons = score.get("failure_reasons")
    if not isinstance(failure_reasons, list):
        return []
    return [str(reason) for reason in failure_reasons]


def _format_failure_counts(failure_counts: dict[str, int]) -> str:
    if not failure_counts:
        return "-"
    return ", ".join(f"{reason}:{count}" for reason, count in sorted(failure_counts.items()))


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "No rows."
    widths = [max(len(row[index]) for row in [headers, *rows]) for index in range(len(headers))]
    lines = [
        "| " + " | ".join(_pad(value, widths[index]) for index, value in enumerate(headers)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_pad(value, widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _pad(value: str, width: int) -> str:
    return value + (" " * (width - len(value)))


if __name__ == "__main__":
    raise SystemExit(main())
