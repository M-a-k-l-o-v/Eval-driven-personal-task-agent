"""Summarize failure reasons from a BAYMAX eval result JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baymax.eval.output_diagnostics import classify_output_contract_failure


@dataclass(frozen=True)
class FailureSummaryRow:
    reason: str
    count: int
    scenario_ids: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Eval result JSON path.")
    parser.add_argument(
        "--output-diagnostics",
        action="store_true",
        help="Also classify rejected/unknown raw responses without changing their scores.",
    )
    args = parser.parse_args()

    try:
        rows = summarize_failures(args.result)
        diagnostics = summarize_output_diagnostics(args.result) if args.output_diagnostics else None
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(format_failure_summary(rows))
    if diagnostics is not None:
        print("\nOutput diagnostics only; official failure reasons and scores are unchanged.")
        print(format_failure_summary(diagnostics, column_name="output_diagnostic"))
    return 0


def summarize_failures(path: Path) -> list[FailureSummaryRow]:
    scenario_results = _load_scenario_results(path)

    scenarios_by_reason: dict[str, list[str]] = defaultdict(list)
    for result in scenario_results:
        if not isinstance(result, dict):
            continue

        scenario_id = str(result.get("scenario_id", "unknown"))
        score = result.get("score")
        if not isinstance(score, dict):
            continue

        failure_reasons = score.get("failure_reasons")
        if not isinstance(failure_reasons, list):
            continue

        for reason in failure_reasons:
            scenarios_by_reason[str(reason)].append(scenario_id)

    return _summary_rows(scenarios_by_reason)


def _load_scenario_results(path: Path) -> list[Any]:
    with path.open(encoding="utf-8") as result_file:
        data: dict[str, Any] = json.load(result_file)

    scenario_results = data.get("scenario_results")
    if not isinstance(scenario_results, list):
        raise ValueError(f"{path} does not contain scenario_results.")

    return scenario_results


def summarize_output_diagnostics(path: Path) -> list[FailureSummaryRow]:
    scenarios_by_diagnostic: dict[str, list[str]] = defaultdict(list)
    for result in _load_scenario_results(path):
        if not isinstance(result, dict):
            continue

        score = result.get("score")
        response = result.get("agent_response")
        if not isinstance(score, dict) or not isinstance(response, dict):
            continue

        failure_reasons = score.get("failure_reasons")
        rejected = response.get("parse_error") is not None or (
            isinstance(failure_reasons, list) and "invalid_model_output" in failure_reasons
        )
        unknown = "decision_type" in response and response["decision_type"] is None
        if not rejected and not unknown:
            continue

        diagnostic = (
            classify_output_contract_failure(result.get("raw_model_output"))
            or "recorded_rejection_or_unknown_not_reproduced"
        )
        scenarios_by_diagnostic[diagnostic].append(str(result.get("scenario_id", "unknown")))

    return _summary_rows(scenarios_by_diagnostic)


def _summary_rows(grouped: dict[str, list[str]]) -> list[FailureSummaryRow]:
    return [
        FailureSummaryRow(reason=reason, count=len(scenario_ids), scenario_ids=scenario_ids)
        for reason, scenario_ids in sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]


def format_failure_summary(
    rows: list[FailureSummaryRow], *, column_name: str = "failure_reason"
) -> str:
    if not rows:
        return "No failure reasons found."

    headers = [column_name, "count", "scenario_ids"]
    table_rows = [[row.reason, str(row.count), ", ".join(row.scenario_ids)] for row in rows]
    widths = [
        max(len(row[index]) for row in [headers, *table_rows]) for index in range(len(headers))
    ]

    lines = [
        "| " + " | ".join(_pad(value, widths[index]) for index, value in enumerate(headers)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_pad(value, widths[index]) for index, value in enumerate(row)) + " |"
        for row in table_rows
    )
    return "\n".join(lines)


def _pad(value: str, width: int) -> str:
    return value + (" " * (width - len(value)))


if __name__ == "__main__":
    raise SystemExit(main())
