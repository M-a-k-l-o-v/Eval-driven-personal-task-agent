"""Summarize failure reasons from a BAYMAX eval result JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureSummaryRow:
    reason: str
    count: int
    scenario_ids: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Eval result JSON path.")
    args = parser.parse_args()

    try:
        rows = summarize_failures(args.result)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(format_failure_summary(rows))
    return 0


def summarize_failures(path: Path) -> list[FailureSummaryRow]:
    with path.open(encoding="utf-8") as result_file:
        data: dict[str, Any] = json.load(result_file)

    scenario_results = data.get("scenario_results")
    if not isinstance(scenario_results, list):
        raise ValueError(f"{path} does not contain scenario_results.")

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

    return [
        FailureSummaryRow(reason=reason, count=len(scenario_ids), scenario_ids=scenario_ids)
        for reason, scenario_ids in sorted(
            scenarios_by_reason.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]


def format_failure_summary(rows: list[FailureSummaryRow]) -> str:
    if not rows:
        return "No failure reasons found."

    headers = ["failure_reason", "count", "scenario_ids"]
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
