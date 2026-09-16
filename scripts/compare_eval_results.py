"""Compare BAYMAX eval result JSON files in a compact table."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RESULTS = [
    "scripted=results/phase1/scripted/v1-baseline.json",
    "openai=results/phase1/openai/gpt-4o-mini-full.json",
    "qwen-base=results/phase1/qwen-base/full.json",
    "xlam-lora=results/phase1/qwen-lora/xlam-medium/full.json",
    "baymax-control=results/phase2/qwen-lora/baymax-control/full.json",
    "baymax-mix=results/phase2/qwen-lora/baymax-mix/full.json",
]

METRICS = [
    "task_success_rate",
    "average_tool_call_accuracy",
    "average_argument_accuracy",
    "average_clarification_accuracy",
    "average_refusal_accuracy",
    "average_hallucination_rate",
    "average_latency_ms",
    "total_cost_usd",
]


@dataclass(frozen=True)
class ResultSummary:
    label: str
    path: Path
    backend_name: str
    scenario_count: int
    metrics: dict[str, float | int | None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="*",
        help="Result path or label=path. Defaults to the main known BAYMAX runs.",
    )
    args = parser.parse_args()

    try:
        summaries = [load_result(value) for value in args.results or DEFAULT_RESULTS]
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(format_markdown_table(summaries))
    return 0


def load_result(value: str) -> ResultSummary:
    label, path = _parse_result_arg(value)
    with path.open(encoding="utf-8") as result_file:
        data: dict[str, Any] = json.load(result_file)

    metrics = data.get("aggregate_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} does not contain aggregate_metrics.")

    return ResultSummary(
        label=label,
        path=path,
        backend_name=str(data.get("backend_name", "")),
        scenario_count=int(data.get("scenario_count", 0)),
        metrics={metric: _number_or_none(metrics.get(metric)) for metric in METRICS},
    )


def format_markdown_table(summaries: list[ResultSummary]) -> str:
    headers = ["run", "scenarios", *METRICS]
    rows = [
        [
            summary.label,
            str(summary.scenario_count),
            *[_format_metric(metric, summary.metrics[metric]) for metric in METRICS],
        ]
        for summary in summaries
    ]
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


def _parse_result_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label:
            raise ValueError(f"missing label in result argument: {value}")
        return label, Path(raw_path)

    path = Path(value)
    return path.stem, path


def _number_or_none(value: object) -> float | int | None:
    if value is None or isinstance(value, int | float):
        return value
    raise ValueError(f"metric value must be numeric or null, got {type(value).__name__}.")


def _format_metric(metric: str, value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if metric == "average_latency_ms":
        return f"{value:.0f}"
    if metric == "total_cost_usd":
        return f"${value:.4f}"
    return f"{value:.3f}"


def _pad(value: str, width: int) -> str:
    return value + (" " * (width - len(value)))


if __name__ == "__main__":
    raise SystemExit(main())
