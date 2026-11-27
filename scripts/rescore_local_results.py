"""Reparse saved local generations and replay fake tools without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from baymax.eval.cli import build_result_payload
from baymax.eval.lora_runner import parse_lora_agent_response
from baymax.eval.runner import ScenarioRunResult, run_scenario
from baymax.eval.scenario_loader import load_scenario_directories


def rescore_local_results(
    source_path: Path, output_path: Path, scenario_dirs: list[Path]
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    for directory in scenario_dirs:
        if not directory.is_dir():
            raise ValueError(f"Scenario directory does not exist: {directory}")
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    scenarios = {scenario.id: scenario for scenario in load_scenario_directories(scenario_dirs)}
    original_results = [ScenarioRunResult.model_validate(row) for row in source["scenario_results"]]
    if len(original_results) != source["scenario_count"]:
        raise ValueError("Source scenario_count does not match its results")
    if len({row.scenario_id for row in original_results}) != len(original_results):
        raise ValueError("Duplicate scenario IDs in source results")
    run_id = str(uuid4())
    results: list[ScenarioRunResult] = []
    used_scenarios: dict[str, Any] = {}
    for original in original_results:
        if original.raw_model_output is None:
            raise ValueError(f"Missing raw generation for {original.scenario_id}")
        if original.scenario_id not in scenarios:
            raise ValueError(f"Missing scenario {original.scenario_id}")
        scenario = scenarios[original.scenario_id]
        used_scenarios[scenario.id] = scenario.model_dump(mode="json")
        response = parse_lora_agent_response(original.raw_model_output)
        replay = run_scenario(scenario, response, run_id=run_id)
        results.append(
            replay.model_copy(
                update={
                    "raw_model_output": original.raw_model_output,
                    "latency_ms": original.latency_ms,
                    "cost_usd": original.cost_usd,
                }
            )
        )
    output = build_result_payload(
        results=results,
        backend_name=source["backend_name"],
        run_id=run_id,
        runner="local-raw-rescore-v1",
    )
    output["benchmark_config"].update(
        {
            "parser_version": "strict-complete-response-v1",
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_run_id": source.get("run_id"),
            "source_benchmark_config": source.get("benchmark_config"),
            "scenario_sha256": hashlib.sha256(
                json.dumps(used_scenarios, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "generation_rerun": False,
            "latency_and_cost_inherited": True,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, action="append", required=True)
    args = parser.parse_args()
    output = rescore_local_results(args.source, args.output, args.scenarios)
    print(f"Re-scored {output['scenario_count']} saved generations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
