import json
from pathlib import Path

import pytest

from baymax.eval.cli import _build_parser, main
from baymax.eval.lora_runner import LocalLoraConfig, LocalLoraInvocation, parse_lora_agent_response
from baymax.eval.scenario_loader import Scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "scenarios" / "v1"


def test_run_scripted_cli_writes_result_json(tmp_path: Path) -> None:
    responses_path = tmp_path / "scripted_responses.json"
    output_path = tmp_path / "v1-baseline.json"
    responses_path.write_text(
        json.dumps(
            {
                "calendar_clarify_time_001": {
                    "message": "What time should I schedule it?",
                    "tool_calls": [],
                    "decision_type": "clarify",
                }
            }
        ),
        encoding="utf-8",
    )
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    source_scenario = SCENARIO_DIR / "calendar_clarify_time_001.json"
    (scenarios_dir / source_scenario.name).write_text(
        source_scenario.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run-scripted",
            "--scenarios",
            str(scenarios_dir),
            "--responses",
            str(responses_path),
            "--output",
            str(output_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["backend_name"] == "scripted"
    assert result["scenario_count"] == 1
    assert result["aggregate_metrics"]["task_success_rate"] == 1.0
    assert result["aggregate_metrics"]["average_decision_accuracy"] == 1.0
    assert result["aggregate_metrics"]["decision_coverage_rate"] == 1.0
    assert result["aggregate_metrics"]["macro_f1"] == 1 / 3
    assert result["aggregate_metrics"]["invalid_output_rate"] == 0.0
    assert result["decision_confusion_matrix"]["clarify"]["clarify"] == 1
    assert result["scenario_results"][0]["scenario_id"] == "calendar_clarify_time_001"
    assert result["scenario_results"][0]["trace_id"] is None
    assert result["scenario_results"][0]["task_id"].endswith(".calendar_clarify_time_001.0")
    assert result["scenario_results"][0]["latency_ms"] >= 0
    assert result["scenario_results"][0]["cost_usd"] == 0.0
    assert result["scenario_results"][0]["score"]["task_success"] is True
    assert result["aggregate_metrics"]["average_latency_ms"] >= 0
    assert result["aggregate_metrics"]["total_cost_usd"] == 0.0
    assert result["benchmark_config"] == {
        "runner": "scripted",
        "uses_fake_adapters": True,
    }


def test_run_scripted_cli_creates_output_parent_directory(tmp_path: Path) -> None:
    responses_path = tmp_path / "scripted_responses.json"
    output_path = tmp_path / "nested" / "results" / "v1-baseline.json"
    responses_path.write_text("{}", encoding="utf-8")
    scenarios_dir = tmp_path / "empty_scenarios"
    scenarios_dir.mkdir()

    exit_code = main(
        [
            "run-scripted",
            "--scenarios",
            str(scenarios_dir),
            "--responses",
            str(responses_path),
            "--output",
            str(output_path),
        ]
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert output_path.exists()
    assert result["scenario_count"] == 0
    assert result["aggregate_metrics"]["task_success_rate"] is None
    assert result["aggregate_metrics"]["average_decision_accuracy"] is None
    assert result["aggregate_metrics"]["decision_coverage_rate"] is None
    assert result["aggregate_metrics"]["macro_f1"] is None
    assert result["aggregate_metrics"]["invalid_output_rate"] is None
    assert result["aggregate_metrics"]["average_latency_ms"] is None
    assert result["aggregate_metrics"]["total_cost_usd"] == 0.0


def test_scenarios_flag_accepts_one_directory() -> None:
    args = _build_parser().parse_args(
        [
            "run-agent-local",
            "--scenarios",
            "scenarios/v1",
            "--output",
            "results/output.json",
        ]
    )

    assert args.scenarios == [Path("scenarios/v1")]


def test_scenarios_flag_can_combine_multiple_directories() -> None:
    args = _build_parser().parse_args(
        [
            "run-agent-local",
            "--scenarios",
            "scenarios/v1",
            "--scenarios",
            "scenarios/v2",
            "--output",
            "results/output.json",
        ]
    )

    assert args.scenarios == [Path("scenarios/v1"), Path("scenarios/v2")]


@pytest.mark.parametrize("command", ["run-agent-local", "run-agent-lora"])
@pytest.mark.parametrize("compact_json", [False, True])
def test_local_cli_records_input_format_without_changing_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, compact_json: bool
) -> None:
    configs: list[LocalLoraConfig] = []
    raw = '[{"name":"request_clarification","arguments":{"question":"What time?"}}]'

    class FakeAgent:
        def __init__(self, config: LocalLoraConfig) -> None:
            configs.append(config)

        def invoke(self, scenario: Scenario) -> LocalLoraInvocation:
            assert scenario.id == "calendar_clarify_time_001"
            return LocalLoraInvocation(raw_text=raw, response=parse_lora_agent_response(raw))

    monkeypatch.setattr("baymax.eval.cli.LocalLoraAgent", FakeAgent)
    output_path = tmp_path / "nested" / "result.json"
    argv = [
        command,
        "--scenarios",
        str(SCENARIO_DIR),
        "--scenario-id",
        "calendar_clarify_time_001",
        "--output",
        str(output_path),
    ]
    if command == "run-agent-lora":
        argv.extend(["--adapter", "unit-adapter"])
    if compact_json:
        argv.append("--compact-json")
    assert main(argv) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert configs[0].compact_json is compact_json
    assert configs[0].max_new_tokens == 256
    assert configs[0].temperature == 0.0
    assert result["benchmark_config"]["input_json_format"] == (
        "compact" if compact_json else "spaced"
    )
    assert result["benchmark_config"]["local_model"] == configs[0].model_dump(mode="json")
    assert result["scenario_count"] == 1
    assert result["aggregate_metrics"]["task_success_rate"] == 1.0
    assert result["scenario_results"][0]["raw_model_output"] == raw
    assert result["decision_confusion_matrix"]["clarify"]["clarify"] == 1


def test_compact_json_cli_rejects_existing_output_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve old result", encoding="utf-8")
    monkeypatch.setattr("baymax.eval.cli.LocalLoraAgent", lambda _: pytest.fail("No model"))
    with pytest.raises(SystemExit, match="requires a new output file"):
        main(
            [
                "run-agent-lora",
                "--scenarios",
                str(SCENARIO_DIR),
                "--adapter",
                "unit-adapter",
                "--compact-json",
                "--output",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "preserve old result"
