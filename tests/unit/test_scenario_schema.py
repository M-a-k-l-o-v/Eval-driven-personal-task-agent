import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "src" / "baymax" / "eval" / "schemas" / "scenario.schema.json"


@pytest.mark.parametrize("suite_name", ["v1", "v2"])
def test_all_scenarios_validate_against_committed_json_schema(suite_name: str) -> None:
    with SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    scenario_paths = sorted((REPO_ROOT / "scenarios" / suite_name).glob("*.json"))

    assert scenario_paths, f"Expected at least one {suite_name} scenario fixture"

    failures: list[str] = []
    for path in scenario_paths:
        with path.open(encoding="utf-8") as scenario_file:
            scenario = json.load(scenario_file)

        errors = sorted(validator.iter_errors(scenario), key=lambda error: error.path)
        failures.extend(f"{path}: {error.message}" for error in errors)

    assert failures == []
