from __future__ import annotations

import json
from pathlib import Path

from baymax.core.meta_tools import META_TOOL_NAMES, meta_tool_specs
from baymax.eval.lora_runner import _available_tool_specs as lora_available_tool_specs
from baymax.eval.scenario_loader import load_scenario
from baymax.models.baymax_sft import _available_tool_specs as baymax_available_tool_specs
from baymax.models.sft_dataset import _tools_with_meta_tools
from baymax.service.inference import _META_TOOLS

SCENARIO_DIR = Path("scenarios/v1")


def test_meta_tool_names_are_stable() -> None:
    assert META_TOOL_NAMES == ("request_clarification", "refuse_request")
    assert [tool["name"] for tool in meta_tool_specs()] == list(META_TOOL_NAMES)


def test_training_and_eval_use_same_meta_tool_specs() -> None:
    expected = meta_tool_specs()
    scenario = load_scenario(SCENARIO_DIR / "calendar_create_001.json")

    xlam_tools = _tools_with_meta_tools([{"name": "lookup_item", "parameters": {}}])
    baymax_tools = baymax_available_tool_specs(["calendar.create_event"])
    lora_tools = lora_available_tool_specs(scenario)

    assert xlam_tools[:2] == expected
    assert baymax_tools[:2] == expected
    assert lora_tools[:2] == expected


def test_openai_meta_tools_wrap_same_specs() -> None:
    assert [tool["function"] for tool in _META_TOOLS] == meta_tool_specs()


def test_meta_tool_training_targets_are_json_tool_calls() -> None:
    clarification = [{"name": "request_clarification", "arguments": {"question": "Who?"}}]
    refusal = [{"name": "refuse_request", "arguments": {"reason": "Unsupported."}}]

    assert json.loads(json.dumps(clarification))[0]["name"] == "request_clarification"
    assert json.loads(json.dumps(refusal))[0]["name"] == "refuse_request"
