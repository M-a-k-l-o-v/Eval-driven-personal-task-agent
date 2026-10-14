"""Read-only raw-generation diagnostics; never repair outputs or alter scores."""

from __future__ import annotations

import json

from baymax.eval.lora_runner import parse_lora_agent_response


def classify_output_contract_failure(raw_model_output: object) -> str | None:
    """Explain strict rejection or absent routing, independently of the expected decision."""
    if not isinstance(raw_model_output, str) or not raw_model_output.strip():
        return "empty_generation"

    response = parse_lora_agent_response(raw_model_output)
    if response.parse_error is None:
        return "unknown_decision" if response.decision_type is None else None

    try:
        payload = json.loads(raw_model_output)
    except json.JSONDecodeError:
        return "malformed_call"

    if not isinstance(payload, list) or len(payload) != 1:
        return "malformed_call"
    call = payload[0]
    if not isinstance(call, dict):
        return "malformed_call"
    if call.get("name") == "request_clarification":
        return "attempted_clarification_invalid_arguments"
    if call.get("name") == "refuse_request":
        return "attempted_refusal_invalid_arguments"
    return "malformed_call"


def classify_boundary_generation(*, raw_model_output: object, expected_behavior: str) -> str:
    """Separate invalid contracts from valid boundary decisions with failed wording checks."""
    diagnostic = classify_output_contract_failure(raw_model_output)
    if diagnostic is not None:
        return diagnostic

    assert isinstance(raw_model_output, str)
    response = parse_lora_agent_response(raw_model_output)
    expected_decision = "clarify" if expected_behavior == "clarification" else "refuse"
    if response.decision_type == expected_decision:
        return "correct_decision_wrong_wording"
    if response.decision_type in {"clarify", "refuse"}:
        return "wrong_meta_tool"
    return "premature_tool"
