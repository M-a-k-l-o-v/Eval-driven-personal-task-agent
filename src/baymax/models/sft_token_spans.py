"""Map unchanged assistant JSON fields onto the actual supervised token positions."""

from __future__ import annotations

import json
from typing import Any

COMPONENTS = (
    "first_tool_name",
    "later_tool_name",
    "question_text",
    "reason_text",
    "other",
    "mixed_boundary",
)


def _json_object_fields(text: str, start: int) -> dict[str, tuple[Any, int, int]]:
    """Read field value spans with the standard JSON decoder, preserving serialization."""
    decoder = json.JSONDecoder()
    cursor = start + 1
    fields: dict[str, tuple[Any, int, int]] = {}
    while True:
        while text[cursor] in " \t\r\n":
            cursor += 1
        if text[cursor] == "}":
            return fields
        key, cursor = decoder.raw_decode(text, cursor)
        while text[cursor] in " \t\r\n":
            cursor += 1
        if not isinstance(key, str) or text[cursor] != ":" or key in fields:
            raise ValueError("Component mapping requires unambiguous JSON object fields.")
        cursor += 1
        while text[cursor] in " \t\r\n":
            cursor += 1
        value, end = decoder.raw_decode(text, cursor)
        fields[key] = value, cursor, end
        cursor = end
        while text[cursor] in " \t\r\n":
            cursor += 1
        if text[cursor] == ",":
            cursor += 1
        elif text[cursor] != "}":
            raise ValueError("Invalid JSON field delimiter in component mapping.")


def target_component_spans(target: str) -> list[tuple[str, int, int]]:
    """Locate only call names and meta-tool text values, excluding JSON quotes."""
    calls = json.loads(target)
    if not isinstance(calls, list) or not calls:
        raise ValueError("Component mapping requires a nonempty JSON array of calls.")
    cursor = target.index("[") + 1
    spans: list[tuple[str, int, int]] = []
    for number, call in enumerate(calls):
        while target[cursor] in " \t\r\n,":
            cursor += 1
        if not isinstance(call, dict) or target[cursor] != "{":
            raise ValueError("Component mapping requires JSON call objects.")
        fields = _json_object_fields(target, cursor)
        name, start, end = fields["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("Component mapping requires nonempty string tool names.")
        spans.append(("first_tool_name" if number == 0 else "later_tool_name", start + 1, end - 1))
        if name in ("request_clarification", "refuse_request"):
            key = "question" if name == "request_clarification" else "reason"
            arguments, arguments_start, _ = fields["arguments"]
            if not isinstance(arguments, dict):
                raise ValueError("Meta-tool arguments must be JSON objects.")
            value, start, end = _json_object_fields(target, arguments_start)[key]
            if not isinstance(value, str) or not value:
                raise ValueError("Meta-tool text must be a nonempty JSON string.")
            spans.append((f"{key}_text", start + 1, end - 1))
        _, cursor = json.JSONDecoder().raw_decode(target, cursor)
    return spans


def token_components(
    row: dict[str, Any], tokenizer: Any, tokenized: dict[str, Any], max_sequence_length: int
) -> list[str | None]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Component loss requires a fast tokenizer with character offsets.")
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    target = messages[-1]["content"]
    if not full.startswith(prompt + target):
        raise ValueError("Cannot locate the unchanged assistant target after the prompt.")
    encoded = tokenizer(
        full,
        truncation=True,
        max_length=max_sequence_length,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if encoded["input_ids"] != tokenized["input_ids"]:
        raise ValueError("Offset tokenization differs from the actual training input IDs.")
    prompt_ids = tokenizer(
        prompt, truncation=True, max_length=max_sequence_length, add_special_tokens=False
    )["input_ids"]
    if encoded["input_ids"][: len(prompt_ids)] != prompt_ids:
        raise ValueError("Rendered prompt tokens are not a prefix of the full training input.")
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(tokenized["labels"]):
        raise ValueError("Tokenizer did not return aligned character offsets.")
    spans = [
        (label, start + len(prompt), end + len(prompt))
        for label, start, end in target_component_spans(target)
    ]
    parts: list[str | None] = []
    for index, ((start, end), label) in enumerate(zip(offsets, tokenized["labels"], strict=True)):
        part = None
        if index and label != -100:
            overlapping = (
                [
                    (name, lower, upper)
                    for name, lower, upper in spans
                    if start < upper and end > lower
                ]
                if end > start
                else []
            )
            part = "other"
            if overlapping:
                name, lower, upper = overlapping[0]
                part = (
                    name
                    if len(overlapping) == 1 and lower <= start < end <= upper
                    else "mixed_boundary"
                )
        parts.append(part)
    return parts
