"""Reports for the exact SFT artifacts used by training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from baymax.models.sft_dataset import SFTSourceRowRef

BehaviorClass = Literal[
    "direct_tool_call",
    "multi_tool_sequence",
    "request_clarification",
    "refuse_request",
    "malformed",
]


class SFTBehaviorReportRow(BaseModel):
    """Behavior class distribution for an SFT artifact."""

    model_config = ConfigDict(extra="forbid")

    behavior_class: BehaviorClass
    rows: int
    row_percent: float
    trainable_tokens: int
    token_percent: float


class SFTSourceReportRow(BaseModel):
    """Source distribution for an SFT artifact."""

    model_config = ConfigDict(extra="forbid")

    source: str
    rows: int
    row_percent: float


class SFTDuplicateReport(BaseModel):
    """Exact duplicate row counts for an SFT artifact."""

    model_config = ConfigDict(extra="forbid")

    exact_duplicate_rows: int
    unique_rows: int


class SFTDuplicateByBehaviorRow(BaseModel):
    """Exact duplicate row counts grouped by behavior class."""

    model_config = ConfigDict(extra="forbid")

    behavior_class: BehaviorClass
    exact_duplicate_rows: int
    unique_rows: int


class SFTArtifactReport(BaseModel):
    """Summary of one final SFT artifact after split/shuffle/cap."""

    model_config = ConfigDict(extra="forbid")

    row_count: int
    behavior: list[SFTBehaviorReportRow]
    source: list[SFTSourceReportRow]
    duplicates: SFTDuplicateReport
    duplicates_by_behavior: list[SFTDuplicateByBehaviorRow]


def build_sft_artifact_report(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
    source_rows: list[SFTSourceRowRef] | None = None,
) -> SFTArtifactReport:
    """Build a report for the final rows passed toward tokenization/training."""

    behavior_row_counts: Counter[BehaviorClass] = Counter()
    behavior_token_counts: Counter[BehaviorClass] = Counter()
    source_counts: Counter[str] = Counter()
    row_hashes: Counter[str] = Counter()
    row_hashes_by_behavior: dict[BehaviorClass, Counter[str]] = {}

    for index, row in enumerate(rows):
        behavior_class = classify_sft_behavior(row)
        trainable_label_tokens = _trainable_label_tokens(
            row=row,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
        )
        if trainable_label_tokens == 0:
            continue

        behavior_row_counts[behavior_class] += 1
        behavior_token_counts[behavior_class] += trainable_label_tokens
        source_counts[_source_name(source_rows[index] if source_rows is not None else None)] += 1
        row_hash = _row_hash(row)
        row_hashes[row_hash] += 1
        row_hashes_by_behavior.setdefault(behavior_class, Counter())[row_hash] += 1

    total_rows = sum(behavior_row_counts.values())
    total_tokens = sum(behavior_token_counts.values())

    return SFTArtifactReport(
        row_count=total_rows,
        behavior=[
            SFTBehaviorReportRow(
                behavior_class=behavior_class,
                rows=behavior_row_counts[behavior_class],
                row_percent=_percent(behavior_row_counts[behavior_class], total_rows),
                trainable_tokens=behavior_token_counts[behavior_class],
                token_percent=_percent(behavior_token_counts[behavior_class], total_tokens),
            )
            for behavior_class in sorted(behavior_row_counts)
        ],
        source=[
            SFTSourceReportRow(
                source=source,
                rows=source_counts[source],
                row_percent=_percent(source_counts[source], total_rows),
            )
            for source in sorted(source_counts)
        ],
        duplicates=SFTDuplicateReport(
            exact_duplicate_rows=sum(count - 1 for count in row_hashes.values() if count > 1),
            unique_rows=len(row_hashes),
        ),
        duplicates_by_behavior=[
            SFTDuplicateByBehaviorRow(
                behavior_class=behavior_class,
                exact_duplicate_rows=sum(
                    count - 1
                    for count in row_hashes_by_behavior[behavior_class].values()
                    if count > 1
                ),
                unique_rows=len(row_hashes_by_behavior[behavior_class]),
            )
            for behavior_class in sorted(row_hashes_by_behavior)
        ],
    )


def classify_sft_behavior(row: dict[str, Any]) -> BehaviorClass:
    """Classify one SFT row by its assistant tool-call target."""

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


def _trainable_label_tokens(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    max_sequence_length: int,
) -> int:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return 0

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_sequence_length,
        add_special_tokens=False,
    )["input_ids"]
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_sequence_length,
        add_special_tokens=False,
    )["input_ids"]
    return max(len(full_tokens) - min(len(prompt_tokens), len(full_tokens)), 0)


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


def _source_name(source_row: SFTSourceRowRef | None) -> str:
    if source_row is None:
        return "unknown"
    source_row_id = source_row.source_row_id
    if isinstance(source_row_id, str) and ":" in source_row_id:
        return source_row_id.split(":", 1)[0]
    return "external"


def _row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _percent(value: int, total: int) -> float:
    if total == 0:
        return 0.0
    return value / total
