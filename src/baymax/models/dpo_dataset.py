"""Build inspectable decision preferences from frozen training-only BAYMAX rows."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from baymax.models.baymax_sft import (
    generate_synthetic_baymax_examples,
    synthetic_baymax_example_to_sft_example,
)
from baymax.models.dpo_act_rejections import AuthoredActRejection, authored_act_rejection
from baymax.models.sft_dataset import SFTExample, SFTMessage, SFTSplitManifest
from baymax.models.sft_mix import _file_sha256, _sft_input_hash
from baymax.models.sft_training import load_sft_jsonl
from baymax.models.sft_validation import validate_sft_target_contract
from baymax.models.training_config import load_training_config, validate_training_config

Decision = Literal["act", "clarify", "refuse"]


class PreferencePair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    counterpart_source_id: str | None = None
    chosen_decision: Decision
    rejected_decision: Decision
    negative_kind: str
    rejection_reason: str | None = None
    prompt: list[SFTMessage]
    chosen: list[SFTMessage]
    rejected: list[SFTMessage]


def decision(example: SFTExample) -> Decision:
    calls = json.loads(example.messages[-1].content)
    names = [call["name"] for call in calls]
    if names == ["request_clarification"]:
        return "clarify"
    if names == ["refuse_request"]:
        return "refuse"
    return "act"


def build_preference_pairs(
    anchors: dict[str, SFTExample],
    training_pool: dict[str, SFTExample],
    groups: dict[str, str],
    *,
    protected_inputs: set[str],
    seed: int,
    act_rejections: dict[str, AuthoredActRejection] | None = None,
) -> tuple[list[PreferencePair], dict[str, int]]:
    """Use wrong decisions, not malformed JSON, as rejected answers to each same prompt."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for source_id in sorted(training_pool):
        if source_id in groups:
            grouped[groups[source_id]].append(source_id)
    pairs: list[PreferencePair] = []
    missing = Counter()
    seen_inputs: dict[str, str] = {}
    fallback = {
        "clarify": [
            {
                "name": "request_clarification",
                "arguments": {
                    "question": "What additional information should I use to proceed?",
                },
            }
        ],
        "refuse": [
            {
                "name": "refuse_request",
                "arguments": {
                    "reason": "No supported tool is available for this request.",
                },
            }
        ],
    }
    for source_id, anchor in sorted(anchors.items()):
        issues = validate_sft_target_contract(anchor, location=source_id)
        if issues:
            raise ValueError(f"{source_id}: invalid chosen target: {issues[0].message}")
        digest = _sft_input_hash(anchor)
        if digest in protected_inputs:
            raise ValueError(f"{source_id}: training prompt overlaps a held-out prompt.")
        if digest in seen_inputs:
            if seen_inputs[digest] != anchor.messages[-1].content:
                raise ValueError(f"{source_id}: duplicate prompt has conflicting chosen answers.")
            continue
        seen_inputs[digest] = anchor.messages[-1].content
        chosen_decision = decision(anchor)
        counterparts = grouped.get(groups.get(source_id, ""), [])
        for rejected_decision in ("act", "clarify", "refuse"):
            if rejected_decision == chosen_decision:
                continue
            counterpart_id = next(
                (
                    candidate
                    for candidate in counterparts
                    if candidate != source_id
                    and decision(training_pool[candidate]) == rejected_decision
                    and _sft_input_hash(training_pool[candidate]) not in protected_inputs
                ),
                None,
            )
            authored = (
                (act_rejections or {}).get(source_id)
                if counterpart_id is None and rejected_decision == "act"
                else None
            )
            if counterpart_id is None and rejected_decision == "act" and authored is None:
                missing[f"{chosen_decision}_without_act_counterpart"] += 1
                continue
            rejected = (
                training_pool[counterpart_id].messages[-1]
                if counterpart_id is not None
                else SFTMessage(
                    role="assistant",
                    content=json.dumps(
                        authored.calls if authored is not None else fallback[rejected_decision],
                        sort_keys=True,
                    ),
                )
            )
            rejected_example = SFTExample(messages=anchor.messages[:-1] + [rejected])
            rejected_issues = validate_sft_target_contract(rejected_example, location=source_id)
            # An unavailable productive tool is an intentional capability negative, not bad JSON.
            if any(
                not issue.message.endswith("tool is not in the supplied tool list.")
                for issue in rejected_issues
            ):
                raise ValueError(f"{source_id}: rejected output has an unintended contract error.")
            if rejected_issues:
                AuthoredActRejection(
                    calls=json.loads(rejected.content),
                    reason="Intentional unavailable-tool negative.",
                )
            pairs.append(
                PreferencePair(
                    source_id=source_id,
                    counterpart_source_id=counterpart_id,
                    chosen_decision=chosen_decision,
                    rejected_decision=rejected_decision,
                    negative_kind=("unavailable_tool" if rejected_issues else "wrong_decision"),
                    rejection_reason=authored.reason if authored is not None else None,
                    prompt=anchor.messages[:-1],
                    chosen=[anchor.messages[-1]],
                    rejected=[rejected],
                )
            )
    if not pairs:
        raise ValueError("No unique training-only preference pairs could be built.")
    random.Random(seed).shuffle(pairs)
    return pairs, dict(missing)


def build_dpo_dataset(
    *,
    reference_config: Path,
    baymax_dir: Path,
    output_dir: Path,
    train_cap: int = 1000,
    complete_act_rejections: bool = False,
) -> dict[str, Any]:
    """Write preference data/review/provenance; never read benchmark fixtures or results."""
    if train_cap <= 0:
        raise ValueError("Training cap must be positive.")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Use a new or empty preference output directory.")
    validation = validate_training_config(reference_config)
    if not validation.valid:
        raise ValueError(f"Invalid reference config: {validation.issues[0].message}")
    config = load_training_config(reference_config)
    for source_dir in (baymax_dir, config.data.train_path.parent):
        destination, source = output_dir.resolve(), source_dir.resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Preference output must not overlap a source dataset.")
    input_paths = (
        reference_config,
        config.data.split_manifest_path,
        config.data.train_path,
        config.data.validation_path,
        config.data.test_path,
        baymax_dir / "split-manifest.json",
        baymax_dir / "train.jsonl",
    )
    input_hashes = {str(path): _file_sha256(path) for path in input_paths}
    manifest = SFTSplitManifest.model_validate_json(
        config.data.split_manifest_path.read_text(encoding="utf-8")
    )
    source_manifest = SFTSplitManifest.model_validate_json(
        (baymax_dir / "split-manifest.json").read_text(encoding="utf-8")
    )
    reference_rows = load_sft_jsonl(config.data.train_path, max_rows=train_cap)
    if len(reference_rows) != train_cap or manifest.splits["train"].count < train_cap:
        raise ValueError("Reference training data is shorter than the requested cap.")
    protected = {
        _sft_input_hash(SFTExample.model_validate(row))
        for path in (config.data.validation_path, config.data.test_path)
        for row in load_sft_jsonl(path)
    }
    source_rows = load_sft_jsonl(baymax_dir / "train.jsonl")
    refs = source_manifest.splits["train"].source_rows
    if len(source_rows) != len(refs):
        raise ValueError("BAYMAX source rows and manifest provenance counts differ.")
    training_pool = {
        str(ref.source_row_id): SFTExample.model_validate(row)
        for ref, row in zip(refs, source_rows, strict=True)
    }
    generated = {row.id: row for row in generate_synthetic_baymax_examples()}
    groups = {
        source_id: source.contrast_group
        for source_id, source in generated.items()
        if source.contrast_group is not None
    }
    for source_id, example in training_pool.items():
        source = generated.get(source_id)
        if (
            source is None
            or synthetic_baymax_example_to_sft_example(
                source,
                system_prompt=example.messages[0].content,
            )
            != example
        ):
            raise ValueError(
                f"{source_id}: generator/source drift; cannot infer contrast membership."
            )
        if _sft_input_hash(example) in protected:
            raise ValueError(
                f"{source_id}: BAYMAX training source overlaps reference held-out data."
            )
    anchors = {}
    for row, ref in zip(
        reference_rows, manifest.splits["train"].source_rows[:train_cap], strict=True
    ):
        source_id = str(ref.source_row_id)
        if not source_id.startswith("baymax:"):
            continue
        parts = source_id.split(":")
        if len(parts) != 4 or parts[1] != "train" or not parts[2].isdigit():
            raise ValueError("Unsupported BAYMAX training provenance.")
        index = int(parts[2])
        if index >= len(refs):
            raise ValueError("BAYMAX provenance index is outside the training source.")
        original_id = str(refs[index].source_row_id)
        if training_pool[original_id] != SFTExample.model_validate(row):
            raise ValueError(f"{original_id}: reference row does not match the training source.")
        anchors[original_id] = training_pool[original_id]
    pairs, missing = build_preference_pairs(
        anchors,
        training_pool,
        groups,
        protected_inputs=protected,
        seed=config.runtime.seed,
        act_rejections=(
            {
                source_id: rejection
                for source_id in anchors
                if (rejection := authored_act_rejection(generated[source_id])) is not None
            }
            if complete_act_rejections
            else None
        ),
    )
    data = "".join(
        json.dumps(
            pair.model_dump(
                mode="json",
                include={
                    "prompt",
                    "chosen",
                    "rejected",
                },
            ),
            sort_keys=True,
        )
        + "\n"
        for pair in pairs
    )
    review = "".join(pair.model_dump_json() + "\n" for pair in pairs)
    if input_hashes != {str(path): _file_sha256(path) for path in input_paths}:
        raise ValueError("Preference inputs changed during the build; no artifact was published.")
    report = {
        "format_version": "baymax-decision-preferences-v1",
        "input_sha256": input_hashes,
        "seed": config.runtime.seed,
        "reference_config_sha256": _file_sha256(reference_config),
        "reference_config_hash": config.config_hash,
        "reference_adapter": str(config.runtime.expected_adapter_path),
        "reference_revision": manifest.revision,
        "reference_manifest_sha256": _file_sha256(config.data.split_manifest_path),
        "reference_prefix_sha256": hashlib.sha256(
            json.dumps(reference_rows, sort_keys=True).encode()
        ).hexdigest(),
        "heldout_sha256": {
            str(path): _file_sha256(path)
            for path in (config.data.validation_path, config.data.test_path)
        },
        "source_revision": source_manifest.revision,
        "source_manifest_sha256": _file_sha256(baymax_dir / "split-manifest.json"),
        "source_train_sha256": _file_sha256(baymax_dir / "train.jsonl"),
        "train_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "pair_count": len(pairs),
        "unique_prompts": len(
            {_sft_input_hash(SFTExample(messages=pair.prompt + pair.chosen)) for pair in pairs}
        ),
        "chosen_counts": dict(Counter(pair.chosen_decision for pair in pairs)),
        "unique_anchor_counts": dict(
            Counter(
                chosen for _, chosen in {(pair.source_id, pair.chosen_decision) for pair in pairs}
            )
        ),
        "direction_counts": dict(
            Counter(f"{pair.chosen_decision}>{pair.rejected_decision}" for pair in pairs)
        ),
        "negative_counts": dict(Counter(pair.negative_kind for pair in pairs)),
        "missing_counterparts": missing,
        "complete_act_rejections": complete_act_rejections,
        "authored_act_rejection_count": sum(pair.rejection_reason is not None for pair in pairs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_text(data, encoding="utf-8", newline="\n")
    (output_dir / "pairs-review.jsonl").write_text(review, encoding="utf-8", newline="\n")
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
