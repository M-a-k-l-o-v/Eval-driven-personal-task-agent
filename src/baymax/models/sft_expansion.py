"""Double a frozen training prefix using unique same-split data; never resplit."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from baymax.models.baymax_sft import synthetic_baymax_example_to_sft_example
from baymax.models.sft_dataset import (
    PreparedSFTRow,
    SFTExample,
    SFTSkippedRow,
    SFTSourceRowRef,
    SFTSplitManifest,
)
from baymax.models.sft_expansion_examples import EXPANSION_VERSION, generate_expansion_examples
from baymax.models.sft_filter import SPLITS, _staging_directory
from baymax.models.sft_mix import (
    _file_sha256,
    _row_behavior,
    _sft_example_hash,
    _sft_input_hash,
    _source_name,
)
from baymax.models.sft_serialization import _json, name_first_example
from baymax.models.sft_validation import validate_sft_target_contract


def _counts(rows: list[PreparedSFTRow]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "source": dict(Counter(_source_name(row) for row in rows)),
        "baymax_classes": dict(
            Counter(_row_behavior(row) for row in rows if _source_name(row) == "baymax")
        ),
        "classes": dict(Counter(_row_behavior(row) for row in rows)),
        "unique_inputs": len({_sft_input_hash(row.example) for row in rows}),
        "unique_rows": len({_sft_example_hash(row.example) for row in rows}),
        "exact_duplicate_rows": len(rows) - len({_sft_example_hash(row.example) for row in rows}),
    }


def expand_training_dataset(
    *, dataset_dir: Path, output_dir: Path, train_cap: int = 1000
) -> SFTSplitManifest:
    """Retain the old prefix, add an equal-sized unique block, and freeze held-outs."""
    source, destination = dataset_dir.resolve(), output_dir.resolve()
    if (
        destination.exists()
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Use a new nonoverlapping dataset directory.")
    manifest_path = source / "split-manifest.json"
    parent_bytes = manifest_path.read_bytes()
    parent = SFTSplitManifest.model_validate_json(parent_bytes)
    if (
        train_cap < 1
        or set(parent.splits) != set(SPLITS)
        or sum(entry.count for entry in parent.splits.values()) != parent.converted_rows
        or parent.loaded_rows - len(parent.skipped_rows) != parent.converted_rows
    ):
        raise ValueError("Invalid parent split counts or training cap.")
    paths = {}
    for split, entry in parent.splits.items():
        path = Path(entry.path)
        if not path.is_absolute() and not path.exists():
            path = source / path.name
        path = path.resolve()
        if not path.is_relative_to(source) or len(entry.source_rows) != entry.count:
            raise ValueError("Invalid parent path or provenance count.")
        paths[split] = path
    parent_hashes = {split: _file_sha256(path) for split, path in paths.items()}
    sources = generate_expansion_examples(seed=parent.seed)
    if not sources or len(sources) >= train_cap:
        raise ValueError("Additions must leave capacity for new xLAM examples.")
    needed_xlam = train_cap - len(sources)
    prefix: list[PreparedSFTRow] = []
    selected: list[PreparedSFTRow] = []
    selected_lines: dict[int, bytes] = {}
    protected: dict[str, tuple[str, int]] = {}
    excluded_tail: dict[int, dict[str, Any]] = {}
    all_inputs: set[str] = set()
    selected_inputs: set[str] = set()
    prompt = None
    for split in ("validation", "test", "train"):
        count = 0
        with paths[split].open("rb") as stream:
            for count, line in enumerate(stream, 1):
                if count > parent.splits[split].count:
                    raise ValueError("Parent file count exceeds its manifest.")
                example = SFTExample.model_validate_json(line)
                system = example.messages[0].content
                if hashlib.sha256(system.encode()).hexdigest() != parent.system_prompt_sha256:
                    raise ValueError("Parent prompt hash mismatch.")
                digest = _sft_input_hash(example)
                all_inputs.add(digest)
                if split != "train":
                    protected.setdefault(digest, (split, count))
                    continue
                if digest in protected:
                    heldout_split, heldout_line = protected[digest]
                    if count <= train_cap:
                        raise ValueError(
                            f"Frozen training prefix line {count} overlaps "
                            f"{heldout_split} line {heldout_line} by input "
                            f"(sha256={digest}). Cannot preserve this prefix safely."
                        )
                    excluded_tail[count] = {
                        "train_line": count,
                        "source": parent.splits[split]
                        .source_rows[count - 1]
                        .model_dump(mode="json"),
                        "input_sha256": digest,
                        "heldout_split": heldout_split,
                        "heldout_line": heldout_line,
                    }
                    continue
                prepared = PreparedSFTRow(
                    example=example, source=parent.splits[split].source_rows[count - 1]
                )
                if count <= train_cap:
                    if digest in selected_inputs or not line.endswith(b"\n"):
                        raise ValueError(
                            "Parent prefix contains duplicate inputs or missing line breaks."
                        )
                    prefix.append(prepared)
                    selected_inputs.add(digest)
                    prompt = system
                elif (
                    len(selected) < needed_xlam
                    and _source_name(prepared) == "xlam"
                    and digest not in selected_inputs
                ):
                    issues = validate_sft_target_contract(example, location=f"train:{count}")
                    if issues:
                        raise ValueError(f"Invalid xLAM addition: {issues[0].message}")
                    if _row_behavior(prepared) not in {"direct_tool_call", "multi_tool_sequence"}:
                        raise ValueError("xLAM additions must be productive tool-call examples.")
                    selected.append(prepared)
                    selected_lines[count] = line
                    selected_inputs.add(digest)
        if count != parent.splits[split].count:
            raise ValueError("Parent file count differs from its manifest.")
    if len(prefix) != train_cap or len(selected) != needed_xlam or prompt is None:
        raise ValueError("Insufficient parent prefix or unique same-split xLAM additions.")
    before = _counts(prefix)
    if set(before["source"]) != {"baymax", "xlam"}:
        raise ValueError("The frozen prefix must contain only BAYMAX and xLAM sources.")
    next_index = 1 + max(
        ref.source_index for entry in parent.splits.values() for ref in entry.source_rows
    )
    additions = []
    for offset, item in enumerate(sources):
        example = name_first_example(
            synthetic_baymax_example_to_sft_example(item, system_prompt=prompt), location=item.id
        )
        prepared = PreparedSFTRow(
            example=example,
            source=SFTSourceRowRef(
                source_index=next_index + offset,
                source_row_id=f"baymax:train:{next_index + offset}:{item.id}",
                row_hash=hashlib.sha256(item.model_dump_json().encode()).hexdigest(),
            ),
        )
        digest = _sft_input_hash(example)
        if digest in all_inputs:
            raise ValueError("New BAYMAX input duplicates a parent training or held-out input.")
        all_inputs.add(digest)
        additions.append(prepared)
    added_counts = _counts(additions)
    if before["baymax_classes"] != added_counts["baymax_classes"]:
        raise ValueError(
            "New BAYMAX quantities must match the frozen prefix, including multi-tool count."
        )
    block = additions + selected
    random.Random(parent.seed).shuffle(block)
    after = _counts(prefix + block)
    if after["unique_inputs"] != train_cap * 2:
        raise ValueError("Expanded prefix contains duplicate inputs.")
    encoded_selected = {
        _sft_input_hash(row.example): line
        for row, line in zip(selected, selected_lines.values(), strict=True)
    }
    recipe = {
        "version": EXPANSION_VERSION,
        "split_guard_version": "input-disjoint-tail-filter-v1",
        "seed": parent.seed,
        "parent_cap": train_cap,
        "parent_manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "parent_files": parent_hashes,
        "source_sha256": hashlib.sha256(
            _json([item.model_dump(mode="json") for item in sources], sort_keys=True).encode()
        ).hexdigest(),
    }
    config_hash = hashlib.sha256(_json(recipe, sort_keys=True).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _staging_directory(destination) as staging:
        with paths["train"].open("rb") as stream, (staging / "train.jsonl").open("xb") as output:
            for number, line in enumerate(stream, 1):
                if number not in selected_lines and number not in excluded_tail:
                    output.write(line)
                if number == train_cap:
                    for row in block:
                        output.write(
                            encoded_selected[_sft_input_hash(row.example)]
                            if _source_name(row) == "xlam"
                            else (row.example.model_dump_json() + "\n").encode()
                        )
        entries = {}
        refs = [row.source for row in prefix + block] + [
            ref
            for number, ref in enumerate(parent.splits["train"].source_rows, 1)
            if number > train_cap and number not in selected_lines and number not in excluded_tail
        ]
        for split in SPLITS:
            if split != "train":
                shutil.copyfile(paths[split], staging / f"{split}.jsonl")
            entries[split] = parent.splits[split].model_copy(
                update={
                    "path": str(output_dir / f"{split}.jsonl"),
                    **({"count": len(refs), "source_rows": refs} if split == "train" else {}),
                }
            )
        hashes = {split: _file_sha256(staging / f"{split}.jsonl") for split in SPLITS}
        if any(hashes[split] != parent_hashes[split] for split in ("validation", "test")):
            raise ValueError("Held-out bytes changed.")
        if manifest_path.read_bytes() != parent_bytes or any(
            _file_sha256(paths[s]) != parent_hashes[s] for s in SPLITS
        ):
            raise ValueError("Parent changed during expansion.")
        revision = hashlib.sha256(
            _json({"recipe": recipe, "files": hashes}, sort_keys=True).encode()
        ).hexdigest()
        derived = parent.model_copy(
            update={
                "revision": revision,
                "config_hash": config_hash,
                "output_dir": str(output_dir),
                "splits": entries,
                "loaded_rows": parent.loaded_rows + len(additions),
                "converted_rows": parent.converted_rows + len(additions) - len(excluded_tail),
                "skipped_rows": [
                    *parent.skipped_rows,
                    *[
                        SFTSkippedRow(
                            source_index=entry["source"]["source_index"],
                            source_row_id=entry["source"]["source_row_id"],
                            reason=(
                                "Excluded unused parent training input overlapping "
                                f"{entry['heldout_split']}:{entry['heldout_line']} "
                                f"(sha256={entry['input_sha256']})."
                            ),
                        )
                        for entry in excluded_tail.values()
                    ],
                ],
            }
        )
        (staging / "split-manifest.json").write_text(
            derived.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (staging / "expansion-sources.jsonl").write_text(
            "".join(item.model_dump_json() + "\n" for item in sources), encoding="utf-8"
        )
        (staging / "expansion-report.json").write_text(
            json.dumps(
                {
                    **recipe,
                    "parent_revision": parent.revision,
                    "output_files": hashes,
                    "before_cap": before,
                    "after_cap": after,
                    "added_baymax": added_counts,
                    "added_xlam": [row.source.model_dump(mode="json") for row in selected],
                    "excluded_parent_tail": list(excluded_tail.values()),
                    "new_groups": sorted(
                        {item.contrast_group for item in sources if item.contrast_group}
                    ),
                    "split_policy": (
                        "New BAYMAX groups are training-only; "
                        "retained parent rows keep their original split membership. "
                        "Overlapping unused training-tail inputs are excluded and recorded; "
                        "frozen-prefix overlap blocks publication."
                    ),
                    "xlam_policy": (
                        "Move the next eligible unique parent-training xLAM rows into the cap; "
                        "do not duplicate them."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    return derived
