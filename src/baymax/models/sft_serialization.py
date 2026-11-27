"""Derive name-first targets without changing examples, answers or split membership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from baymax.models.sft_dataset import SFTExample, SFTSplitManifest, SFTSplitManifestEntry
from baymax.models.sft_filter import SPLITS, _staging_directory
from baymax.models.sft_validation import validate_sft_target_contract

TRANSFORM_VERSION = "assistant-call-name-first-v1"


def _json(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def name_first_example(example: SFTExample, *, location: str) -> SFTExample:
    """Require valid canonical targets; reorder only each call's two outer keys."""
    issues = validate_sft_target_contract(example, location=location)
    if issues:
        raise ValueError(f"{location}: invalid source target: {issues[0].message}")
    target = example.messages[-1].content
    calls = json.loads(target, object_pairs_hook=_unique_object)
    if target != _json(calls, sort_keys=True):
        raise ValueError(f"{location}: source target is not canonical arguments-first JSON.")
    reordered = [{"name": call["name"], "arguments": call["arguments"]} for call in calls]
    messages = [
        *example.messages[:-1],
        example.messages[-1].model_copy(update={"content": _json(reordered)}),
    ]
    return example.model_copy(update={"messages": messages})


def serialize_sft_name_first(*, dataset_dir: Path, output_dir: Path) -> SFTSplitManifest:
    """Publish a separate derived dataset and content hashes; never modify its parent."""
    source_dir, destination = dataset_dir.resolve(), output_dir.resolve()
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new artifact version.")
    if destination.is_relative_to(source_dir) or source_dir.is_relative_to(destination):
        raise ValueError("Source and output directories must not overlap.")
    manifest_path = source_dir / "split-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    parent = SFTSplitManifest.model_validate_json(manifest_bytes)
    if (
        set(parent.splits) != set(SPLITS)
        or sum(entry.count for entry in parent.splits.values()) != parent.converted_rows
    ):
        raise ValueError("Source manifest splits or converted-row count are inconsistent.")
    if parent.loaded_rows - len(parent.skipped_rows) != parent.converted_rows:
        raise ValueError("Source manifest loaded/skipped-row counts are inconsistent.")
    recipe = {
        "transform_version": TRANSFORM_VERSION,
        "parent_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    config_hash = hashlib.sha256(_json(recipe, sort_keys=True).encode()).hexdigest()
    entries: dict[str, SFTSplitManifestEntry] = {}
    files: dict[str, dict[str, Any]] = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _staging_directory(destination) as staging:
        for split in SPLITS:
            entry = parent.splits[split]
            path = Path(entry.path)
            if not path.is_absolute() and not path.exists():
                path = source_dir / path.name
            path = path.resolve()
            if not path.is_relative_to(source_dir) or len(entry.source_rows) != entry.count:
                raise ValueError(f"{split}: invalid source path or provenance count.")
            input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
            rows, calls = 0, 0
            with path.open("rb") as source, (staging / f"{split}.jsonl").open("xb") as out:
                for rows, line in enumerate(source, 1):
                    if rows > entry.count:
                        raise ValueError(f"{split}: row count exceeds the manifest.")
                    input_hash.update(line)
                    example = SFTExample.model_validate(
                        json.loads(line, object_pairs_hook=_unique_object)
                    )
                    converted = name_first_example(example, location=f"{split}:{rows}")
                    if hashlib.sha256(example.messages[0].content.encode()).hexdigest() != (
                        parent.system_prompt_sha256
                    ):
                        raise ValueError(f"{split}:{rows}: system prompt hash mismatch.")
                    calls += len(json.loads(converted.messages[-1].content))
                    encoded = (converted.model_dump_json() + "\n").encode()
                    output_hash.update(encoded)
                    out.write(encoded)
            if rows != entry.count:
                raise ValueError(f"{split}: row count is below the manifest.")
            entries[split] = entry.model_copy(update={"path": str(output_dir / f"{split}.jsonl")})
            files[split] = {
                "rows": rows,
                "calls": calls,
                "parent_sha256": input_hash.hexdigest(),
                "output_sha256": output_hash.hexdigest(),
            }
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Source manifest changed during conversion.")
        revision = hashlib.sha256(
            _json({"config_hash": config_hash, "files": files}, sort_keys=True).encode()
        ).hexdigest()
        derived = parent.model_copy(
            update={
                "config_hash": config_hash,
                "revision": revision,
                "output_dir": str(output_dir),
                "splits": entries,
            }
        )
        (staging / "split-manifest.json").write_text(
            derived.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (staging / "serialization-provenance.json").write_text(
            json.dumps(
                {
                    **recipe,
                    "parent_dataset_id": parent.dataset_id,
                    "parent_revision": parent.revision,
                    "files": files,
                    "source_refs": "Preserved parent identities; not transformed-content hashes.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    return derived
