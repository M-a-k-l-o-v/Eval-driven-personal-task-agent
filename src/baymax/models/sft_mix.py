"""Mix external xLAM SFT data with synthetic BAYMAX-control SFT data."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from baymax.models.sft_dataset import (
    SFT_FORMAT_VERSION,
    PreparedSFTRow,
    SFTExample,
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SFTSplitRatios,
    SplitName,
)

EXPECTED_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


class SFTMixConfig(BaseModel):
    """Configuration for combining xLAM and BAYMAX-control SFT datasets."""

    model_config = ConfigDict(extra="forbid")

    xlam_dir: Path = Path("data/sft/v1/xlam")
    baymax_dir: Path = Path("data/sft/v1/baymax-control")
    output_dir: Path = Path("data/sft/v1/xlam-baymax-mix")
    seed: int = 1
    train_baymax_repeat_count: int = Field(default=3, ge=1)
    train_baymax_boundary_extra_repeat_count: int = Field(default=2, ge=0)
    eval_baymax_repeat_count: int = Field(default=1, ge=1)
    shuffle: bool = True


def mix_sft_datasets(config: SFTMixConfig) -> SFTSplitManifest:
    """Write a mixed SFT dataset from existing xLAM and BAYMAX-control JSONL files."""

    xlam_manifest = _read_manifest(config.xlam_dir)
    baymax_manifest = _read_manifest(config.baymax_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[SplitName, Path] = {
        "train": config.output_dir / "train.jsonl",
        "validation": config.output_dir / "validation.jsonl",
        "test": config.output_dir / "test.jsonl",
    }
    split_rows: dict[SplitName, list[PreparedSFTRow]] = {}
    output_source_index = 0

    for split_name in EXPECTED_SPLITS:
        baymax_repeat_count = (
            config.train_baymax_repeat_count
            if split_name == "train"
            else config.eval_baymax_repeat_count
        )
        rows, output_source_index = _mixed_split_rows(
            split_name=split_name,
            config=config,
            baymax_repeat_count=baymax_repeat_count,
            output_source_index=output_source_index,
        )
        if config.shuffle:
            random.Random(_split_shuffle_seed(config.seed, split_name)).shuffle(rows)

        split_rows[split_name] = rows
        _write_jsonl(output_paths[split_name], [row.example for row in rows])

    revision = mixed_sft_revision(
        config=config,
        xlam_manifest=xlam_manifest,
        baymax_manifest=baymax_manifest,
    )
    converted_rows = sum(len(rows) for rows in split_rows.values())
    manifest = SFTSplitManifest(
        format_version=SFT_FORMAT_VERSION,
        config_hash=sft_mix_config_hash(
            config,
            xlam_manifest=xlam_manifest,
            baymax_manifest=baymax_manifest,
        ),
        dataset_id=f"mixed:{xlam_manifest.dataset_id}+{baymax_manifest.dataset_id}",
        revision=revision,
        source_split="mixed",
        seed=config.seed,
        max_rows=None,
        split_ratios=SFTSplitRatios(),
        system_prompt_sha256=xlam_manifest.system_prompt_sha256,
        output_dir=str(config.output_dir),
        splits={
            split_name: SFTSplitManifestEntry(
                path=str(output_paths[split_name]),
                count=len(rows),
                source_rows=[row.source for row in rows],
            )
            for split_name, rows in split_rows.items()
        },
        loaded_rows=converted_rows,
        converted_rows=converted_rows,
        skipped_rows=[],
    )
    (config.output_dir / "split-manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


def sft_mix_config_hash(
    config: SFTMixConfig,
    *,
    xlam_manifest: SFTSplitManifest,
    baymax_manifest: SFTSplitManifest,
) -> str:
    """Return the stable SHA256 hash for content-affecting mix config."""

    return _sha256_json(
        {
            "format_version": SFT_FORMAT_VERSION,
            "seed": config.seed,
            "train_baymax_repeat_count": config.train_baymax_repeat_count,
            "train_baymax_boundary_extra_repeat_count": (
                config.train_baymax_boundary_extra_repeat_count
            ),
            "eval_baymax_repeat_count": config.eval_baymax_repeat_count,
            "shuffle": config.shuffle,
            "xlam_dataset_id": xlam_manifest.dataset_id,
            "xlam_revision": xlam_manifest.revision,
            "xlam_config_hash": xlam_manifest.config_hash,
            "baymax_dataset_id": baymax_manifest.dataset_id,
            "baymax_revision": baymax_manifest.revision,
            "baymax_config_hash": baymax_manifest.config_hash,
        }
    )


def mixed_sft_revision(
    *,
    config: SFTMixConfig,
    xlam_manifest: SFTSplitManifest,
    baymax_manifest: SFTSplitManifest,
) -> str:
    """Return a content hash for the mixed dataset inputs and repeat policy."""

    return _sha256_json(
        {
            "mix_config_hash": sft_mix_config_hash(
                config,
                xlam_manifest=xlam_manifest,
                baymax_manifest=baymax_manifest,
            ),
            "xlam_split_counts": {
                split_name: split.count for split_name, split in xlam_manifest.splits.items()
            },
            "baymax_split_counts": {
                split_name: split.count for split_name, split in baymax_manifest.splits.items()
            },
        }
    )


def _mixed_split_rows(
    *,
    split_name: SplitName,
    config: SFTMixConfig,
    baymax_repeat_count: int,
    output_source_index: int,
) -> tuple[list[PreparedSFTRow], int]:
    rows: list[PreparedSFTRow] = []
    for source_name, dataset_dir, repeat_count in (
        ("xlam", config.xlam_dir, 1),
        ("baymax", config.baymax_dir, baymax_repeat_count),
    ):
        split_examples = _read_sft_examples(dataset_dir / f"{split_name}.jsonl")
        for source_index, example in enumerate(split_examples):
            example_repeat_count = _example_repeat_count(
                source_name=source_name,
                split_name=split_name,
                base_repeat_count=repeat_count,
                config=config,
                example=example,
            )
            for repeat_index in range(example_repeat_count):
                row_id = f"{source_name}:{split_name}:{source_index}:r{repeat_index}"
                rows.append(
                    PreparedSFTRow(
                        source=SFTSourceRowRef(
                            source_index=output_source_index,
                            source_row_id=row_id,
                            row_hash=_sha256_json(
                                {
                                    "seed": config.seed,
                                    "split": split_name,
                                    "source_row_id": row_id,
                                }
                            ),
                        ),
                        example=example,
                    )
                )
                output_source_index += 1

    return rows, output_source_index


def _example_repeat_count(
    *,
    source_name: str,
    split_name: SplitName,
    base_repeat_count: int,
    config: SFTMixConfig,
    example: SFTExample,
) -> int:
    if source_name != "baymax" or split_name != "train":
        return base_repeat_count
    if not _is_baymax_boundary_example(example):
        return base_repeat_count

    return base_repeat_count + config.train_baymax_boundary_extra_repeat_count


def _is_baymax_boundary_example(example: SFTExample) -> bool:
    assistant_message = example.messages[-1]
    try:
        payload = json.loads(assistant_message.content)
    except json.JSONDecodeError:
        return False

    if not isinstance(payload, list) or not payload:
        return False
    first_call = payload[0]
    if not isinstance(first_call, dict):
        return False

    return first_call.get("name") in {"request_clarification", "refuse_request"}


def _read_manifest(dataset_dir: Path) -> SFTSplitManifest:
    manifest_path = dataset_dir / "split-manifest.json"
    return SFTSplitManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def _read_sft_examples(path: Path) -> list[SFTExample]:
    examples: list[SFTExample] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                raw_example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            examples.append(SFTExample.model_validate(raw_example))

    return examples


def _write_jsonl(path: Path, examples: list[SFTExample]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(example.model_dump_json() + "\n")


def _split_shuffle_seed(seed: int, split_name: SplitName) -> int:
    return int(_sha256_json({"seed": seed, "split": split_name})[:8], 16)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
