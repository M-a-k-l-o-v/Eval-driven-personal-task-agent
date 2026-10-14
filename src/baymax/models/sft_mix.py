"""Mix external xLAM SFT data with synthetic BAYMAX-control SFT data."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from baymax.models.sft_filter import TargetFilterReview

EXPECTED_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")


class SFTMixConfig(BaseModel):
    """Configuration for combining xLAM and BAYMAX-control SFT datasets."""

    model_config = ConfigDict(extra="forbid")

    xlam_dir: Path = Path("data/sft/v1/xlam")
    baymax_dir: Path = Path("data/sft/v1/baymax-control")
    output_dir: Path = Path("data/sft/v1/xlam-baymax-mix")
    seed: int = 1
    train_baymax_repeat_count: int = Field(default=1, ge=1)
    train_baymax_clarification_extra_repeat_count: int = Field(default=0, ge=0)
    train_baymax_refusal_extra_repeat_count: int = Field(default=0, ge=0)
    eval_baymax_repeat_count: int = Field(default=1, ge=1)
    shuffle: bool = True
    train_cap_row_count: int | None = Field(default=None, ge=1)
    train_cap_baymax_row_count: int = Field(default=400, ge=0)
    train_cap_baymax_clarification_count: int = Field(default=100, ge=0)
    train_cap_baymax_refusal_count: int = Field(default=100, ge=0)
    preserve_contrast_groups: bool = False
    validation_cap_row_count: int | None = Field(default=None, ge=1)
    validation_cap_baymax_row_count: int = Field(default=90, ge=0)
    validation_cap_baymax_clarification_count: int = Field(default=30, ge=0)
    validation_cap_baymax_refusal_count: int = Field(default=30, ge=0)
    reference_mix_dir: Path | None = None
    replace_reference_baymax_train: bool = False
    rebalance_reference_train: bool = False

    @model_validator(mode="after")
    def validate_validation_quota(self) -> SFTMixConfig:
        if self.rebalance_reference_train and (
            self.reference_mix_dir is None
            or self.replace_reference_baymax_train
            or self.preserve_contrast_groups
        ):
            raise ValueError(
                "Train rebalancing requires a reference mix and no contrast replacement."
            )
        if self.replace_reference_baymax_train and (
            self.reference_mix_dir is None or not self.preserve_contrast_groups
        ):
            raise ValueError("BAYMAX replacement requires a reference mix and contrast groups.")
        if self.reference_mix_dir is not None and (
            self.train_cap_row_count is None
            or self.validation_cap_row_count is None
            or (self.preserve_contrast_groups and not self.replace_reference_baymax_train)
            or self.train_baymax_repeat_count != 1
            or self.eval_baymax_repeat_count != 1
            or self.train_baymax_clarification_extra_repeat_count != 0
            or self.train_baymax_refusal_extra_repeat_count != 0
        ):
            raise ValueError(
                "Reference sampling requires both caps, no repeats and no group sampler."
            )
        if self.preserve_contrast_groups and self.train_cap_row_count is None:
            raise ValueError("Group-preserving sampling requires a training cap.")
        if self.validation_cap_row_count is not None and (
            self.validation_cap_baymax_row_count > self.validation_cap_row_count
            or self.validation_cap_baymax_clarification_count
            + self.validation_cap_baymax_refusal_count
            > self.validation_cap_baymax_row_count
        ):
            raise ValueError("Validation quotas exceed the validation cap or BAYMAX contribution.")
        return self


def mix_sft_datasets(config: SFTMixConfig) -> SFTSplitManifest:
    """Write a mixed SFT dataset from existing xLAM and BAYMAX-control JSONL files."""

    xlam_manifest = _read_manifest(config.xlam_dir)
    baymax_manifest = _read_manifest(config.baymax_dir)
    if xlam_manifest.system_prompt_sha256 != baymax_manifest.system_prompt_sha256:
        raise ValueError("Source datasets must have the same system prompt.")
    reference = _read_reference_sampling(config) if config.reference_mix_dir is not None else None
    sampling_report: dict[str, Any] = {"splits": {}}
    if reference is not None:
        if config.output_dir.exists() and (
            not config.output_dir.is_dir() or any(config.output_dir.iterdir())
        ):
            raise ValueError("Reference sampling requires a new or empty output directory.")
        destination = config.output_dir.resolve()
        for directory in (config.xlam_dir, config.baymax_dir, config.reference_mix_dir):
            assert directory is not None
            source = directory.resolve()
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ValueError(
                    "Reference sampling input and output directories must not overlap."
                )
        sampling_report["reference"] = reference[2]
    group_by_index = (
        _read_contrast_groups(config.baymax_dir, baymax_manifest)
        if config.preserve_contrast_groups
        else None
    )
    reference_rows = (
        _read_full_reference_rows(config)
        if config.replace_reference_baymax_train or config.rebalance_reference_train
        else None
    )
    if (
        config.preserve_contrast_groups
        and config.output_dir.exists()
        and any(config.output_dir.iterdir())
    ):
        raise ValueError("Group-preserving mixes require a new, empty output directory.")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[SplitName, Path] = {
        "train": config.output_dir / "train.jsonl",
        "validation": config.output_dir / "validation.jsonl",
        "test": config.output_dir / "test.jsonl",
    }
    split_rows: dict[SplitName, list[PreparedSFTRow]] = {}
    output_source_index = 0

    for split_name in EXPECTED_SPLITS:
        if reference_rows is not None and split_name != "train":
            split_rows[split_name] = reference_rows[split_name]
            sampling_report["splits"][split_name] = {"preserved_rows": len(split_rows[split_name])}
            continue
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
        if reference_rows is not None and config.rebalance_reference_train:
            assert reference is not None
            rows, report = _rebalance_reference_train_order(
                rows, reference_rows, reference[1], config
            )
            sampling_report["splits"][split_name] = report
        elif reference_rows is not None:
            assert reference is not None and group_by_index is not None
            rows, report = _contrast_reference_train_order(
                rows, reference_rows, reference[1], config, group_by_index
            )
            sampling_report["splits"][split_name] = report
        elif reference is not None and split_name in reference[0]:
            rows, report = _reference_cap_order(
                rows, reference[0][split_name], reference[1], config, split_name
            )
            sampling_report["splits"][split_name] = report
        elif split_name == "train" and config.train_cap_row_count is not None:
            rows = _cap_aware_train_order(rows, config, group_by_index=group_by_index)
        if (
            reference is None
            and split_name == "validation"
            and config.validation_cap_row_count is not None
        ):
            rows = _cap_aware_validation_order(rows, config)

        split_rows[split_name] = rows
        if reference is None:
            _write_jsonl(output_paths[split_name], [row.example for row in rows])

    if reference is not None:
        if _read_reference_sampling(config)[2] != reference[2]:
            raise ValueError("Reference sampling inputs changed during the build.")
        for split_name, rows in split_rows.items():
            if reference_rows is not None and split_name != "train":
                assert config.reference_mix_dir is not None
                shutil.copyfile(
                    config.reference_mix_dir / f"{split_name}.jsonl", output_paths[split_name]
                )
            else:
                _write_jsonl(output_paths[split_name], [row.example for row in rows])
        (config.output_dir / "sampling-report.json").write_text(
            json.dumps(sampling_report, indent=2) + "\n", encoding="utf-8"
        )

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

    payload = {
        "format_version": SFT_FORMAT_VERSION,
        "seed": config.seed,
        "train_baymax_repeat_count": config.train_baymax_repeat_count,
        "train_baymax_clarification_extra_repeat_count": (
            config.train_baymax_clarification_extra_repeat_count
        ),
        "train_baymax_refusal_extra_repeat_count": (config.train_baymax_refusal_extra_repeat_count),
        "eval_baymax_repeat_count": config.eval_baymax_repeat_count,
        "shuffle": config.shuffle,
        "train_cap_row_count": config.train_cap_row_count,
        "train_cap_baymax_row_count": config.train_cap_baymax_row_count,
        "train_cap_baymax_clarification_count": (config.train_cap_baymax_clarification_count),
        "train_cap_baymax_refusal_count": config.train_cap_baymax_refusal_count,
        "xlam_dataset_id": xlam_manifest.dataset_id,
        "xlam_revision": xlam_manifest.revision,
        "xlam_config_hash": xlam_manifest.config_hash,
        "baymax_dataset_id": baymax_manifest.dataset_id,
        "baymax_revision": baymax_manifest.revision,
        "baymax_config_hash": baymax_manifest.config_hash,
    }
    # Opt-in fields leave historical mix hashes unchanged when the sampler is disabled.
    if config.validation_cap_row_count is not None:
        payload["validation_cap"] = {
            "row_count": config.validation_cap_row_count,
            "baymax_row_count": config.validation_cap_baymax_row_count,
            "clarification_count": config.validation_cap_baymax_clarification_count,
            "refusal_count": config.validation_cap_baymax_refusal_count,
            "sampler_version": 1,
        }
    if config.preserve_contrast_groups:
        payload["contrast_sampler_version"] = 1
        payload["contrast_groups_sha256"] = _sha256_json(
            json.loads((config.baymax_dir / "contrast-groups.json").read_text(encoding="utf-8"))
        )
    if config.reference_mix_dir is not None:
        payload["reference_sampling"] = _read_reference_sampling(config)[2]
    return _sha256_json(payload)


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


def _read_reference_sampling(
    config: SFTMixConfig,
) -> tuple[
    dict[SplitName, list[PreparedSFTRow]],
    dict[tuple[SplitName, str], list[str]],
    dict[str, Any],
]:
    assert config.reference_mix_dir is not None
    manifest_bytes = (config.reference_mix_dir / "split-manifest.json").read_bytes()
    reference_manifest = SFTSplitManifest.model_validate_json(manifest_bytes)
    xlam_manifest = _read_manifest(config.xlam_dir)
    if (
        reference_manifest.system_prompt_sha256 != xlam_manifest.system_prompt_sha256
        or reference_manifest.seed != config.seed
    ):
        raise ValueError("Reference sampling must preserve the prompt identity and seed.")
    review_bytes = (config.xlam_dir / "filter-review.json").read_bytes()
    review = TargetFilterReview.model_validate_json(review_bytes)
    provenance_bytes = (config.xlam_dir / "filter-provenance.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    if not isinstance(provenance, dict) or not isinstance(provenance.get("files"), dict):
        raise ValueError("Invalid target-filter provenance.")
    review_hash = hashlib.sha256(review_bytes).hexdigest()
    expected_config_hash = _sha256_json(
        {"filter_version": 2, "policy": review.policy, "review_sha256": review_hash}
    )
    if (
        provenance.get("review_sha256") != review_hash
        or provenance.get("policy") != review.policy
        or provenance.get("parent_manifest_sha256") != review.manifest_sha256
        or xlam_manifest.config_hash != expected_config_hash
        or xlam_manifest.revision
        != _sha256_json({"config_hash": expected_config_hash, "files": provenance.get("files")})
    ):
        raise ValueError("Current xLAM artifact is not bound to its target-filter review.")
    for split in EXPECTED_SPLITS:
        entry = provenance["files"].get(split)
        if not isinstance(entry, dict) or not isinstance(entry.get("retained_sha256"), str):
            raise ValueError(f"{split}: target-filter file identity is missing.")
        with (config.xlam_dir / f"{split}.jsonl").open("rb") as source:
            actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_hash != entry["retained_sha256"]:
            raise ValueError(f"{split}: filtered xLAM file changed after publication.")
    exclusions: dict[tuple[SplitName, str], list[str]] = {
        (row.split, row.example_sha256): [str(reason) for reason in row.reasons]
        for row in review.approved_rows
    }
    prefixes: dict[SplitName, list[PreparedSFTRow]] = {}
    prefix_hashes: dict[str, str] = {}
    caps: dict[SplitName, int | None] = {
        "train": config.train_cap_row_count,
        "validation": config.validation_cap_row_count,
    }
    for split, cap in caps.items():
        assert cap is not None
        entry = reference_manifest.splits[split]
        if entry.count != len(entry.source_rows) or entry.count < cap:
            raise ValueError(f"{split}: reference provenance is inconsistent or cap is too large.")
        rows = []
        with (config.reference_mix_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
            for line_number, line in enumerate(islice(source, cap), 1):
                try:
                    example = SFTExample.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(
                        f"{split} line {line_number}: invalid reference SFT row."
                    ) from error
                rows.append(
                    PreparedSFTRow(source=entry.source_rows[line_number - 1], example=example)
                )
        if len(rows) != cap:
            raise ValueError(f"{split}: reference file is shorter than its cap.")
        prefixes[split] = rows
        prefix_hashes[split] = _sha256_json(
            [
                {
                    "source": row.source.model_dump(mode="json"),
                    "content_sha256": _sft_example_hash(row.example),
                }
                for row in rows
            ]
        )
    replacement_identity: dict[str, Any] = {}
    if config.replace_reference_baymax_train:
        replacement_identity = {
            "baymax_train_replacement_version": 1,
            "reference_split_sha256": {
                split: _file_sha256(config.reference_mix_dir / f"{split}.jsonl")
                for split in EXPECTED_SPLITS
            },
            "replacement_train_sha256": _file_sha256(config.baymax_dir / "train.jsonl"),
            "contrast_groups_sha256": _file_sha256(config.baymax_dir / "contrast-groups.json"),
        }
    elif config.rebalance_reference_train:
        replacement_identity = {
            "train_rebalance_version": 1,
            "reference_split_sha256": {
                split: _file_sha256(config.reference_mix_dir / f"{split}.jsonl")
                for split in EXPECTED_SPLITS
            },
            "baymax_train_sha256": _file_sha256(config.baymax_dir / "train.jsonl"),
        }
    return (
        prefixes,
        exclusions,
        {
            "sampler_version": 1,
            "reference_revision": reference_manifest.revision,
            "reference_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "prefix_sha256": prefix_hashes,
            "exclusion_review_sha256": review_hash,
            "filter_provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            **replacement_identity,
        },
    )


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_full_reference_rows(config: SFTMixConfig) -> dict[SplitName, list[PreparedSFTRow]]:
    assert config.reference_mix_dir is not None
    manifest = _read_manifest(config.reference_mix_dir)
    result: dict[SplitName, list[PreparedSFTRow]] = {}
    for split in EXPECTED_SPLITS:
        entry = manifest.splits[split]
        examples = _read_sft_examples(config.reference_mix_dir / f"{split}.jsonl")
        if len(examples) != entry.count or len(entry.source_rows) != entry.count:
            raise ValueError(f"{split}: full reference file and provenance counts differ.")
        result[split] = [
            PreparedSFTRow(source=source, example=example)
            for source, example in zip(entry.source_rows, examples, strict=True)
        ]
    return result


def _sft_input_hash(example: SFTExample) -> str:
    return _sha256_json([m.model_dump(mode="json") for m in example.messages[:-1]])


def _contrast_reference_train_order(
    rows: list[PreparedSFTRow],
    reference_rows: dict[SplitName, list[PreparedSFTRow]],
    exclusions: dict[tuple[SplitName, str], list[str]],
    config: SFTMixConfig,
    group_by_index: dict[int, str],
) -> tuple[list[PreparedSFTRow], dict[str, Any]]:
    assert config.train_cap_row_count is not None
    cap = config.train_cap_row_count
    original = reference_rows["train"]
    # Reuse reference quota/duplicate checks, without allowing any xLAM replacement.
    _reference_cap_order(original, original[:cap], exclusions, config, "train")
    if any(_source_name(row) != "xlam" for row in original[cap:]):
        raise ValueError("Reference training tail must contain only xLAM rows.")
    available_xlam = {_sft_example_hash(row.example) for row in rows if _source_name(row) == "xlam"}
    if any(
        _sft_example_hash(row.example) not in available_xlam
        for row in original
        if _source_name(row) == "xlam"
    ):
        raise ValueError("Matched replacement must preserve every reference xLAM training row.")
    baymax_rows = [row for row in rows if _source_name(row) == "baymax"]
    if len(baymax_rows) != _read_manifest(config.baymax_dir).splits["train"].count:
        raise ValueError("Replacement training rows do not match their source manifest.")
    protected_inputs = {
        _sft_input_hash(row.example)
        for split in ("validation", "test")
        for row in reference_rows[split]
    }
    blocked_groups = {
        group_by_index[_baymax_source_index(row)]
        for row in baymax_rows
        if _baymax_source_index(row) in group_by_index
        and _sft_input_hash(row.example) in protected_inputs
    }
    eligible = [
        row
        for row in baymax_rows
        if _sft_input_hash(row.example) not in protected_inputs
        and group_by_index.get(_baymax_source_index(row)) not in blocked_groups
    ]
    eligible_groups = {
        index: group for index, group in group_by_index.items() if group not in blocked_groups
    }
    fixed_xlam_prefix = [row for row in original[:cap] if _source_name(row) == "xlam"]
    selected = _cap_aware_train_order(
        eligible + fixed_xlam_prefix, config, group_by_index=eligible_groups
    )
    baymax_selected = [row for row in selected[:cap] if _source_name(row) == "baymax"]
    grouped_counts = Counter(
        eligible_groups[_baymax_source_index(row)]
        for row in baymax_selected
        if _baymax_source_index(row) in eligible_groups
    )
    group_sizes = Counter(eligible_groups.values())
    complete_groups = sorted(g for g, count in grouped_counts.items() if count == group_sizes[g])
    if not complete_groups:
        raise ValueError("Matched replacement selected no complete contrast groups.")
    buckets: dict[str, list[PreparedSFTRow]] = {}
    for row in baymax_selected:
        buckets.setdefault(_reference_behavior(row), []).append(row)
    result = []
    replacements = []
    for position, row in enumerate(original, 1):
        if _source_name(row) != "baymax":
            result.append(row)
            continue
        replacement = buckets[_reference_behavior(row)].pop(0)
        result.append(replacement)
        old_hash, new_hash = _sft_example_hash(row.example), _sft_example_hash(replacement.example)
        if old_hash != new_hash:
            replacements.append(
                {
                    "position": position,
                    "old_content_sha256": old_hash,
                    "new_content_sha256": new_hash,
                }
            )
    if any(buckets.values()):
        raise ValueError("Matched replacement did not consume the requested BAYMAX contribution.")
    return result, {
        "cap_rows": cap,
        "kept_baymax": len(baymax_selected) - len(replacements),
        "replaced_baymax": len(replacements),
        "kept_xlam": len(fixed_xlam_prefix),
        "replaced_xlam": 0,
        "preserved_xlam_total": len(original) - len(baymax_selected),
        "excluded_heldout_groups": sorted(blocked_groups),
        "complete_contrast_groups": complete_groups,
        "baymax_class_counts": dict(Counter(_reference_behavior(row) for row in baymax_selected)),
        "replacements": replacements,
    }


def _reference_behavior(row: PreparedSFTRow) -> str:
    behavior = _row_behavior(row)
    return behavior if behavior in {"request_clarification", "refuse_request"} else "productive"


def _rebalance_reference_train_order(
    rows: list[PreparedSFTRow],
    reference_rows: dict[SplitName, list[PreparedSFTRow]],
    exclusions: dict[tuple[SplitName, str], list[str]],
    config: SFTMixConfig,
) -> tuple[list[PreparedSFTRow], dict[str, Any]]:
    """Add unique boundary rows in place of capped xLAM, retaining the reference controls."""
    assert config.train_cap_row_count is not None
    cap = config.train_cap_row_count
    original = reference_rows["train"]
    prefix = original[:cap]
    counts = Counter(_reference_behavior(row) for row in prefix if _source_name(row) == "baymax")
    old_config = config.model_copy(
        update={
            "train_cap_baymax_row_count": sum(counts.values()),
            "train_cap_baymax_clarification_count": counts["request_clarification"],
            "train_cap_baymax_refusal_count": counts["refuse_request"],
        }
    )
    _, check = _reference_cap_order(rows, prefix, exclusions, old_config, "train")
    if check["replaced_xlam"] or any(_source_name(row) != "xlam" for row in original[cap:]):
        raise ValueError("Rebalancing requires a compatible reference with an xLAM-only tail.")
    available_xlam = {_sft_example_hash(row.example) for row in rows if _source_name(row) == "xlam"}
    if any(_sft_example_hash(row.example) not in available_xlam for row in original[cap:]):
        raise ValueError("Rebalancing cannot retain reference xLAM rows missing from the source.")
    productive_target = (
        config.train_cap_baymax_row_count
        - config.train_cap_baymax_clarification_count
        - config.train_cap_baymax_refusal_count
    )
    if productive_target != counts["productive"]:
        raise ValueError("Rebalancing must preserve the reference BAYMAX productive count.")
    protected_inputs = {
        _sft_input_hash(row.example) for split_rows in reference_rows.values() for row in split_rows
    }
    candidates = [
        row
        for row in rows
        if _source_name(row) == "baymax" and _sft_input_hash(row.example) not in protected_inputs
    ]
    candidates.sort(
        key=lambda row: _sha256_json(
            {
                "seed": config.seed,
                "content_sha256": _sft_example_hash(row.example),
            }
        )
    )
    additions: list[PreparedSFTRow] = []
    seen_inputs: set[str] = set()
    for behavior, target in (
        ("request_clarification", config.train_cap_baymax_clarification_count),
        ("refuse_request", config.train_cap_baymax_refusal_count),
    ):
        needed = target - counts[behavior]
        if needed < 0:
            raise ValueError("Rebalancing cannot remove existing BAYMAX boundary rows.")
        selected = 0
        for row in candidates:
            digest = _sft_input_hash(row.example)
            if selected == needed:
                break
            if _reference_behavior(row) != behavior or digest in seen_inputs:
                continue
            additions.append(row)
            seen_inputs.add(digest)
            selected += 1
        if selected != needed:
            raise ValueError(
                f"Not enough unique same-split {behavior} additions: "
                f"found {selected}, need {needed}."
            )
    slots = [index for index, row in enumerate(prefix) if _source_name(row) == "xlam"]
    if len(additions) > len(slots):
        raise ValueError("Boundary additions exceed the available capped xLAM slots.")
    random.Random(config.seed).shuffle(slots)
    random.Random(config.seed).shuffle(additions)
    selected_prefix = list(prefix)
    displaced = []
    for slot, addition in zip(slots, additions):
        displaced.append(prefix[slot])
        selected_prefix[slot] = addition
    report = {
        "cap_rows": cap,
        "kept_baymax": sum(counts.values()),
        "kept_xlam": len(slots) - len(additions),
        "replaced_xlam": 0,
        "added_boundary_rows": len(additions),
        "added_content_sha256": [_sft_example_hash(row.example) for row in additions],
        "displaced_xlam_sha256": [_sft_example_hash(row.example) for row in displaced],
        "baymax_class_counts": dict(
            Counter(
                _reference_behavior(row) for row in selected_prefix if _source_name(row) == "baymax"
            )
        ),
    }
    return selected_prefix + displaced + original[cap:], report


def _reference_cap_order(
    rows: list[PreparedSFTRow],
    reference: list[PreparedSFTRow],
    exclusions: dict[tuple[SplitName, str], list[str]],
    config: SFTMixConfig,
    split: SplitName,
) -> tuple[list[PreparedSFTRow], dict[str, Any]]:
    keys = [(_source_name(row), _sft_example_hash(row.example)) for row in reference]
    if len({key[1] for key in keys}) != len(keys) or any(
        source not in {"xlam", "baymax"} for source, _ in keys
    ):
        raise ValueError(f"{split}: reference cap contains duplicates or unknown sources.")
    baymax_count = (
        config.train_cap_baymax_row_count
        if split == "train"
        else config.validation_cap_baymax_row_count
    )
    clarify_count = (
        config.train_cap_baymax_clarification_count
        if split == "train"
        else config.validation_cap_baymax_clarification_count
    )
    refuse_count = (
        config.train_cap_baymax_refusal_count
        if split == "train"
        else config.validation_cap_baymax_refusal_count
    )
    expected = Counter(
        {
            "request_clarification": clarify_count,
            "refuse_request": refuse_count,
            "productive": baymax_count - clarify_count - refuse_count,
        }
    )
    actual = Counter(_reference_behavior(row) for row in reference if _source_name(row) == "baymax")
    if actual != expected:
        raise ValueError(
            f"{split}: reference BAYMAX class counts do not match the requested quotas."
        )
    available: dict[tuple[str, str], PreparedSFTRow] = {}
    for row in rows:
        key = (_source_name(row), _sft_example_hash(row.example))
        if key[0] == "xlam" and (split, key[1]) in exclusions:
            raise ValueError(f"{split}: an approved exclusion is still in the filtered source.")
        available.setdefault(key, row)
    reserved = {digest for source, digest in keys if (source, digest) in available}
    candidates = [
        (digest, row)
        for (source, digest), row in available.items()
        if source == "xlam" and digest not in reserved
    ]
    candidates.sort(
        key=lambda pair: _sha256_json(
            {"seed": config.seed, "split": split, "content_sha256": pair[0]}
        )
    )
    selected: list[PreparedSFTRow] = []
    selected_hashes = set(reserved)
    kept = Counter()
    replacements = []
    for position, (source, digest) in enumerate(keys, 1):
        if (source, digest) in available:
            selected.append(available[source, digest])
            kept[source] += 1
            continue
        if source != "xlam" or (split, digest) not in exclusions:
            raise ValueError(
                f"{split} position {position}: reference row is missing "
                "without an approved xLAM exclusion."
            )
        candidate = next(((h, row) for h, row in candidates if h not in selected_hashes), None)
        if candidate is None:
            raise ValueError(f"{split}: not enough unique same-split replacement xLAM rows.")
        new_digest, replacement = candidate
        selected.append(replacement)
        selected_hashes.add(new_digest)
        replacements.append(
            {
                "position": position,
                "old_content_sha256": digest,
                "new_content_sha256": new_digest,
                "reasons": exclusions[split, digest],
            }
        )
    selected_ids = {id(row) for row in selected}
    if split == "train":
        _select_unique_rows(
            rows=rows,
            selected=selected,
            selected_ids=selected_ids,
            seen_hashes=selected_hashes,
            limit=len(rows),
            predicate=lambda row: _source_name(row) == "xlam",
        )
    else:
        selected.extend(row for row in rows if id(row) not in selected_ids)
    return selected, {
        "cap_rows": len(reference),
        "kept_baymax": kept["baymax"],
        "kept_xlam": kept["xlam"],
        "replaced_xlam": len(replacements),
        "replacements": replacements,
    }


def _cap_aware_train_order(
    rows: list[PreparedSFTRow],
    config: SFTMixConfig,
    *,
    group_by_index: dict[int, str] | None = None,
) -> list[PreparedSFTRow]:
    """Return a fixed unique BAYMAX pool followed by unique xLAM rows."""

    cap = config.train_cap_row_count
    if cap is None:
        return rows

    baymax_target = min(config.train_cap_baymax_row_count, cap)
    clarification_target = min(config.train_cap_baymax_clarification_count, baymax_target)
    refusal_target = min(
        config.train_cap_baymax_refusal_count, baymax_target - clarification_target
    )
    baymax_other_target = baymax_target - clarification_target - refusal_target

    selected: list[PreparedSFTRow] = []
    selected_ids: set[int] = set()
    seen_hashes: set[str] = set()

    if group_by_index is not None:
        _select_contrast_groups(
            rows,
            group_by_index,
            {
                "request_clarification": clarification_target,
                "refuse_request": refusal_target,
                "productive": baymax_other_target,
            },
            selected,
            selected_ids,
            seen_hashes,
        )
        # Boundary fillers must be ungrouped; productive fillers may stand alone.
        rows_for_baymax = [
            row
            for row in rows
            if _source_name(row) != "baymax" or _baymax_source_index(row) not in group_by_index
        ]
    else:
        rows_for_baymax = rows
    selected_clarification = sum(_row_behavior(row) == "request_clarification" for row in selected)
    selected_refusal = sum(_row_behavior(row) == "refuse_request" for row in selected)
    selected_productive = len(selected) - selected_clarification - selected_refusal

    _select_unique_rows(
        rows=rows_for_baymax,
        selected=selected,
        selected_ids=selected_ids,
        seen_hashes=seen_hashes,
        limit=clarification_target - selected_clarification,
        predicate=lambda row: (
            _source_name(row) == "baymax" and _row_behavior(row) == "request_clarification"
        ),
    )
    clarification_count = sum(_row_behavior(row) == "request_clarification" for row in selected)
    if clarification_count != clarification_target:
        raise ValueError(
            f"Training split has only {clarification_count} selectable unique "
            "BAYMAX clarification rows; "
            f"the cap requires {clarification_target}. Add unique training data or lower the quota."
        )
    _select_unique_rows(
        rows=rows_for_baymax,
        selected=selected,
        selected_ids=selected_ids,
        seen_hashes=seen_hashes,
        limit=refusal_target - selected_refusal,
        predicate=lambda row: (
            _source_name(row) == "baymax" and _row_behavior(row) == "refuse_request"
        ),
    )
    refusal_count = sum(_row_behavior(row) == "refuse_request" for row in selected)
    if refusal_count != refusal_target:
        raise ValueError(
            f"Training split has only {refusal_count} unique BAYMAX refusal rows; "
            f"the cap requires {refusal_target}. Add unique training data or lower the quota."
        )
    _select_unique_rows(
        rows=rows,
        selected=selected,
        selected_ids=selected_ids,
        seen_hashes=seen_hashes,
        limit=baymax_other_target - selected_productive,
        predicate=lambda row: (
            _source_name(row) == "baymax"
            and _row_behavior(row) not in {"request_clarification", "refuse_request"}
        ),
    )
    other_count = len(selected) - clarification_target - refusal_target
    if other_count != baymax_other_target:
        raise ValueError(
            f"Training split has only {other_count} unique BAYMAX productive rows; "
            f"the cap requires {baymax_other_target}. Add unique training data or lower the quota."
        )
    _select_unique_rows(
        rows=rows,
        selected=selected,
        selected_ids=selected_ids,
        seen_hashes=seen_hashes,
        limit=cap - len(selected),
        predicate=lambda row: _source_name(row) == "xlam",
    )
    if len(selected) < cap:
        raise ValueError(f"Only {len(selected)} unique rows available for a training cap of {cap}.")
    _select_unique_rows(
        rows=rows,
        selected=selected,
        selected_ids=selected_ids,
        seen_hashes=seen_hashes,
        limit=len(rows),
        predicate=lambda row: _source_name(row) == "xlam",
    )

    return selected


def _baymax_source_index(row: PreparedSFTRow) -> int:
    return int(str(row.source.source_row_id).split(":")[2])


def _read_contrast_groups(dataset_dir: Path, manifest: SFTSplitManifest) -> dict[int, str]:
    """Bind exported source IDs to the training partition, not evaluation fixtures."""
    payload = json.loads((dataset_dir / "contrast-groups.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("revision") != manifest.revision:
        raise ValueError("Contrast-group metadata does not match the source revision.")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Missing contrast-group membership.")
    partition_by_id = {
        source.source_row_id: split_name
        for split_name, split in manifest.splits.items()
        for source in split.source_rows
    }
    if len(partition_by_id) != sum(split.count for split in manifest.splits.values()):
        raise ValueError("Source manifest contains duplicate IDs or inconsistent row counts.")
    group_by_id: dict[str, str] = {}
    for group, members in groups.items():
        if not isinstance(group, str) or not isinstance(members, list) or len(members) < 2:
            raise ValueError("A contrast group must have a name and at least two source IDs.")
        partitions = set()
        for member in members:
            if (
                not isinstance(member, str)
                or member not in partition_by_id
                or member in group_by_id
            ):
                raise ValueError("Unknown or repeated contrast-group source ID.")
            group_by_id[member] = group
            partitions.add(partition_by_id[member])
        if len(partitions) != 1:
            raise ValueError("Contrast group crosses source split boundaries.")
    return {
        index: group_by_id[source.source_row_id]
        for index, source in enumerate(manifest.splits["train"].source_rows)
        if source.source_row_id in group_by_id
    }


def _select_contrast_groups(
    rows: list[PreparedSFTRow],
    group_by_index: dict[int, str],
    quotas: dict[str, int],
    selected: list[PreparedSFTRow],
    selected_ids: set[int],
    seen_hashes: set[str],
) -> None:
    groups: dict[str, list[PreparedSFTRow]] = {}
    # Ignore weighted copies: one instance per source row is sufficient.
    seen_indices: set[int] = set()
    for row in rows:
        if _source_name(row) != "baymax":
            continue
        index = _baymax_source_index(row)
        if index in group_by_index and index not in seen_indices:
            groups.setdefault(group_by_index[index], []).append(row)
            seen_indices.add(index)
    expected_sizes = Counter(group_by_index.values())
    if any(len(group) != expected_sizes[name] for name, group in groups.items()) or set(
        groups
    ) != set(expected_sizes):
        raise ValueError("Training rows are missing a contrast-group member.")
    remaining = dict(quotas)
    # Triplets first, then productive/boundary pairs; ties retain seeded input ordering.
    ordered = sorted(
        groups.values(),
        key=lambda group: (
            -len(set(_row_behavior(row) for row in group)),
            -int(
                any(
                    _row_behavior(row) in {"direct_tool_call", "multi_tool_sequence"}
                    for row in group
                )
            ),
        ),
    )
    for group in ordered:
        hashes = [_sft_example_hash(row.example) for row in group]
        counts = Counter(
            _row_behavior(row)
            if _row_behavior(row) in {"request_clarification", "refuse_request"}
            else "productive"
            for row in group
        )
        if len(set(hashes)) != len(group) or seen_hashes.intersection(hashes):
            continue
        if any(count > remaining[label] for label, count in counts.items()):
            continue
        selected.extend(group)
        selected_ids.update(id(row) for row in group)
        seen_hashes.update(hashes)
        for label, count in counts.items():
            remaining[label] -= count


def _cap_aware_validation_order(
    rows: list[PreparedSFTRow], config: SFTMixConfig
) -> list[PreparedSFTRow]:
    cap = config.validation_cap_row_count
    if cap is None:
        return rows
    selected: list[PreparedSFTRow] = []
    selected_ids: set[int] = set()
    seen_hashes: set[str] = set()
    targets: list[tuple[str, int, Callable[[PreparedSFTRow], bool]]] = [
        (
            "BAYMAX clarification",
            config.validation_cap_baymax_clarification_count,
            lambda row: (
                _source_name(row) == "baymax" and _row_behavior(row) == "request_clarification"
            ),
        ),
        (
            "BAYMAX refusal",
            config.validation_cap_baymax_refusal_count,
            lambda row: _source_name(row) == "baymax" and _row_behavior(row) == "refuse_request",
        ),
        (
            "BAYMAX productive",
            config.validation_cap_baymax_row_count
            - config.validation_cap_baymax_clarification_count
            - config.validation_cap_baymax_refusal_count,
            lambda row: (
                _source_name(row) == "baymax"
                and _row_behavior(row) in {"direct_tool_call", "multi_tool_sequence"}
            ),
        ),
        (
            "xLAM",
            cap - config.validation_cap_baymax_row_count,
            lambda row: _source_name(row) == "xlam",
        ),
    ]
    for label, target, predicate in targets:
        before = len(selected)
        _select_unique_rows(
            rows=rows,
            selected=selected,
            selected_ids=selected_ids,
            seen_hashes=seen_hashes,
            limit=target,
            predicate=predicate,
        )
        if len(selected) - before != target:
            raise ValueError(
                f"Validation split has only {len(selected) - before} unique {label} rows; "
                f"the cap requires {target}. Lower the quota; do not borrow from another split."
            )
    if config.shuffle:
        random.Random(_split_shuffle_seed(config.seed, "validation")).shuffle(selected)
    # Keep every original validation row exactly once, including unselected tail rows.
    return selected + [row for row in rows if id(row) not in selected_ids]


def _select_unique_rows(
    *,
    rows: list[PreparedSFTRow],
    selected: list[PreparedSFTRow],
    selected_ids: set[int],
    seen_hashes: set[str],
    limit: int,
    predicate: Callable[[PreparedSFTRow], bool],
) -> None:
    if limit <= 0:
        return

    selected_count = 0
    for row in rows:
        if selected_count >= limit:
            return
        if id(row) in selected_ids or not predicate(row):
            continue
        row_hash = _sft_example_hash(row.example)
        if row_hash in seen_hashes:
            continue
        selected.append(row)
        selected_ids.add(id(row))
        seen_hashes.add(row_hash)
        selected_count += 1


def _source_name(row: PreparedSFTRow) -> str:
    source_row_id = row.source.source_row_id
    if isinstance(source_row_id, str) and ":" in source_row_id:
        return source_row_id.split(":", 1)[0]
    return "unknown"


def _row_behavior(row: PreparedSFTRow) -> str:
    payload = _assistant_call_payload(row.example)
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


def _assistant_call_payload(example: SFTExample) -> list[dict[str, Any]] | None:
    assistant_message = example.messages[-1]
    try:
        payload = json.loads(assistant_message.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if not all(isinstance(call, dict) for call in payload):
        return None
    return payload


def _sft_example_hash(example: SFTExample) -> str:
    return _sha256_json(example.model_dump(mode="json"))


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

    first_call_name = _first_assistant_call_name(example)
    if first_call_name == "request_clarification":
        return base_repeat_count + config.train_baymax_clarification_extra_repeat_count
    if first_call_name == "refuse_request":
        return base_repeat_count + config.train_baymax_refusal_extra_repeat_count

    return base_repeat_count


def _first_assistant_call_name(example: SFTExample) -> str | None:
    assistant_message = example.messages[-1]
    try:
        payload = json.loads(assistant_message.content)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, list) or not payload:
        return None
    first_call = payload[0]
    if not isinstance(first_call, dict):
        return None

    name = first_call.get("name")
    return name if isinstance(name, str) else None


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
