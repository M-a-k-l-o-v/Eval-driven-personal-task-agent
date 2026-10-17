"""Prepare a separate grounding-data revision; never train or load a model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from baymax.models.sft_dataset import SFTExample, SFTSplitManifest
from baymax.models.sft_grounding import derive_grounding_dataset, grounding_preview
from baymax.models.sft_training import load_sft_jsonl
from baymax.models.training_config import (
    SFTTrainingConfig,
    load_training_config,
    training_config_with_hash,
    validate_training_config,
    write_training_config,
)

DEFAULT_REFERENCE = Path(
    "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json"
)


def prepare_experiment(
    *,
    reference_config: Path,
    output_dir: Path,
    config_path: Path,
    dry_run: bool = False,
    train_cap: int = 1000,
) -> SFTTrainingConfig | None:
    if not validate_training_config(reference_config, check_paths=True).valid:
        raise ValueError("Reference config, hash, paths or manifest is invalid.")
    reference = load_training_config(reference_config)
    manifest = SFTSplitManifest.model_validate_json(
        reference.data.split_manifest_path.read_text(encoding="utf-8")
    )
    if reference.runtime.seed != manifest.seed:
        raise ValueError("Reference training and dataset seeds must match.")
    if reference.training.first_tool_name_weight != 1 or reference.training.num_train_epochs != 2:
        raise ValueError("Use the unweighted two-epoch reference.")
    model_dir = Path("models") / config_path.stem
    if config_path.exists() or model_dir.exists() or output_dir.exists():
        raise ValueError("Dataset, config and model destinations must be new.")
    for directory in (reference.data.train_path.parent, output_dir):
        if config_path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Config must be outside dataset directories.")
    if dry_run:
        rows = [
            SFTExample.model_validate(row)
            for row in load_sft_jsonl(reference.data.train_path, max_rows=train_cap)
        ]
        if len(rows) != train_cap:
            raise ValueError("The reference must provide the requested training cap.")
        _, changes = grounding_preview(
            rows,
            [row.source_row_id for row in manifest.splits["train"].source_rows[:train_cap]],
            seed=reference.runtime.seed,
        )
        applied = [change for change in changes if "skipped_reason" not in change]
        print(f"Preview only: {len(applied)} revised training rows.")
        print(f"Duplicate undated variants kept unchanged: {len(changes) - len(applied)}")
        print(
            json.dumps(
                dict(Counter(rule for change in applied for rule in change["rules"])), indent=2
            )
        )
        print(
            "No artifacts, model or tokenizer created. "
            "Full split-overlap checks run during preparation."
        )
        return None
    manifest = derive_grounding_dataset(
        dataset_dir=reference.data.train_path.parent, output_dir=output_dir, train_cap=train_cap
    )
    payload = reference.model_dump(mode="json")
    payload["config_name"] = config_path.stem
    payload["data"].update(
        {
            "revision": manifest.revision,
            "split_manifest_path": str(output_dir / "split-manifest.json"),
            "validation_report_path": str(output_dir / "validation-report.json"),
            **{f"{split}_path": entry.path for split, entry in manifest.splits.items()},
        }
    )
    payload["runtime"].update(
        {
            "run_name": config_path.stem,
            "output_dir": str(model_dir),
            "expected_adapter_path": str(model_dir / "adapter"),
        }
    )
    config = training_config_with_hash(SFTTrainingConfig.model_validate(payload))
    write_training_config(config_path, config)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/sft/v1/name-first-grounding-v1")
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/sft/qwen2_5_1_5b_lora_grounding_v1_epochs2.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = prepare_experiment(**vars(args))
    except (ValueError, OSError) as error:
        print(f"error: {error}")
        return 1
    if config is not None:
        print(f"Dataset: {args.output_dir}")
        print(f"Config: {args.config_path}")
        print(f"Revision: {config.data.revision}")
        print(f"Config hash: {config.config_hash}")
        print("Class counts and xLAM slots preserved; validation/test copied byte-for-byte.")
        print(
            "Review grounding-changes.json, validate the dataset, "
            "and inspect tokenized counts before training."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
