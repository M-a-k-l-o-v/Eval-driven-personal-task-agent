"""Prepare E's serialization-only experiment without loading a model or tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from baymax.models.sft_serialization import serialize_sft_name_first
from baymax.models.training_config import (
    SFTTrainingConfig,
    load_training_config,
    training_config_with_hash,
    validate_training_config,
    write_training_config,
)


def prepare_experiment(
    *, reference_config: Path, output_dir: Path, config_path: Path
) -> SFTTrainingConfig:
    report = validate_training_config(reference_config)
    if not report.valid:
        raise ValueError("Reference config/hash/manifest validation failed.")
    reference = load_training_config(reference_config)
    if config_path.exists():
        raise ValueError("Use a new config path; existing experiment configs are not overwritten.")
    model_dir = Path("models") / config_path.stem
    if model_dir.exists():
        raise ValueError("Use a new experiment name; its model output directory already exists.")
    if any(
        config_path.resolve().is_relative_to(directory.resolve())
        for directory in (reference.data.split_manifest_path.parent, output_dir)
    ):
        raise ValueError("The new config must be outside both dataset directories.")
    manifest = serialize_sft_name_first(
        dataset_dir=reference.data.split_manifest_path.parent, output_dir=output_dir
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
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/sft/v1/xlam-baymax-target-compatible-v14-act200-anchored-val200-name-first"
        ),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json"
        ),
    )
    args = parser.parse_args(argv)
    try:
        config = prepare_experiment(**vars(args))
    except (ValidationError, json.JSONDecodeError):
        print("error: Invalid source/config JSON structure; no malformed data was repaired.")
        return 1
    except (ValueError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(f"Dataset: {args.output_dir}")
    print(f"Config: {args.config_path}")
    print(f"Revision: {config.data.revision}")
    print(f"Config hash: {config.config_hash}")
    print("Only assistant call-key order changed. Inputs, answers, order and splits preserved.")
    print("Validate the dataset and inspect tokenized artifact counts before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
