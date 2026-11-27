"""Prepare the unique 2k/two-epoch experiment without loading a model or tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baymax.models.sft_dataset import SFTSplitManifest
from baymax.models.sft_expansion import expand_training_dataset
from baymax.models.training_config import (
    SFTTrainingConfig,
    load_training_config,
    training_config_with_hash,
    validate_training_config,
    write_training_config,
)

DEFAULT_REFERENCE = Path("configs/sft/qwen2_5_1_5b_lora_grounding_v1_epochs2.json")


def prepare_experiment(
    *, reference_config: Path, output_dir: Path, config_path: Path, train_cap: int = 1000
) -> SFTTrainingConfig:
    if not validate_training_config(reference_config, check_paths=True).valid:
        raise ValueError("Reference config, hash, paths or manifest is invalid.")
    reference = load_training_config(reference_config)
    if reference.training.first_tool_name_weight != 1 or reference.training.num_train_epochs != 2:
        raise ValueError("Use the unweighted two-epoch reference.")
    parent = SFTSplitManifest.model_validate_json(reference.data.split_manifest_path.read_bytes())
    if parent.seed != reference.runtime.seed:
        raise ValueError("Dataset and training seeds must match.")
    model_dir = Path("models") / config_path.stem
    if config_path.exists() or output_dir.exists() or model_dir.exists():
        raise ValueError("Dataset, config and model destinations must be new.")
    for directory in (reference.data.train_path.parent, output_dir):
        if config_path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Config must be outside dataset directories.")
    manifest = expand_training_dataset(
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
        "--output-dir", type=Path, default=Path("data/sft/v1/name-first-coverage-2k-v2")
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/sft/qwen2_5_1_5b_lora_coverage_2k_v2_epochs2.json"),
    )
    args = parser.parse_args(argv)
    try:
        config = prepare_experiment(**vars(args))
    except (ValueError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(f"Dataset: {args.output_dir}")
    print(f"Config: {args.config_path}")
    print(f"Revision: {config.data.revision}")
    print(f"Config hash: {config.config_hash}")
    print("Raw cap: 2000 = 800 BAYMAX (400 ACT / 200 CLARIFY / 200 REFUSE) + 1200 xLAM.")
    print("Original 1000-row prefix and full validation/test bytes retained. No training was run.")
    report = json.loads((args.output_dir / "expansion-report.json").read_text(encoding="utf-8"))
    print(f"Excluded unused parent-tail input overlaps: {len(report['excluded_parent_tail'])}")
    print(
        "Review expansion-sources.jsonl and expansion-report.json, "
        "validate, then inspect tokenized counts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
