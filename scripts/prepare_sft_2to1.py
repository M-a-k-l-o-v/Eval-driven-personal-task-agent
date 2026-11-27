"""Prepare a separate two-epoch 2:1 SFT mixture; do not train or load a model."""

from __future__ import annotations

import argparse
from pathlib import Path

from baymax.models.sft_mix import SFTMixConfig, mix_sft_datasets
from baymax.models.training_config import (
    SFTTrainingConfig,
    load_training_config,
    training_config_hash,
    training_config_with_hash,
    write_training_config,
)


def prepare_experiment(
    *,
    reference_config: Path,
    xlam_dir: Path,
    baymax_dir: Path,
    output_dir: Path,
    config_path: Path,
) -> SFTTrainingConfig:
    reference = load_training_config(reference_config)
    if reference.config_hash != training_config_hash(reference):
        raise ValueError("The reference config hash does not match its contents.")
    if reference.training.num_train_epochs != 2:
        raise ValueError("The reference must use two epochs.")
    if config_path.exists():
        raise ValueError("Use a new config path; existing experiment configs are not overwritten.")
    name = config_path.stem
    model_dir = Path("models") / name
    if model_dir.exists():
        raise ValueError("Use a new experiment name; a model output directory already exists.")
    manifest = mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=output_dir,
            seed=reference.runtime.seed,
            reference_mix_dir=reference.data.train_path.parent,
            rebalance_reference_train=True,
            train_cap_row_count=1000,
            train_cap_baymax_row_count=534,
            train_cap_baymax_clarification_count=164,
            train_cap_baymax_refusal_count=170,
            validation_cap_row_count=200,
        )
    )
    payload = reference.model_dump(mode="json")
    payload["config_name"] = name
    payload["data"].update(
        {
            "dataset_id": manifest.dataset_id,
            "revision": manifest.revision,
            "format_version": manifest.format_version,
            "system_prompt_sha256": manifest.system_prompt_sha256,
            "split_manifest_path": str(output_dir / "split-manifest.json"),
            "validation_report_path": str(output_dir / "validation-report.json"),
            **{
                f"{split}_path": str(output_dir / f"{split}.jsonl")
                for split in ("train", "validation", "test")
            },
        }
    )
    payload["runtime"].update(
        {
            "run_name": name,
            "output_dir": str(model_dir),
            "expected_adapter_path": str(model_dir / "adapter"),
        }
    )
    config = training_config_with_hash(SFTTrainingConfig.model_validate(payload))
    write_training_config(config_path, config)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json"
        ),
    )
    parser.add_argument(
        "--xlam-dir",
        type=Path,
        default=Path("data/sft/v1/xlam-content-groups-target-compatible-v1"),
    )
    parser.add_argument(
        "--baymax-dir", type=Path, default=Path("data/sft/v1/baymax-control-policy-v14")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sft/v1/xlam-baymax-target-compatible-v14-2to1-anchored-val200"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_2to1_anchored_val200_epochs2.json"
        ),
    )
    args = parser.parse_args()
    try:
        config = prepare_experiment(**vars(args))
    except (ValueError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(f"Dataset: {args.output_dir}")
    print(f"Config: {args.config_path}")
    print(f"Revision: {config.data.revision}")
    print(f"Config hash: {config.config_hash}")
    print("Raw 1000-row cap: ACT=666, CLARIFY=164, REFUSE=170 (approximately 2:1).")
    print("Validation/test preserved. Verify post-tokenization counts before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
