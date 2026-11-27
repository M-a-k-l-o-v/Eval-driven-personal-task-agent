"""Build reviewable DPO preferences from the anchored SFT training data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baymax.models.dpo_dataset import build_dpo_dataset


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
        "--baymax-dir", type=Path, default=Path("data/sft/v1/baymax-control-policy-v14")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/dpo/v1/anchored-epochs2-preferences")
    )
    parser.add_argument("--train-cap", type=int, default=1000)
    parser.add_argument(
        "--complete-act-rejections",
        action="store_true",
        help="Fill supported missing ACT negatives with authored training-only wrong actions.",
    )
    args = parser.parse_args()
    try:
        report = build_dpo_dataset(**vars(args))
    except (ValueError, OSError) as error:
        print(f"error: {error}")
        return 1
    print(f"Preference dataset: {args.output_dir}")
    print(f"Pairs: {report['pair_count']}; unique prompts: {report['unique_prompts']}")
    print("Chosen classes:", json.dumps(report["chosen_counts"], sort_keys=True))
    print("Unique anchors:", json.dumps(report["unique_anchor_counts"], sort_keys=True))
    print("Preference directions:", json.dumps(report["direction_counts"], sort_keys=True))
    print("Uncovered ACT rejections:", json.dumps(report["missing_counterparts"], sort_keys=True))
    print("Authored ACT rejections:", report["authored_act_rejection_count"])
    print("Review pairs-review.jsonl before training. No model was loaded or trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
