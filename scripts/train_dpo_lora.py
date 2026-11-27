"""Continue the frozen two-epoch SFT adapter with optional TRL DPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baymax.models.dpo_training import DPOTrainingOverrides, run_dpo_training
from baymax.models.sft_training import TrainingDependencyError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dpo/qwen2_5_1_5b_anchored_epochs2_v1.json")
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate without ML imports/model loading."
    )
    parser.add_argument(
        "--reviewed", action="store_true", help="Acknowledge semantic review of preference pairs."
    )
    parser.add_argument(
        "--max-pairs", type=int, help="Use a deterministic prefix for a memory smoke test."
    )
    parser.add_argument(
        "--max-steps", type=int, help="Limit optimizer steps for a memory smoke test."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    try:
        result = run_dpo_training(
            config_path=args.config,
            overrides=DPOTrainingOverrides(
                dry_run=args.dry_run,
                reviewed=args.reviewed,
                max_pairs=args.max_pairs,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                allow_cpu=args.allow_cpu,
            ),
        )
    except (ValueError, OSError, KeyError, TrainingDependencyError) as error:
        print(f"error: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
