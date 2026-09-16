"""Build synthetic BAYMAX-control SFT JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from baymax.models.baymax_sft import BaymaxSFTBuildConfig, build_baymax_sft_dataset
from baymax.models.sft_dataset import SFTSplitRatios

DEFAULT_OUTPUT_DIR = Path("data/sft/v1/baymax-control")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for train.jsonl, validation.jsonl, test.jsonl, and split-manifest.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed used for deterministic split assignment.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    args = parser.parse_args()

    try:
        config = BaymaxSFTBuildConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            split_ratios=SFTSplitRatios(
                train=args.train_ratio,
                validation=args.validation_ratio,
                test=args.test_ratio,
            ),
        )
    except ValidationError as error:
        print(f"error: invalid BAYMAX SFT build config: {error}", file=sys.stderr)
        return 1

    manifest = build_baymax_sft_dataset(config)
    print(f"Wrote BAYMAX SFT dataset to {config.output_dir}")
    print(
        "Counts: "
        f"train={manifest.splits['train'].count}, "
        f"validation={manifest.splits['validation'].count}, "
        f"test={manifest.splits['test'].count}"
    )
    print(f"Config hash: {manifest.config_hash}")
    print(f"Revision: {manifest.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
