"""Mix external xLAM SFT data with synthetic BAYMAX-control SFT data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from baymax.models.sft_mix import SFTMixConfig, mix_sft_datasets

DEFAULT_XLAM_DIR = Path("data/sft/v1/xlam")
DEFAULT_BAYMAX_DIR = Path("data/sft/v1/baymax-control")
DEFAULT_OUTPUT_DIR = Path("data/sft/v1/xlam-baymax-mix")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xlam-dir",
        type=Path,
        default=DEFAULT_XLAM_DIR,
        help="Existing xLAM SFT dataset directory.",
    )
    parser.add_argument(
        "--baymax-dir",
        type=Path,
        default=DEFAULT_BAYMAX_DIR,
        help="Existing synthetic BAYMAX-control SFT dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the mixed SFT dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed used for deterministic output shuffling.",
    )
    parser.add_argument(
        "--train-baymax-repeat-count",
        type=int,
        default=3,
        help="How many times to include BAYMAX train rows in the mixed train split.",
    )
    parser.add_argument(
        "--train-baymax-boundary-extra-repeat-count",
        type=int,
        default=2,
        help="Extra train-only repeats for BAYMAX clarification/refusal rows.",
    )
    parser.add_argument(
        "--eval-baymax-repeat-count",
        type=int,
        default=1,
        help="How many times to include BAYMAX validation/test rows.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle rows inside each output split.",
    )
    args = parser.parse_args()

    try:
        config = SFTMixConfig(
            xlam_dir=args.xlam_dir,
            baymax_dir=args.baymax_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            train_baymax_repeat_count=args.train_baymax_repeat_count,
            train_baymax_boundary_extra_repeat_count=(
                args.train_baymax_boundary_extra_repeat_count
            ),
            eval_baymax_repeat_count=args.eval_baymax_repeat_count,
            shuffle=not args.no_shuffle,
        )
    except ValidationError as error:
        print(f"error: invalid SFT mix config: {error}", file=sys.stderr)
        return 1

    try:
        manifest = mix_sft_datasets(config)
    except FileNotFoundError as error:
        print(f"error: missing source dataset file: {error}", file=sys.stderr)
        return 1

    print(f"Wrote mixed SFT dataset to {config.output_dir}")
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
