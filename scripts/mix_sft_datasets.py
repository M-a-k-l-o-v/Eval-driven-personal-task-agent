"""Mix external xLAM SFT data with synthetic BAYMAX-control SFT data."""

from __future__ import annotations

import argparse
import json
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
        default=1,
        help="How many times to include BAYMAX train rows in the mixed train split.",
    )
    parser.add_argument(
        "--train-baymax-clarification-extra-repeat-count",
        type=int,
        default=0,
        help="Extra train-only repeats for BAYMAX clarification rows.",
    )
    parser.add_argument(
        "--train-baymax-refusal-extra-repeat-count",
        type=int,
        default=0,
        help="Extra train-only repeats for BAYMAX refusal rows.",
    )
    parser.add_argument(
        "--eval-baymax-repeat-count",
        type=int,
        default=1,
        help="How many times to include BAYMAX validation/test rows.",
    )
    parser.add_argument(
        "--train-cap-row-count",
        type=int,
        default=None,
        help="Move a source/class-aware train cap slice to the front of train.jsonl.",
    )
    parser.add_argument(
        "--train-cap-baymax-row-count",
        type=int,
        default=400,
        help="Target BAYMAX rows inside the cap-aware train prefix.",
    )
    parser.add_argument(
        "--train-cap-baymax-clarification-count",
        type=int,
        default=100,
        help="Target BAYMAX clarification rows inside the cap-aware train prefix.",
    )
    parser.add_argument(
        "--train-cap-baymax-refusal-count",
        type=int,
        default=100,
        help="Target BAYMAX refusal rows inside the cap-aware train prefix.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Do not shuffle rows inside each output split.",
    )
    parser.add_argument(
        "--validation-cap-row-count",
        type=int,
        default=None,
        help="Reorder validation only to make a source/class-aware cap prefix.",
    )
    parser.add_argument("--validation-cap-baymax-row-count", type=int, default=90)
    parser.add_argument("--validation-cap-baymax-clarification-count", type=int, default=30)
    parser.add_argument("--validation-cap-baymax-refusal-count", type=int, default=30)
    parser.add_argument(
        "--preserve-contrast-groups",
        action="store_true",
        help="Select complete BAYMAX contrast groups within the existing class quotas.",
    )
    parser.add_argument(
        "--reference-mix-dir",
        type=Path,
        default=None,
        help="Anchor caps to a previous mix; replace only reviewed xLAM exclusions.",
    )
    args = parser.parse_args()

    try:
        config = SFTMixConfig(
            xlam_dir=args.xlam_dir,
            baymax_dir=args.baymax_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            preserve_contrast_groups=args.preserve_contrast_groups,
            reference_mix_dir=args.reference_mix_dir,
            train_baymax_repeat_count=args.train_baymax_repeat_count,
            train_baymax_clarification_extra_repeat_count=(
                args.train_baymax_clarification_extra_repeat_count
            ),
            train_baymax_refusal_extra_repeat_count=(args.train_baymax_refusal_extra_repeat_count),
            eval_baymax_repeat_count=args.eval_baymax_repeat_count,
            shuffle=not args.no_shuffle,
            train_cap_row_count=args.train_cap_row_count,
            train_cap_baymax_row_count=args.train_cap_baymax_row_count,
            train_cap_baymax_clarification_count=args.train_cap_baymax_clarification_count,
            train_cap_baymax_refusal_count=args.train_cap_baymax_refusal_count,
            validation_cap_row_count=args.validation_cap_row_count,
            validation_cap_baymax_row_count=args.validation_cap_baymax_row_count,
            validation_cap_baymax_clarification_count=args.validation_cap_baymax_clarification_count,
            validation_cap_baymax_refusal_count=args.validation_cap_baymax_refusal_count,
        )
    except ValidationError as error:
        print(f"error: invalid SFT mix config: {error}", file=sys.stderr)
        return 1

    try:
        manifest = mix_sft_datasets(config)
    except FileNotFoundError as error:
        print(f"error: missing source dataset file: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
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
    if config.reference_mix_dir is not None:
        report_path = config.output_dir / "sampling-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for split, counts in report["splits"].items():
            print(
                f"Reference {split}: kept BAYMAX={counts['kept_baymax']}, "
                f"kept xLAM={counts['kept_xlam']}, replaced xLAM={counts['replaced_xlam']}"
            )
        print(f"Sampling report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
