"""Create a versioned SFT artifact from reviewed credential or target exclusions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from baymax.models.sft_filter import filter_sft_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = filter_sft_dataset(
            dataset_dir=args.dataset_dir,
            audit_path=args.audit,
            review_path=args.review,
            output_dir=args.output_dir,
        )
    except (ValidationError, json.JSONDecodeError):
        print("error: Invalid audit, review, manifest or selected-row structure.", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Inputs may contain credentials; structural failures must not echo payloads.
        message = (
            str(error) if isinstance(error, (OSError, ValueError)) else "Invalid audit structure."
        )
        print(f"error: {message}", file=sys.stderr)
        return 1
    print(f"Wrote reviewed dataset to {args.output_dir}")
    print(
        "Retained: " + ", ".join(f"{name}={entry.count}" for name, entry in manifest.splits.items())
    )
    print(f"Revision: {manifest.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
