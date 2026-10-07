"""Report the effective SFT artifact after split/shuffle/cap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baymax.models.sft_artifact_report import SFTArtifactReport
from baymax.models.sft_training import load_sft_jsonl, load_split_source_rows
from baymax.models.training_config import load_training_config

DEFAULT_CONFIG_PATH = Path("configs/sft/qwen2_5_1_5b_lora_baymax_mix_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("error: missing transformers. Install training dependencies first.", file=sys.stderr)
        return 1

    config = load_training_config(args.config)
    split_path = {
        "train": config.data.train_path,
        "validation": config.data.validation_path,
        "test": config.data.test_path,
    }[args.split]
    max_sequence_length = args.max_sequence_length or config.training.max_sequence_length

    try:
        rows = load_sft_jsonl(split_path, max_rows=args.max_rows)
        source_rows = load_split_source_rows(
            config.data.split_manifest_path,
            split_name=args.split,
            max_rows=args.max_rows,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=args.trust_remote_code,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    from baymax.models.sft_artifact_report import build_sft_artifact_report

    report = build_sft_artifact_report(
        rows=rows,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        source_rows=source_rows,
    )

    print(f"Split: {args.split}")
    print(f"Rows loaded before trainable-label filtering: {len(rows)}")
    print(f"Max sequence length: {max_sequence_length}")
    print()
    print("Final training artifact" if args.split == "train" else "Final SFT artifact")
    print(format_artifact_report(report))
    return 0


def format_artifact_report(report: SFTArtifactReport) -> str:
    lines = [
        _format_table(
            ["class", "rows", "row %", "trainable tokens", "token %"],
            [
                [
                    row.behavior_class,
                    str(row.rows),
                    _format_percent(row.row_percent),
                    str(row.trainable_tokens),
                    _format_percent(row.token_percent),
                ]
                for row in report.behavior
            ],
        ),
        "",
        "source:",
        _format_table(
            ["source", "rows", "row %"],
            [
                [row.source, str(row.rows), _format_percent(row.row_percent)]
                for row in report.source
            ],
        ),
        "",
        "duplicates:",
        f"exact duplicate rows: {report.duplicates.exact_duplicate_rows}",
        f"unique rows: {report.duplicates.unique_rows}",
        "",
        "duplicates by class:",
        _format_table(
            ["class", "exact duplicate rows", "unique rows"],
            [
                [
                    row.behavior_class,
                    str(row.exact_duplicate_rows),
                    str(row.unique_rows),
                ]
                for row in report.duplicates_by_behavior
            ],
        ),
    ]
    return "\n".join(lines)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in [headers, *rows]) for index in range(len(headers))]
    lines = [
        "  " + "  ".join(_pad(value, widths[index]) for index, value in enumerate(headers)),
        "  " + "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  " + "  ".join(_pad(value, widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _pad(value: str, width: int) -> str:
    return value + (" " * (width - len(value)))


if __name__ == "__main__":
    raise SystemExit(main())
