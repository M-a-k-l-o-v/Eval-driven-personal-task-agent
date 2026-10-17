"""Diagnose SFT tokenization health before LoRA training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from baymax.models.sft_training import (
    _tokenize_chat_example,
    diagnose_sft_tokenization,
    load_sft_jsonl,
)
from baymax.models.training_config import SFTTrainingConfig, load_training_config

DEFAULT_CONFIG_PATH = Path("configs/sft/qwen2_5_1_5b_lora_v1.json")
SplitName = Literal["train", "validation", "test"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Training config JSON path.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="validation",
        help="SFT split to diagnose.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows for quick diagnostics.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Override sequence length for this diagnostic.",
    )
    parser.add_argument(
        "--problem-row-limit",
        type=int,
        default=20,
        help="Maximum problem rows to include in the report.",
    )
    parser.add_argument(
        "--inspect-row",
        type=int,
        action="append",
        default=[],
        help="Preview actual labels for a zero-based loaded row index; repeat for multiple rows.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full diagnostic report.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("error: missing transformers. Install training dependencies first.")
        return 1

    config = load_training_config(args.config)
    split_path = _split_path(config_path=args.split, config=config)
    rows = load_sft_jsonl(split_path, max_rows=args.max_rows)
    if any(index < 0 or index >= len(rows) for index in args.inspect_row):
        parser.error("--inspect-row must be a zero-based index within the loaded row cap")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    max_sequence_length = args.max_sequence_length or config.training.max_sequence_length
    diagnostics = diagnose_sft_tokenization(
        rows=rows,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        max_problem_rows=args.problem_row_limit,
    )
    previews = [
        {"row_index": index, **inspect_training_labels(rows[index], tokenizer, max_sequence_length)}
        for index in args.inspect_row
    ]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = diagnostics.model_dump()
        if previews:
            payload["label_previews"] = previews
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Split: {args.split}")
    print(f"Rows checked: {diagnostics.row_count}")
    print(f"Max sequence length: {diagnostics.max_sequence_length}")
    print(f"Rows with trainable labels: {diagnostics.rows_with_trainable_labels}")
    print(f"Rows without trainable labels: {diagnostics.rows_without_trainable_labels}")
    print(f"Truncated rows: {diagnostics.truncated_rows}")
    print(f"Prompt fills context rows: {diagnostics.prompt_fills_context_rows}")
    print(f"Total trainable label tokens: {diagnostics.total_trainable_label_tokens}")
    print(f"Min trainable label tokens: {diagnostics.min_trainable_label_tokens}")
    print(f"Max trainable label tokens: {diagnostics.max_trainable_label_tokens}")
    if diagnostics.problem_rows:
        print("Problem row examples:")
        for row in diagnostics.problem_rows:
            print(
                f"- row {row.row_index}: {', '.join(row.issues)} "
                f"({row.trainable_label_tokens} trainable labels)"
            )
    if previews:
        print("Actual label previews (zero-based indices; -100 labels excluded from decoded text):")
        print(json.dumps(previews, indent=2))

    return 0


def inspect_training_labels(
    row: dict[str, Any], tokenizer: Any, max_sequence_length: int
) -> dict[str, Any]:
    """Read actual training labels; independently check the rendered prompt boundary."""
    tokenized = _tokenize_chat_example(row, tokenizer, max_sequence_length)
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, truncation=False, add_special_tokens=False)["input_ids"]
    labels, input_ids = tokenized["labels"], tokenized["input_ids"]
    boundary = min(len(prompt_ids), len(labels))
    decoded = tokenizer.decode(
        [label for label in labels if label != -100],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    target = messages[-1]["content"]
    return {
        "expected_target": target,
        "decoded_trainable_labels": decoded,
        "prompt_prefix_matches": full_ids[: len(prompt_ids)] == prompt_ids,
        "prompt_tokens_masked": all(label == -100 for label in labels[:boundary]),
        "visible_assistant_labels_match_inputs": bool(labels[boundary:])
        and labels[boundary:] == input_ids[boundary:],
        "complete_target_visible": decoded.startswith(target),
    }


def _split_path(*, config_path: SplitName, config: SFTTrainingConfig) -> Path:
    if config_path == "train":
        return config.data.train_path
    if config_path == "validation":
        return config.data.validation_path

    return config.data.test_path


if __name__ == "__main__":
    raise SystemExit(main())
