"""Measure assistant-token loss by decision class on an existing SFT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from baymax.eval.lora_runner import _load_local_lora_dependencies
from baymax.models.sft_artifact_report import classify_sft_behavior
from baymax.models.sft_mix import _file_sha256
from baymax.models.sft_token_spans import COMPONENTS, token_components
from baymax.models.sft_token_spans import target_component_spans as target_component_spans
from baymax.models.sft_training import (
    _tokenize_chat_example,
    load_sft_jsonl,
    load_split_source_rows,
)
from baymax.models.sft_validation import validate_training_targets
from baymax.models.training_config import (
    SFTTrainingConfig,
    load_training_config,
    validate_training_config,
)

DEFAULT_CONFIG = Path(
    "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json"
)
CLASSES = ("act", "clarify", "refuse")


def component_losses(
    logits: Any, tokenized: dict[str, Any], parts: list[str | None], torch: Any
) -> dict[str, Any]:
    positions = [
        index for index, label in enumerate(tokenized["labels"]) if index and label != -100
    ]
    if len(parts) != len(tokenized["labels"]):
        raise ValueError("Component labels are not aligned with the training labels.")
    losses: dict[str, list[float]] = {part: [] for part in COMPONENTS}
    # Select only supervised logits, in small chunks to limit temporary FP32 memory on 6GB GPUs.
    for start in range(0, len(positions), 64):
        chunk = positions[start : start + 64]
        preceding = torch.tensor(
            [index - 1 for index in chunk], dtype=torch.long, device=logits.device
        )
        targets = torch.tensor(
            [tokenized["labels"][index] for index in chunk], dtype=torch.long, device=logits.device
        )
        values = torch.nn.functional.cross_entropy(
            logits[0].index_select(0, preceding).float(), targets, reduction="none"
        ).tolist()
        for index, value in zip(chunk, values, strict=True):
            part = parts[index]
            if part not in losses or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid component label or non-finite/negative token loss.")
            losses[part].append(value)
    return {
        part: {
            "scored_tokens": len(values),
            "loss": math.fsum(values) / len(values) if values else None,
        }
        for part, values in losses.items()
    }


def summarize_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        part: summarize_losses(
            [{**row["components"][part], "truncated": row["truncated"]} for row in rows]
        )
        for part in COMPONENTS
    }


def decision_class(row: dict[str, Any]) -> str:
    behavior = classify_sft_behavior(row)
    if behavior in ("direct_tool_call", "multi_tool_sequence"):
        return "act"
    if behavior == "request_clarification":
        return "clarify"
    if behavior == "refuse_request":
        return "refuse"
    raise ValueError("Cannot score a malformed assistant target.")


def summarize_losses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(row["scored_tokens"] for row in rows)
    scored = [row for row in rows if row["loss"] is not None]
    total_nll = math.fsum(row["loss"] * row["scored_tokens"] for row in scored)
    return {
        "loaded_rows": len(rows),
        "scored_rows": len(scored),
        "skipped_rows": len(rows) - len(scored),
        "truncated_rows": sum(row["truncated"] for row in rows),
        "scored_tokens": tokens,
        "total_negative_log_likelihood": total_nll,
        "token_mean_loss": total_nll / tokens if tokens else None,
        "row_mean_loss": math.fsum(row["loss"] for row in scored) / len(scored) if scored else None,
    }


def score_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    torch: Any,
    max_sequence_length: int,
    by_component: bool = False,
) -> dict[str, Any]:
    """Use the training tokenizer and the model's built-in shifted causal-LM loss."""
    model.eval()
    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            label = decision_class(row)
            tokenized = _tokenize_chat_example(row, tokenizer, max_sequence_length)
            full_text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            full_length = len(
                tokenizer(full_text, truncation=False, add_special_tokens=False)["input_ids"]
            )
            # Position zero has no preceding logit; -100 prompt labels never enter the loss.
            count = sum(token != -100 for token in tokenized["labels"][1:])
            loss = None
            components = {part: {"scored_tokens": 0, "loss": None} for part in COMPONENTS}
            parts = []
            if by_component:
                try:
                    parts = token_components(row, tokenizer, tokenized, max_sequence_length)
                except ValueError as error:
                    raise ValueError(f"Row {number}: {error}") from error
            if count:
                batch = {
                    key: torch.tensor([values], dtype=torch.long, device=device)
                    for key, values in tokenized.items()
                }
                model_output = model(**batch, use_cache=False, return_dict=True)
                loss = float(model_output.loss.item())
                if not math.isfinite(loss) or loss < 0:
                    raise ValueError(f"Row {number}: non-finite or negative loss; report aborted.")
                if by_component:
                    components = component_losses(model_output.logits, tokenized, parts, torch)
                    mean = summarize_losses(
                        [{**stats, "truncated": False} for stats in components.values()]
                    )["token_mean_loss"]
                    if not math.isclose(mean, loss, rel_tol=1e-4, abs_tol=1e-5):
                        raise ValueError(f"Row {number}: component losses do not match model loss.")
                del model_output
            results.append(
                {
                    "file_line": number,
                    "decision_class": label,
                    "behavior_class": classify_sft_behavior(row),
                    "scored_tokens": count,
                    "truncated": full_length > max_sequence_length,
                    "loss": loss,
                }
            )
            if by_component:
                results[-1]["components"] = components
            if number % 100 == 0 or number == len(rows):
                print(f"Processed {number}/{len(rows)} rows.", flush=True)
    if not any(row["scored_tokens"] for row in results):
        raise ValueError("No assistant tokens remain after masking and truncation.")
    report: dict[str, Any] = {
        "overall": summarize_losses(results),
        "by_class": {
            label: summarize_losses([row for row in results if row["decision_class"] == label])
            for label in CLASSES
        },
        "rows": results,
    }
    if by_component:
        report["by_component"] = summarize_components(results)
        for label, stats in report["by_class"].items():
            stats["components"] = summarize_components(
                [row for row in results if row["decision_class"] == label]
            )
    return report


def _load_model(config: SFTTrainingConfig, *, allow_cpu: bool) -> tuple[Any, Any, Any]:
    dependencies = _load_local_lora_dependencies(include_peft=True)
    torch, transformers, peft = (dependencies[name] for name in ("torch", "transformers", "peft"))
    cuda = torch.cuda.is_available()
    if not cuda and not allow_cpu:
        raise RuntimeError("CUDA is not available. Use --allow-cpu only for a slow CPU diagnostic.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.base_model, trust_remote_code=False
    )
    kwargs: dict[str, Any] = {"trust_remote_code": False}
    if cuda:
        kwargs.update(
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    model = peft.PeftModel.from_pretrained(
        model, config.runtime.expected_adapter_path, is_trainable=False
    )
    model.requires_grad_(False)
    return torch, model, tokenizer


def run_report(
    *,
    config_path: Path,
    output: Path,
    split: Literal["train", "validation"] = "train",
    max_rows: int = 1000,
    max_sequence_length: int = 1024,
    dry_run: bool = False,
    allow_cpu: bool = False,
    by_component: bool = False,
) -> dict[str, Any]:
    if split not in ("train", "validation") or min(max_rows, max_sequence_length) < 1:
        raise ValueError("Use a train/validation split and positive row/context caps.")
    if output.exists():
        raise ValueError("Output already exists; choose a new path to preserve prior reports.")
    validation = validate_training_config(config_path, check_paths=True)
    if not validation.valid:
        raise ValueError("Training config/hash/manifest validation failed.")
    config = load_training_config(config_path)
    adapter = config.runtime.expected_adapter_path
    for directory in (
        config.data.train_path.parent,
        config.data.validation_path.parent,
        config.data.test_path.parent,
        config.runtime.output_dir,
        adapter,
    ):
        if output.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Report output must be outside dataset and model directories.")
    path = config.data.train_path if split == "train" else config.data.validation_path
    rows = load_sft_jsonl(path, max_rows=max_rows)
    if len(rows) != max_rows:
        raise ValueError("The selected split is shorter than the requested row cap.")
    validate_training_targets(rows, path=path)
    for number, row in enumerate(rows, 1):
        if hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest() != (
            config.data.system_prompt_sha256
        ):
            raise ValueError(f"Row {number}: system prompt hash mismatch.")
    refs = load_split_source_rows(
        config.data.split_manifest_path, split_name=split, max_rows=max_rows
    )
    if len(refs) != len(rows):
        raise ValueError("Selected rows and manifest source references differ in count.")
    metadata = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    modules = metadata.get("target_modules") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or (
        metadata.get("base_model_name_or_path") != config.base_model
        or metadata.get("peft_type") != "LORA"
        or metadata.get("r") != config.lora.r
        or metadata.get("lora_alpha") != config.lora.lora_alpha
        or not isinstance(modules, list)
        or not all(isinstance(module, str) for module in modules)
        or set(modules) != set(config.lora.target_modules)
    ):
        raise ValueError("Adapter metadata does not match the selected training config.")
    report: dict[str, Any] = {
        "diagnostic": "sft-class-component-loss-v1" if by_component else "sft-class-loss-v1",
        "measured": not dry_run,
        "config_path": str(config_path),
        "training_config_hash": config.config_hash,
        "base_model": config.base_model,
        "dataset_revision": config.data.revision,
        "adapter_path": str(adapter),
        "split": split,
        "max_rows": max_rows,
        "max_sequence_length": max_sequence_length,
        "input_presentation": "sft-exact",
        "raw_class_counts": dict(Counter(decision_class(row) for row in rows)),
        "sha256": {
            "config": _file_sha256(config_path),
            "manifest": _file_sha256(config.data.split_manifest_path),
            "split_file": _file_sha256(path),
            "adapter_config": _file_sha256(adapter / "adapter_config.json"),
            "adapter_weights": _file_sha256(adapter / "adapter_model.safetensors"),
        },
        "loss_definition": (
            "Teacher-forced causal cross-entropy in nats on shifted non--100 assistant labels, "
            "including visible chat-ending tokens; token mean weights each row by scored tokens. "
            "No generation, training, decision accuracy or question-quality scoring."
        ),
    }
    if by_component:
        report["component_definition"] = (
            "Character offsets on unchanged assistant JSON: first/later call-name values, "
            "question/reason string values, other tokens (including syntax, ACT arguments and "
            "chat endings), and separately reported mixed-boundary tokens. Groups partition "
            "all scored tokens. Later names and text are conditioned on preceding gold tokens; "
            "these losses are not free-generation routing probabilities."
        )
    if not dry_run:
        torch, model, tokenizer = _load_model(config, allow_cpu=allow_cpu)
        report.update(
            score_rows(
                rows,
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                max_sequence_length=max_sequence_length,
                by_component=by_component,
            )
        )
        for row, ref in zip(report["rows"], refs, strict=True):
            row["source"] = ref.model_dump(mode="json")
        report["versions"] = {name: version(name) for name in ("torch", "transformers", "peft")}
        report["model_revision"] = getattr(model.config, "_commit_hash", None)
        report["model_dtype"] = str(next(model.parameters()).dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument(
        "--dry-run", action="store_true", help="Preflight only; no model/tokenizer."
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--by-component",
        action="store_true",
        help="Separate names, meta-tool text and other-token losses.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_report(
            config_path=args.config,
            **{key: value for key, value in vars(args).items() if key != "config"},
        )
    except (OSError, ValueError, RuntimeError, ValidationError) as error:
        print(f"error: {error}")
        return 1
    print(f"Report: {args.output}")
    if args.dry_run:
        print(f"Raw class counts: {report['raw_class_counts']}")
        print("Dry run only; no model/tokenizer loaded and no loss measured.")
    else:
        print("class       rows  skipped  truncated  scored tokens  loss/token  mean row loss")
        for label, stats in report["by_class"].items():
            loss = stats["token_mean_loss"]
            row_loss = stats["row_mean_loss"]
            values = "n/a         n/a" if loss is None else f"{loss:.6f}    {row_loss:.6f}"
            print(
                f"{label:<11} {stats['scored_rows']:>4}  {stats['skipped_rows']:>7}  "
                f"{stats['truncated_rows']:>9}  {stats['scored_tokens']:>13}  {values}"
            )
        print("Teacher-forced answer loss only, not routing accuracy or benchmark success.")
        if args.by_component:
            print("\nclass       component          rows  scored tokens  loss/token")
            for label, stats in report["by_class"].items():
                for part, values in stats["components"].items():
                    if values["scored_tokens"]:
                        print(
                            f"{label:<11} {part:<18} {values['scored_rows']:>4}  "
                            f"{values['scored_tokens']:>13}  {values['token_mean_loss']:.6f}"
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
