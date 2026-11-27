"""Probe SFT behaviors without training, repair or tool execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from baymax.eval.lora_runner import LocalLoraAgent, LocalLoraConfig, parse_lora_agent_response
from baymax.models.sft_dataset import SFTExample
from baymax.models.sft_mix import _file_sha256, _sft_input_hash
from baymax.models.sft_training import load_sft_jsonl, load_split_source_rows
from baymax.models.sft_validation import validate_training_targets
from baymax.models.training_config import load_training_config, validate_training_config

DEFAULT_CONFIG = Path(
    "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json"
)
BEHAVIOR_LABELS = {"clarify": "clarification", "refuse": "refusal", "multi-tool": "multi-tool"}


def benchmark_json_example(example: SFTExample) -> SFTExample:
    """Use eval's JSON serialization without changing tool/context values or time spelling."""
    header, separator, context_text = example.messages[1].content.partition("\n\nUser request:\n")
    title, newline, tools_text = header.partition("\n")
    if not separator or not newline:
        raise ValueError("Cannot locate the SFT tool list and request wrapper.")
    tools, context = json.loads(tools_text), json.loads(context_text)
    for text, value in ((tools_text, tools), (context_text, context)):
        if text != json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")):
            raise ValueError("JSON-presentation comparison requires canonical compact SFT inputs.")
    content = "\n".join(
        [
            title,
            json.dumps(tools, sort_keys=True),
            "",
            "User request:",
            json.dumps(context, sort_keys=True),
        ]
    )
    return example.model_copy(
        update={
            "messages": [
                example.messages[0],
                example.messages[1].model_copy(update={"content": content}),
                *example.messages[2:],
            ]
        }
    )


def _check_reference(
    path: Path, report: dict[str, Any], selected: list[tuple[int, SFTExample]]
) -> dict[str, Any]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or not isinstance(reference.get("summary"), dict):
        raise ValueError("Reference must contain a completed probe summary.")
    if (
        not reference.get("generated")
        or reference.get("input_presentation", "sft-exact") != "sft-exact"
    ):
        raise ValueError("Reference must be a completed original-SFT presentation probe.")
    if reference.get("behavior", "clarify") != report["behavior"]:
        raise ValueError("Reference behavior differs from the selected probe.")
    for key in (
        "training_config_hash",
        "config_file_sha256",
        "dataset_revision",
        "manifest_sha256",
        "validation_sha256",
        "validation_cap",
        "training_overlap_check_cap",
        "base_model",
        "adapter_path",
        "adapter_config_sha256",
        "adapter_weights_sha256",
        "generation",
    ):
        if reference.get(key) != report[key]:
            raise ValueError(
                f"Reference differs in {key}; this is not a presentation-only comparison."
            )
    reference_rows = reference.get("rows")
    if not isinstance(reference_rows, list) or len(reference_rows) != len(selected):
        raise ValueError("Reference selection differs in count.")
    for row, (number, example) in zip(reference_rows, selected, strict=True):
        expected = {
            "validation_row": number,
            "input_sha256": _sft_input_hash(example),
            "messages": [message.model_dump() for message in example.messages[:-1]],
            "expected_target": example.messages[-1].content,
        }
        if not isinstance(row, dict) or any(
            row.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Reference example inputs, targets or selection order changed.")
    return reference


def run_probe(
    *,
    config_path: Path,
    output: Path,
    limit: int = 10,
    validation_cap: int = 200,
    train_cap: int = 1000,
    dry_run: bool = False,
    allow_cpu: bool = False,
    benchmark_json: bool = False,
    reference_report: Path | None = None,
    training_rows: list[int] | None = None,
    behavior: str = "clarify",
) -> dict[str, Any]:
    if behavior not in BEHAVIOR_LABELS:
        raise ValueError("Behavior must be clarify, refuse or multi-tool.")
    behavior_label = BEHAVIOR_LABELS[behavior]
    expected_decision = "act" if behavior == "multi-tool" else behavior
    if min(limit, validation_cap, train_cap) < 1:
        raise ValueError("Limit and row caps must be positive.")
    if output.exists():
        raise ValueError("Output already exists; use a new path to preserve prior diagnostics.")
    in_sample = training_rows is not None
    if in_sample and behavior != "clarify":
        raise ValueError("In-sample selection supports clarification only.")
    if training_rows is not None:
        if (
            len(training_rows) != limit
            or len(set(training_rows)) != limit
            or any(number < 1 or number > train_cap for number in training_rows)
        ):
            raise ValueError(
                "Training rows must be distinct file lines within the cap, matching --limit."
            )
        if benchmark_json or reference_report is not None:
            raise ValueError(
                "In-sample training probes cannot use presentation comparison/reference options."
            )
    if benchmark_json and reference_report is None:
        raise ValueError(
            "Benchmark JSON comparison requires --reference-report from the SFT probe."
        )
    validation = validate_training_config(config_path)
    if not validation.valid:
        raise ValueError("Training config/hash/manifest validation failed.")
    config = load_training_config(config_path)
    for directory in (
        config.data.train_path.parent,
        config.data.validation_path.parent,
        config.runtime.output_dir,
    ):
        if output.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Diagnostic output must be outside the dataset and model directories.")
    split_name = "train" if in_sample else "validation"
    source_path = config.data.train_path if in_sample else config.data.validation_path
    row_cap = train_cap if in_sample else validation_cap
    rows = load_sft_jsonl(source_path, max_rows=row_cap)
    validate_training_targets(rows, path=source_path)
    refs = load_split_source_rows(
        config.data.split_manifest_path, split_name=split_name, max_rows=row_cap
    )
    if len(refs) != len(rows):
        raise ValueError("Selected split rows and source references differ in count.")
    training_examples = (
        rows if in_sample else load_sft_jsonl(config.data.train_path, max_rows=train_cap)
    )
    if len(training_examples) != train_cap:
        raise ValueError("Training file is shorter than the requested overlap-check cap.")
    training_inputs = {_sft_input_hash(SFTExample.model_validate(row)) for row in training_examples}
    requested_rows = set(training_rows or [])
    row_label = "Training" if in_sample else "Validation"
    selected: list[tuple[int, SFTExample]] = []
    seen: set[str] = set()
    available = 0
    for number, row in enumerate(rows, 1):
        if in_sample and number not in requested_rows:
            continue
        example = SFTExample.model_validate(row)
        if behavior != "clarify" and not str(refs[number - 1].source_row_id).startswith("baymax:"):
            continue
        if hashlib.sha256(example.messages[0].content.encode()).hexdigest() != (
            config.data.system_prompt_sha256
        ):
            raise ValueError(f"{row_label} row {number}: system prompt hash mismatch.")
        expected_response = parse_lora_agent_response(example.messages[-1].content)
        if expected_response.decision_type != expected_decision:
            continue
        if behavior == "multi-tool" and len(expected_response.tool_calls) < 2:
            continue
        digest = _sft_input_hash(example)
        if not in_sample and digest in training_inputs:
            raise ValueError(f"Validation row {number}: {behavior_label} prompt overlaps training.")
        if digest in seen:
            continue
        seen.add(digest)
        available += 1
        if len(selected) < limit:
            selected.append((number, example))
    if len(selected) != limit:
        if in_sample:
            raise ValueError("Every requested training row must be a unique clarification input.")
        raise ValueError(f"Only {available} unique {behavior_label} rows available; need {limit}.")
    adapter = config.runtime.expected_adapter_path
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        adapter_config.get("base_model_name_or_path") != config.base_model
        or adapter_config.get("peft_type") != "LORA"
        or adapter_config.get("r") != config.lora.r
        or adapter_config.get("lora_alpha") != config.lora.lora_alpha
        or set(adapter_config.get("target_modules", [])) != set(config.lora.target_modules)
    ):
        raise ValueError("Adapter metadata does not match the selected training config.")
    report: dict[str, Any] = {
        "diagnostic": "insample-sft-clarification-probe-v1"
        if in_sample
        else f"heldout-sft-{behavior_label}-probe-v1",
        "behavior": behavior,
        "data_split": split_name,
        "generated": not dry_run,
        "config_path": str(config_path),
        "training_config_hash": config.config_hash,
        "config_file_sha256": _file_sha256(config_path),
        "dataset_revision": config.data.revision,
        "manifest_sha256": _file_sha256(config.data.split_manifest_path),
        f"{split_name}_path": str(source_path),
        f"{split_name}_sha256": _file_sha256(source_path),
        f"available_unique_{behavior_label.replace('-', '_')}_rows": available,
        "selection": (
            "Explicit training file lines, in unchanged file order; in-sample diagnostic only."
            if in_sample
            else f"First unique {behavior_label} inputs in unchanged validation order."
        ),
        "source_scope": "baymax" if behavior != "clarify" else "all",
        "base_model": config.base_model,
        "adapter_path": str(adapter),
        "adapter_config_sha256": _file_sha256(adapter / "adapter_config.json"),
        "adapter_weights_sha256": _file_sha256(adapter / "adapter_model.safetensors"),
        "generation": {"max_new_tokens": 256, "temperature": 0.0},
        "input_presentation": "benchmark-json" if benchmark_json else "sft-exact",
        "rows": [],
    }
    if in_sample:
        report["training_cap"] = train_cap
        report["requested_training_rows"] = sorted(requested_rows)
    else:
        report["validation_cap"] = validation_cap
        report["training_overlap_check_cap"] = train_cap
    if reference_report is not None:
        reference = _check_reference(reference_report, report, selected)
        report["reference_report_path"] = str(reference_report)
        report["reference_report_sha256"] = _file_sha256(reference_report)
        report["reference_summary"] = reference["summary"]
    presentations = [
        (number, example, benchmark_json_example(example) if benchmark_json else example)
        for number, example in selected
    ]
    agent = (
        None
        if dry_run
        else LocalLoraAgent(
            LocalLoraConfig(
                base_model=config.base_model,
                adapter_path=adapter,
                max_new_tokens=256,
                temperature=0.0,
                allow_cpu=allow_cpu,
            )
        )
    )
    counts: Counter[str] = Counter()
    for number, example, presented in presentations:
        messages = [message.model_dump() for message in presented.messages[:-1]]
        row_report: dict[str, Any] = {
            "training_row" if in_sample else "validation_row": number,
            "source": refs[number - 1].model_dump(mode="json"),
            "input_sha256": _sft_input_hash(example),
            "model_input_sha256": _sft_input_hash(presented),
            "messages": messages,
            "expected_target": example.messages[-1].content,
            "expected_decision": expected_decision,
        }
        if behavior == "multi-tool":
            row_report["expected_tool_sequence"] = [
                call.tool
                for call in parse_lora_agent_response(example.messages[-1].content).tool_calls
            ]
        if agent is not None:
            invocation = agent.invoke_messages(messages)
            prediction = invocation.response.decision_type or "invalid/unknown"
            counts[prediction] += 1
            row_report.update(
                {
                    "raw_model_output": invocation.raw_text,
                    "agent_response": invocation.response.model_dump(mode="json"),
                    "correct_decision": prediction == expected_decision,
                }
            )
            if behavior == "multi-tool":
                actual_tools = [call.tool for call in invocation.response.tool_calls]
                row_report["actual_tool_sequence"] = actual_tools
                row_report["tool_sequence_matches"] = (
                    invocation.response.parse_error is None
                    and actual_tools == row_report["expected_tool_sequence"]
                )
            print(f"{row_label} row {number}: expected {expected_decision}, predicted {prediction}")
        report["rows"].append(row_report)
    report["summary"] = (
        None
        if dry_run
        else {
            "examples": limit,
            "decision_counts": {
                label: counts[label] for label in ("act", "clarify", "refuse", "invalid/unknown")
            },
            f"correct_{behavior_label.replace('-', '_')}_decisions": counts[expected_decision],
        }
    )
    if report["summary"] is not None:
        if behavior == "multi-tool":
            report["summary"]["matching_tool_sequences"] = sum(
                row["tool_sequence_matches"] for row in report["rows"]
            )
        else:
            meta_tool = "request_clarification" if behavior == "clarify" else "refuse_request"
            report["summary"][f"raw_outputs_mentioning_{behavior_label}"] = sum(
                meta_tool in row["raw_model_output"] for row in report["rows"]
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as diagnostic_file:
        json.dump(report, diagnostic_file, indent=2)
        diagnostic_file.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--validation-cap", type=int, default=200)
    parser.add_argument("--train-cap", type=int, default=1000)
    parser.add_argument(
        "--behavior",
        choices=tuple(BEHAVIOR_LABELS),
        default="clarify",
        help="Held-out behavior to probe; refusal and multi-tool modes select BAYMAX source rows.",
    )
    parser.add_argument(
        "--training-rows",
        type=int,
        nargs="+",
        default=None,
        help="Explicit one-based training file lines; in-sample only, read in file order.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Select/check rows without loading a model."
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--benchmark-json",
        action="store_true",
        help="Use benchmark JSON spacing/escaping only; preserve every input value.",
    )
    parser.add_argument(
        "--reference-report", type=Path, help="Completed original SFT probe to match."
    )
    args = parser.parse_args(argv)
    try:
        report = run_probe(
            config_path=args.config,
            **{key: value for key, value in vars(args).items() if key != "config"},
        )
    except (ValidationError, json.JSONDecodeError):
        print("error: Invalid config, dataset or adapter JSON structure.")
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 1
    sample = "in-sample training" if report["data_split"] == "train" else "held-out"
    print(
        f"Selected {len(report['rows'])} {sample} {BEHAVIOR_LABELS[report['behavior']]} examples."
    )
    print(f"Report: {args.output}")
    if args.dry_run:
        print("Dry run only; no model/tokenizer loaded or generation performed.")
    else:
        print(json.dumps(report["summary"], indent=2))
        if "reference_summary" in report:
            print("Saved original-SFT reference:")
            print(json.dumps(report["reference_summary"], indent=2))
        if report["data_split"] == "train":
            print("In-sample diagnostic only; not held-out accuracy or benchmark task success.")
        print(
            "Diagnostic only; not benchmark success, argument accuracy or semantic-quality scoring."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
