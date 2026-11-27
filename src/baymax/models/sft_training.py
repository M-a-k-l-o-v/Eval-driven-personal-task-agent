"""LoRA SFT training helpers for Phase 2."""

from __future__ import annotations

import json
from importlib import import_module
from inspect import signature
from math import ceil, isfinite
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from baymax.models.sft_artifact_report import SFTArtifactReport, build_sft_artifact_report
from baymax.models.sft_dataset import SFTSourceRowRef, SFTSplitManifest, SplitName
from baymax.models.sft_validation import validate_training_targets
from baymax.models.training_config import (
    SFTTrainingConfig,
    load_training_config,
    validate_training_config,
)


class TrainingDependencyError(RuntimeError):
    """Raised when optional ML training dependencies are not installed."""


class SFTTrainingOverrides(BaseModel):
    """Runtime overrides for local smoke runs and low-VRAM machines."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool = False
    local_low_vram: bool = False
    load_in_4bit: bool = False
    allow_cpu: bool = False
    trust_remote_code: bool = False
    max_train_rows: int | None = Field(default=None, ge=1)
    max_eval_rows: int | None = Field(default=None, ge=1)
    max_sequence_length: int | None = Field(default=None, ge=1)
    output_dir: Path | None = None


class SFTTrainingPlan(BaseModel):
    """Concrete training settings after applying runtime overrides."""

    model_config = ConfigDict(extra="forbid")

    config_name: str
    config_hash: str
    base_model: str
    train_path: Path
    validation_path: Path
    output_dir: Path
    adapter_path: Path
    train_rows: int
    eval_rows: int
    max_sequence_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    local_low_vram: bool
    load_in_4bit: bool
    allow_cpu: bool
    trust_remote_code: bool
    first_tool_name_weight: float = Field(default=1.0, ge=1, allow_inf_nan=False)


class SFTTrainingResult(BaseModel):
    """Result metadata from a training or dry-run invocation."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    plan: SFTTrainingPlan
    trained: bool
    saved_adapter_path: Path | None = None
    train_artifact_report: SFTArtifactReport | None = None
    validation_artifact_report: SFTArtifactReport | None = None


class SFTTokenizationRowDiagnostic(BaseModel):
    """Tokenization health for one SFT row."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    input_tokens: int
    prompt_tokens_before_truncation: int
    full_tokens_before_truncation: int
    trainable_label_tokens: int
    truncated: bool
    prompt_fills_context: bool
    has_trainable_labels: bool
    issues: list[str]


class SFTTokenizationDiagnostics(BaseModel):
    """Aggregate tokenization health for an SFT split."""

    model_config = ConfigDict(extra="forbid")

    row_count: int
    max_sequence_length: int
    rows_with_trainable_labels: int
    rows_without_trainable_labels: int
    truncated_rows: int
    prompt_fills_context_rows: int
    total_trainable_label_tokens: int
    min_trainable_label_tokens: int
    max_trainable_label_tokens: int
    problem_rows: list[SFTTokenizationRowDiagnostic]


def run_sft_lora_training(
    *,
    config_path: Path,
    overrides: SFTTrainingOverrides,
) -> SFTTrainingResult:
    """Run or dry-run LoRA SFT training from a config file."""

    validation_report = validate_training_config(config_path, check_paths=True)
    if not validation_report.valid:
        issue_text = "\n".join(
            f"- {issue.location}: {issue.message}" for issue in validation_report.issues
        )
        raise ValueError(f"Invalid training config:\n{issue_text}")

    config = load_training_config(config_path)
    train_rows = load_sft_jsonl(config.data.train_path, max_rows=overrides.max_train_rows)
    eval_rows = load_sft_jsonl(config.data.validation_path, max_rows=overrides.max_eval_rows)
    validate_training_targets(train_rows, path=config.data.train_path)
    validate_training_targets(eval_rows, path=config.data.validation_path)
    plan = build_training_plan(
        config=config,
        overrides=overrides,
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
    )
    if (
        plan.first_tool_name_weight > 1
        and plan.output_dir.exists()
        and (not plan.output_dir.is_dir() or any(plan.output_dir.iterdir()))
    ):
        raise ValueError(
            "Weighted SFT requires a new empty output directory; reference runs are immutable."
        )

    if overrides.dry_run:
        return SFTTrainingResult(dry_run=True, plan=plan, trained=False)

    train_source_rows = load_split_source_rows(
        config.data.split_manifest_path,
        split_name="train",
        max_rows=overrides.max_train_rows,
    )
    eval_source_rows = load_split_source_rows(
        config.data.split_manifest_path,
        split_name="validation",
        max_rows=overrides.max_eval_rows,
    )
    train_artifact_report, validation_artifact_report = _train_with_huggingface(
        config=config,
        overrides=overrides,
        plan=plan,
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_source_rows=train_source_rows,
        eval_source_rows=eval_source_rows,
    )
    return SFTTrainingResult(
        dry_run=False,
        plan=plan,
        trained=True,
        saved_adapter_path=plan.adapter_path,
        train_artifact_report=train_artifact_report,
        validation_artifact_report=validation_artifact_report,
    )


def build_training_plan(
    *,
    config: SFTTrainingConfig,
    overrides: SFTTrainingOverrides,
    train_rows: int,
    eval_rows: int,
) -> SFTTrainingPlan:
    """Build concrete training settings after applying runtime overrides."""

    output_dir = overrides.output_dir or config.runtime.output_dir
    adapter_path = output_dir / "adapter"
    max_sequence_length = overrides.max_sequence_length or config.training.max_sequence_length
    per_device_train_batch_size = config.training.per_device_train_batch_size
    per_device_eval_batch_size = config.training.per_device_eval_batch_size
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    if overrides.local_low_vram:
        max_sequence_length = min(max_sequence_length, 1024)
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 1
        gradient_accumulation_steps = max(gradient_accumulation_steps, 16)

    return SFTTrainingPlan(
        config_name=config.config_name,
        config_hash=config.config_hash or "",
        base_model=config.base_model,
        train_path=config.data.train_path,
        validation_path=config.data.validation_path,
        output_dir=output_dir,
        adapter_path=adapter_path,
        train_rows=train_rows,
        eval_rows=eval_rows,
        max_sequence_length=max_sequence_length,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        local_low_vram=overrides.local_low_vram,
        load_in_4bit=overrides.load_in_4bit,
        allow_cpu=overrides.allow_cpu,
        trust_remote_code=overrides.trust_remote_code,
        first_tool_name_weight=config.training.first_tool_name_weight,
    )


def load_sft_jsonl(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Load SFT JSONL examples, optionally capped for smoke runs."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if max_rows is not None and len(rows) >= max_rows:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")

            rows.append(row)

    return rows


def load_split_source_rows(
    manifest_path: Path,
    *,
    split_name: SplitName,
    max_rows: int | None = None,
) -> list[SFTSourceRowRef]:
    """Load source-row refs for the final split order, honoring the same row cap."""

    manifest = SFTSplitManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    source_rows = manifest.splits[split_name].source_rows
    if max_rows is None:
        return source_rows
    return source_rows[:max_rows]


def diagnose_sft_tokenization(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
    *,
    max_problem_rows: int = 20,
) -> SFTTokenizationDiagnostics:
    """Inspect whether SFT rows retain trainable assistant labels after truncation."""

    row_diagnostics = [
        _diagnose_sft_tokenization_row(
            row=row,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            row_index=row_index,
        )
        for row_index, row in enumerate(rows)
    ]
    trainable_label_counts = [
        row.trainable_label_tokens for row in row_diagnostics if row.has_trainable_labels
    ]
    problem_rows = [row for row in row_diagnostics if row.issues][:max_problem_rows]

    return SFTTokenizationDiagnostics(
        row_count=len(row_diagnostics),
        max_sequence_length=max_sequence_length,
        rows_with_trainable_labels=sum(row.has_trainable_labels for row in row_diagnostics),
        rows_without_trainable_labels=sum(not row.has_trainable_labels for row in row_diagnostics),
        truncated_rows=sum(row.truncated for row in row_diagnostics),
        prompt_fills_context_rows=sum(row.prompt_fills_context for row in row_diagnostics),
        total_trainable_label_tokens=sum(row.trainable_label_tokens for row in row_diagnostics),
        min_trainable_label_tokens=min(trainable_label_counts, default=0),
        max_trainable_label_tokens=max(trainable_label_counts, default=0),
        problem_rows=problem_rows,
    )


def _train_with_huggingface(
    *,
    config: SFTTrainingConfig,
    overrides: SFTTrainingOverrides,
    plan: SFTTrainingPlan,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    train_source_rows: list[SFTSourceRowRef],
    eval_source_rows: list[SFTSourceRowRef],
) -> tuple[SFTArtifactReport, SFTArtifactReport]:
    dependencies = _load_training_dependencies()
    torch = dependencies["torch"]
    transformers = dependencies["transformers"]
    datasets = dependencies["datasets"]
    peft = dependencies["peft"]

    transformers.set_seed(config.runtime.seed)

    if not torch.cuda.is_available() and not overrides.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Use --allow-cpu for a very slow CPU smoke run, "
            "or run this on a CUDA GPU machine."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=overrides.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_artifact_report = build_sft_artifact_report(
        rows=train_rows,
        tokenizer=tokenizer,
        max_sequence_length=plan.max_sequence_length,
        source_rows=train_source_rows,
    )
    validation_artifact_report = build_sft_artifact_report(
        rows=eval_rows,
        tokenizer=tokenizer,
        max_sequence_length=plan.max_sequence_length,
        source_rows=eval_source_rows,
    )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": overrides.trust_remote_code,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if overrides.load_in_4bit:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if _supports_bf16(torch) else torch.float16,
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    model.config.use_cache = False
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = peft.LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=peft.TaskType.CAUSAL_LM,
    )
    model = peft.get_peft_model(model, lora_config)

    train_tokenized = tokenize_sft_rows_for_training(
        rows=train_rows,
        tokenizer=tokenizer,
        max_sequence_length=plan.max_sequence_length,
        split_name="train",
        first_tool_name_weight=plan.first_tool_name_weight,
    )
    train_dataset = datasets.Dataset.from_list(train_tokenized)
    eval_dataset = datasets.Dataset.from_list(
        tokenize_sft_rows_for_training(
            rows=eval_rows,
            tokenizer=tokenizer,
            max_sequence_length=plan.max_sequence_length,
            split_name="validation",
        )
    )

    bf16 = config.training.bf16 and _supports_bf16(torch)
    fp16 = config.training.fp16 or (torch.cuda.is_available() and not bf16)
    training_args = transformers.TrainingArguments(
        **build_training_arguments_kwargs(
            config=config,
            plan=plan,
            bf16=bf16,
            fp16=fp16,
            training_arguments_type=transformers.TrainingArguments,
        )
    )
    trainer_type = transformers.Trainer
    collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    if plan.first_tool_name_weight > 1:
        from baymax.models.sft_loss_weighting import WeightedDataCollator, weighted_trainer_class

        trainer_type = weighted_trainer_class(transformers.Trainer, torch)
        collator = WeightedDataCollator(collator, torch, tokenizer.padding_side)
        weighted_tokens = sum(
            sum(value > 1 for value in row["loss_weights"]) for row in train_tokenized
        )
        if not weighted_tokens:
            raise ValueError("No visible first tool-name tokens were selected for weighting.")
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        with (plan.output_dir / "weighted-loss-run.json").open("x", encoding="utf-8") as metadata:
            json.dump(
                {
                    "format_version": "first-tool-name-weight-v1",
                    "config": config.model_dump(mode="json"),
                    "plan": plan.model_dump(mode="json"),
                    "weighted_first_tool_name_tokens": weighted_tokens,
                    "effective_training_rows": len(train_tokenized),
                    "effective_training_token_mass": sum(
                        sum(row["loss_weights"]) for row in train_tokenized
                    ),
                    "normalization": (
                        "sum(weight * token_nll) / sum(weights) over each accumulation window"
                    ),
                    "validation_loss": "unweighted assistant-token cross-entropy",
                    "mixed_boundary_tokens": "weight 1, matching the component diagnostic",
                },
                metadata,
                indent=2,
                allow_nan=False,
            )
            metadata.write("\n")
    del train_tokenized
    trainer = trainer_type(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    plan.adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(plan.adapter_path)
    tokenizer.save_pretrained(plan.adapter_path)
    return train_artifact_report, validation_artifact_report


def build_training_arguments_kwargs(
    *,
    config: SFTTrainingConfig,
    plan: SFTTrainingPlan,
    bf16: bool,
    fp16: bool,
    training_arguments_type: type[Any],
) -> dict[str, Any]:
    """Build TrainingArguments kwargs across supported transformers versions."""

    accepted_parameters = set(signature(training_arguments_type.__init__).parameters)
    kwargs: dict[str, Any] = {
        "output_dir": str(plan.output_dir),
        "run_name": config.runtime.run_name,
        "seed": config.runtime.seed,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
        "per_device_train_batch_size": plan.per_device_train_batch_size,
        "per_device_eval_batch_size": plan.per_device_eval_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "weight_decay": config.training.weight_decay,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "max_grad_norm": config.training.max_grad_norm,
        "logging_steps": config.training.logging_steps,
        "eval_steps": config.training.eval_steps,
        "save_strategy": "steps",
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
        "report_to": config.runtime.report_to,
        "remove_unused_columns": False,
    }

    if "warmup_ratio" in accepted_parameters:
        kwargs["warmup_ratio"] = config.training.warmup_ratio
    elif "warmup_steps" in accepted_parameters:
        kwargs["warmup_steps"] = _warmup_steps(config=config, plan=plan)

    if "eval_strategy" in accepted_parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in accepted_parameters:
        kwargs["evaluation_strategy"] = "steps"

    return {key: value for key, value in kwargs.items() if key in accepted_parameters}


def tokenize_sft_rows_for_training(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
    split_name: str,
    first_tool_name_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Tokenize rows and skip examples that cannot contribute training loss."""

    if not isfinite(first_tool_name_weight) or first_tool_name_weight < 1:
        raise ValueError("First tool-name weight must be finite and at least 1.")
    tokenized_rows = []
    for number, row in enumerate(rows, 1):
        tokenized = _tokenize_chat_example(row, tokenizer, max_sequence_length)
        if not _has_trainable_labels(tokenized):
            continue
        if first_tool_name_weight > 1:
            from baymax.models.sft_loss_weighting import first_tool_name_weights

            try:
                tokenized["loss_weights"] = first_tool_name_weights(
                    row, tokenizer, tokenized, max_sequence_length, first_tool_name_weight
                )
            except ValueError as error:
                raise ValueError(f"{split_name} row {number}: {error}") from error
        tokenized_rows.append(tokenized)
    if not tokenized_rows:
        raise ValueError(
            f"{split_name} split has no rows with trainable assistant labels after tokenization"
        )

    return tokenized_rows


def _warmup_steps(*, config: SFTTrainingConfig, plan: SFTTrainingPlan) -> int:
    optimizer_steps = ceil(
        (plan.train_rows / plan.per_device_train_batch_size / plan.gradient_accumulation_steps)
        * config.training.num_train_epochs
    )
    return ceil(max(optimizer_steps, 1) * config.training.warmup_ratio)


def _tokenize_chat_example(
    row: dict[str, Any],
    tokenizer: Any,
    max_sequence_length: int,
) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("SFT row must contain exactly three messages")

    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_sequence_length,
        add_special_tokens=False,
    )
    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_sequence_length,
        add_special_tokens=False,
    )
    labels = list(tokenized_full["input_ids"])
    prompt_token_count = min(len(tokenized_prompt["input_ids"]), len(labels))
    labels[:prompt_token_count] = [-100] * prompt_token_count

    return {
        "input_ids": tokenized_full["input_ids"],
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }


def _has_trainable_labels(tokenized_row: dict[str, Any]) -> bool:
    return any(label != -100 for label in tokenized_row["labels"])


def _diagnose_sft_tokenization_row(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    max_sequence_length: int,
    row_index: int,
) -> SFTTokenizationRowDiagnostic:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return SFTTokenizationRowDiagnostic(
            row_index=row_index,
            input_tokens=0,
            prompt_tokens_before_truncation=0,
            full_tokens_before_truncation=0,
            trainable_label_tokens=0,
            truncated=False,
            prompt_fills_context=False,
            has_trainable_labels=False,
            issues=["invalid_messages"],
        )

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_token_count = len(
        tokenizer(
            prompt_text,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
    )
    full_token_count = len(
        tokenizer(
            full_text,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
    )
    tokenized = _tokenize_chat_example(row, tokenizer, max_sequence_length)
    trainable_label_tokens = sum(label != -100 for label in tokenized["labels"])
    truncated = full_token_count > max_sequence_length
    prompt_fills_context = prompt_token_count >= max_sequence_length
    issues: list[str] = []

    if truncated:
        issues.append("truncated")
    if prompt_fills_context:
        issues.append("prompt_fills_context")
    if trainable_label_tokens == 0:
        issues.append("no_trainable_labels")

    return SFTTokenizationRowDiagnostic(
        row_index=row_index,
        input_tokens=len(tokenized["input_ids"]),
        prompt_tokens_before_truncation=prompt_token_count,
        full_tokens_before_truncation=full_token_count,
        trainable_label_tokens=trainable_label_tokens,
        truncated=truncated,
        prompt_fills_context=prompt_fills_context,
        has_trainable_labels=trainable_label_tokens > 0,
        issues=issues,
    )


def _supports_bf16(torch: Any) -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def _load_training_dependencies() -> dict[str, Any]:
    try:
        datasets = import_module("datasets")
        peft = import_module("peft")
        torch = import_module("torch")
        transformers = import_module("transformers")
    except ImportError as error:
        raise TrainingDependencyError(
            "Missing training dependencies. Install them before real training, for example: "
            "python -m pip install torch transformers peft accelerate"
        ) from error

    return {
        "datasets": datasets,
        "peft": peft,
        "torch": torch,
        "transformers": transformers,
    }
