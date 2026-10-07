from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from baymax.models import sft_training
from baymax.models.sft_dataset import convert_xlam_row_to_sft_example
from baymax.models.sft_training import (
    SFTTrainingOverrides,
    build_training_arguments_kwargs,
    build_training_plan,
    diagnose_sft_tokenization,
    load_sft_jsonl,
    run_sft_lora_training,
    tokenize_sft_rows_for_training,
)
from baymax.models.training_config import (
    LoraTrainingConfig,
    OutputAdapterConfig,
    SFTTrainingConfig,
    TrainingDataConfig,
    TrainingHyperparameters,
    TrainingRuntimeConfig,
    training_config_with_hash,
    write_training_config,
)

PROMPT_SHA256 = "a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468"


def test_seed_is_applied_before_model_and_adapter_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _training_config(tmp_path)
    plan = build_training_plan(
        config=config, overrides=SFTTrainingOverrides(), train_rows=1, eval_rows=1
    )
    samples: list[float] = []

    class StopBeforeTraining(Exception):
        pass

    def initialize_adapter(*args: object) -> None:
        samples.append(random.random())
        raise StopBeforeTraining

    model_loader = Mock(
        side_effect=lambda *args, **kwargs: (samples.append(random.random()), Mock())[1]
    )
    transformers = SimpleNamespace(
        set_seed=Mock(side_effect=random.seed),
        AutoTokenizer=Mock(),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
    )
    monkeypatch.setattr(
        sft_training,
        "_load_training_dependencies",
        lambda: {
            "torch": Mock(),
            "transformers": transformers,
            "datasets": Mock(),
            "peft": SimpleNamespace(
                LoraConfig=Mock(),
                TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
                get_peft_model=initialize_adapter,
            ),
        },
    )
    monkeypatch.setattr(sft_training, "build_sft_artifact_report", Mock())
    for ambient_seed in (7, 99):
        random.seed(ambient_seed)
        with pytest.raises(StopBeforeTraining):
            sft_training._train_with_huggingface(
                config=config,
                overrides=SFTTrainingOverrides(),
                plan=plan,
                train_rows=[],
                eval_rows=[],
                train_source_rows=[],
                eval_source_rows=[],
            )
    assert samples[:2] == samples[2:]
    assert transformers.set_seed.call_count == 2
    transformers.set_seed.assert_called_with(config.runtime.seed)


def test_load_sft_jsonl_honors_max_rows(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.jsonl"
    _write_sft_jsonl(jsonl_path, row_count=3)

    rows = load_sft_jsonl(jsonl_path, max_rows=2)

    assert len(rows) == 2


def test_load_sft_jsonl_rejects_invalid_json(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_sft_jsonl(jsonl_path)


def test_build_training_plan_applies_low_vram_overrides(tmp_path: Path) -> None:
    config = _training_config(tmp_path)
    overrides = SFTTrainingOverrides(
        local_low_vram=True,
        max_sequence_length=2048,
        max_train_rows=100,
        max_eval_rows=20,
    )

    plan = build_training_plan(
        config=config,
        overrides=overrides,
        train_rows=100,
        eval_rows=20,
    )

    assert plan.max_sequence_length == 1024
    assert plan.per_device_train_batch_size == 1
    assert plan.per_device_eval_batch_size == 1
    assert plan.gradient_accumulation_steps == 16
    assert plan.train_rows == 100
    assert plan.eval_rows == 20


def test_build_training_arguments_kwargs_uses_warmup_ratio_when_supported(
    tmp_path: Path,
) -> None:
    config = _training_config(tmp_path)
    plan = build_training_plan(
        config=config,
        overrides=SFTTrainingOverrides(),
        train_rows=100,
        eval_rows=20,
    )

    kwargs = build_training_arguments_kwargs(
        config=config,
        plan=plan,
        bf16=False,
        fp16=True,
        training_arguments_type=_TrainingArgumentsWithWarmupRatio,
    )

    assert kwargs["warmup_ratio"] == 0.03
    assert "warmup_steps" not in kwargs
    assert kwargs["eval_strategy"] == "steps"


def test_build_training_arguments_kwargs_falls_back_to_warmup_steps(
    tmp_path: Path,
) -> None:
    config = _training_config(tmp_path)
    plan = build_training_plan(
        config=config,
        overrides=SFTTrainingOverrides(local_low_vram=True),
        train_rows=100,
        eval_rows=20,
    )

    kwargs = build_training_arguments_kwargs(
        config=config,
        plan=plan,
        bf16=False,
        fp16=True,
        training_arguments_type=_TrainingArgumentsWithWarmupSteps,
    )

    assert kwargs["warmup_steps"] == 1
    assert "warmup_ratio" not in kwargs
    assert kwargs["eval_strategy"] == "steps"


def test_diagnose_sft_tokenization_counts_problem_rows() -> None:
    rows = [
        _short_tokenization_row(),
        _sft_row_with_long_prompt(),
        {"messages": []},
    ]

    diagnostics = diagnose_sft_tokenization(
        rows=rows,
        tokenizer=_WhitespaceTokenizer(),
        max_sequence_length=8,
    )

    assert diagnostics.row_count == 3
    assert diagnostics.rows_with_trainable_labels == 1
    assert diagnostics.rows_without_trainable_labels == 2
    assert diagnostics.truncated_rows == 1
    assert diagnostics.prompt_fills_context_rows == 1
    assert [row.row_index for row in diagnostics.problem_rows] == [1, 2]
    assert "no_trainable_labels" in diagnostics.problem_rows[0].issues
    assert "invalid_messages" in diagnostics.problem_rows[1].issues


def test_tokenize_sft_rows_for_training_skips_rows_without_labels() -> None:
    tokenized_rows = tokenize_sft_rows_for_training(
        rows=[_short_tokenization_row(), _sft_row_with_long_prompt()],
        tokenizer=_WhitespaceTokenizer(),
        max_sequence_length=8,
        split_name="train",
    )

    assert len(tokenized_rows) == 1
    assert any(label != -100 for label in tokenized_rows[0]["labels"])


def test_tokenize_sft_rows_for_training_rejects_empty_effective_split() -> None:
    with pytest.raises(ValueError, match="validation split has no rows"):
        tokenize_sft_rows_for_training(
            rows=[_sft_row_with_long_prompt()],
            tokenizer=_WhitespaceTokenizer(),
            max_sequence_length=8,
            split_name="validation",
        )


def test_run_sft_lora_training_dry_run_does_not_need_ml_dependencies(tmp_path: Path) -> None:
    config_path = _write_training_config_with_data(tmp_path)

    result = run_sft_lora_training(
        config_path=config_path,
        overrides=SFTTrainingOverrides(
            dry_run=True,
            local_low_vram=True,
            max_train_rows=2,
            max_eval_rows=1,
        ),
    )

    assert result.dry_run is True
    assert result.trained is False
    assert result.plan.train_rows == 2
    assert result.plan.eval_rows == 1
    assert result.saved_adapter_path is None


def test_training_rejects_invalid_target_before_loading_ml_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_training_config_with_data(tmp_path)
    row = _sft_row(1)
    row["messages"][2]["content"] = (
        '[{"name":"request_clarification","arguments":{"reason":"missing"}}]'
    )
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    loader = Mock()
    monkeypatch.setattr(sft_training, "_load_training_dependencies", loader)
    with pytest.raises(ValueError, match="meta-tool must be exclusive"):
        run_sft_lora_training(config_path=config_path, overrides=SFTTrainingOverrides())
    loader.assert_not_called()


def test_run_sft_lora_training_rejects_missing_data_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(_training_config(tmp_path)))

    with pytest.raises(ValueError, match="Invalid training config"):
        run_sft_lora_training(
            config_path=config_path,
            overrides=SFTTrainingOverrides(dry_run=True),
        )


def test_run_sft_lora_training_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    config_path = _write_training_config_with_data(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["data"]["split_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["system_prompt_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="data.system_prompt_sha256"):
        run_sft_lora_training(
            config_path=config_path,
            overrides=SFTTrainingOverrides(dry_run=True),
        )


def _write_training_config_with_data(tmp_path: Path) -> Path:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    split_manifest_path = tmp_path / "split-manifest.json"
    validation_report_path = tmp_path / "validation-report.json"
    scenarios_path = tmp_path / "scenarios"
    scenarios_path.mkdir()

    _write_sft_jsonl(train_path, row_count=3)
    _write_sft_jsonl(validation_path, row_count=2)
    _write_sft_jsonl(test_path, row_count=1)
    validation_report_path.write_text("{}", encoding="utf-8")

    config = _training_config(
        tmp_path,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        split_manifest_path=split_manifest_path,
        validation_report_path=validation_report_path,
        scenarios_path=scenarios_path,
    )
    _write_split_manifest(config)
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(config))
    return config_path


def _training_config(
    tmp_path: Path,
    *,
    train_path: Path | None = None,
    validation_path: Path | None = None,
    test_path: Path | None = None,
    split_manifest_path: Path | None = None,
    validation_report_path: Path | None = None,
    scenarios_path: Path | None = None,
) -> SFTTrainingConfig:
    return SFTTrainingConfig(
        config_name="test_config",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        data=TrainingDataConfig(
            dataset_id="Salesforce/xLAM-function-calling-60k",
            revision="26d14ebfe18b1f7b524bd39b404b50af5dc97866",
            format_version="qwen-chat-messages-v2",
            system_prompt_sha256=PROMPT_SHA256,
            train_path=train_path or tmp_path / "missing-train.jsonl",
            validation_path=validation_path or tmp_path / "missing-validation.jsonl",
            test_path=test_path or tmp_path / "missing-test.jsonl",
            split_manifest_path=split_manifest_path or tmp_path / "missing-manifest.json",
            validation_report_path=validation_report_path or tmp_path / "missing-report.json",
            eval_scenarios_paths=[scenarios_path or tmp_path / "missing-scenarios"],
        ),
        lora=LoraTrainingConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        training=TrainingHyperparameters(
            learning_rate=0.0002,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            max_sequence_length=2048,
            warmup_ratio=0.03,
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            gradient_checkpointing=True,
            bf16=True,
            fp16=False,
            logging_steps=10,
            eval_steps=250,
            save_steps=250,
            save_total_limit=2,
        ),
        runtime=TrainingRuntimeConfig(
            seed=1,
            run_name="test_config",
            output_dir=tmp_path / "model-output",
            model_registry_path=tmp_path / "models.json",
            expected_adapter_path=tmp_path / "model-output" / "adapter",
            report_to=[],
        ),
        output_adapter=OutputAdapterConfig(
            native_output_format="xlam_function_calling",
            baymax_boundary="AgentResponse",
            adapter_owner="Marv",
        ),
    )


def _write_split_manifest(config: SFTTrainingConfig) -> None:
    manifest = {
        "format_version": config.data.format_version,
        "config_hash": "manifest-config-hash",
        "dataset_id": config.data.dataset_id,
        "revision": config.data.revision,
        "source_split": "train",
        "seed": 1,
        "max_rows": None,
        "split_ratios": {"train": 0.7, "validation": 0.2, "test": 0.1},
        "system_prompt_sha256": config.data.system_prompt_sha256,
        "output_dir": str(config.data.split_manifest_path.parent),
        "splits": {
            split_name: {
                "path": str(path),
                "count": 0,
                "source_rows": [],
            }
            for split_name, path in {
                "train": config.data.train_path,
                "validation": config.data.validation_path,
                "test": config.data.test_path,
            }.items()
        },
        "loaded_rows": 0,
        "converted_rows": 0,
        "skipped_rows": [],
    }
    config.data.split_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_sft_jsonl(path: Path, *, row_count: int) -> None:
    rows = [_sft_row(index) for index in range(row_count)]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _sft_row(index: int) -> dict[str, Any]:
    return convert_xlam_row_to_sft_example(
        {
            "query": f"user {index}",
            "tools": [{"name": "lookup", "parameters": {"item_id": {"type": "int"}}}],
            "answers": [{"name": "lookup", "arguments": {"item_id": index}}],
        },
        system_prompt="system",
    ).model_dump(mode="json")


def _sft_row_with_long_prompt() -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "one two three four five six seven eight nine"},
            {"role": "assistant", "content": "answer"},
        ]
    }


def _short_tokenization_row() -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "answer"},
        ]
    }


class _WhitespaceTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            text = f"{text}\nassistant:"

        return text

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        add_special_tokens: bool,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        tokens = text.split()
        if truncation and max_length is not None:
            tokens = tokens[:max_length]

        input_ids = list(range(len(tokens)))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }


class _TrainingArgumentsWithWarmupRatio:
    def __init__(
        self,
        *,
        output_dir: str,
        warmup_ratio: float,
        eval_strategy: str,
    ) -> None:
        pass


class _TrainingArgumentsWithWarmupSteps:
    def __init__(
        self,
        *,
        output_dir: str,
        warmup_steps: int,
        eval_strategy: str,
    ) -> None:
        pass
