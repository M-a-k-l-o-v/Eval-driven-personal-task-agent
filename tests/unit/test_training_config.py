from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import pytest

from baymax.models.training_config import (
    LoraTrainingConfig,
    OutputAdapterConfig,
    SFTTrainingConfig,
    TrainingDataConfig,
    TrainingHyperparameters,
    TrainingRuntimeConfig,
    load_training_config,
    training_config_hash,
    training_config_with_hash,
    validate_training_config,
    write_training_config,
)

PROMPT_SHA256 = "a31446de6c8b67f1fc8ba8fc09ed8e2d7425ba832334dc8fd6fe2d55ef978468"


def test_training_config_hash_is_stable_and_excludes_hash_field() -> None:
    config = _training_config()
    first_hash = training_config_hash(config)
    hashed_config = config.model_copy(update={"config_hash": "old-value"})

    assert training_config_hash(hashed_config) == first_hash
    assert training_config_with_hash(config).config_hash == first_hash


def test_validate_training_config_reports_hash_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_training_config(config_path, _training_config(config_hash="incorrect"))

    report = validate_training_config(config_path)

    assert report.valid is False
    assert report.config_hash == "incorrect"
    assert report.expected_config_hash != "incorrect"
    assert report.issues[0].location == "config_hash"


def test_validate_training_config_accepts_matching_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(_training_config()))

    report = validate_training_config(config_path)

    assert report.valid is True
    assert report.issues == []


def test_validate_training_config_can_check_referenced_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(_training_config()))

    report = validate_training_config(config_path, check_paths=True)

    assert report.valid is False
    assert {issue.location for issue in report.issues} == {
        "data.train_path",
        "data.validation_path",
        "data.test_path",
        "data.split_manifest_path",
        "data.validation_report_path",
        "data.eval_scenarios_paths[0]",
    }


def test_validate_training_config_accepts_matching_manifest(tmp_path: Path) -> None:
    config = _training_config_with_manifest(tmp_path)
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(config))

    report = validate_training_config(config_path)

    assert report.valid is True
    assert report.issues == []


@pytest.mark.parametrize(
    ("field_name", "manifest_value"),
    [
        ("dataset_id", "different/dataset"),
        ("revision", "different-revision"),
        ("format_version", "different-format"),
        ("system_prompt_sha256", "0" * 64),
    ],
)
def test_validate_training_config_reports_manifest_identity_mismatch(
    tmp_path: Path,
    field_name: str,
    manifest_value: str,
) -> None:
    config = _training_config_with_manifest(tmp_path)
    manifest_path = config.data.split_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field_name] = manifest_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(config))

    report = validate_training_config(config_path)

    assert report.valid is False
    assert {issue.location for issue in report.issues} == {f"data.{field_name}"}


def test_validate_training_config_reports_invalid_manifest(tmp_path: Path) -> None:
    config = _training_config_with_manifest(tmp_path)
    config.data.split_manifest_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(config))

    report = validate_training_config(config_path)

    assert report.valid is False
    assert [issue.location for issue in report.issues] == ["data.split_manifest_path"]


def test_validate_training_config_reports_manifest_split_path_mismatch(
    tmp_path: Path,
) -> None:
    config = _training_config_with_manifest(tmp_path)
    manifest_path = config.data.split_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["path"] = "different/train.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = tmp_path / "config.json"
    write_training_config(config_path, training_config_with_hash(config))

    report = validate_training_config(config_path)

    assert report.valid is False
    assert [issue.location for issue in report.issues] == ["data.train_path"]


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/sft/qwen2_5_1_5b_lora_v1.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_baymax_control_v1.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_baymax_mix_v1.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_val200.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_act200_val200.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_matched_act200_val200.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_val200.json"),
        Path("configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200.json"),
        Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json"
        ),
        Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs4.json"
        ),
    ],
)
def test_committed_qwen_lora_configs_are_valid(config_path: Path) -> None:
    config = load_training_config(config_path)
    report = validate_training_config(config_path)

    assert report.valid is True
    assert config.lora.r == 8
    assert config.lora.lora_alpha == 16
    assert config.lora.target_modules == ["q_proj", "v_proj"]
    assert config.lora.lora_dropout == 0.05
    assert config.lora.bias == "none"
    assert config.lora.task_type == "CAUSAL_LM"


def test_auth_filtered_config_preserves_training_settings_and_uses_new_paths() -> None:
    previous = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_content_groups_70_30_v12.json")
    )
    current = load_training_config(Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14.json"))

    assert current.base_model == previous.base_model
    assert current.lora == previous.lora
    assert current.training == previous.training
    assert current.runtime.seed == previous.runtime.seed == 1
    assert current.data.system_prompt_sha256 == previous.data.system_prompt_sha256
    assert current.data.revision != previous.data.revision
    directory = PureWindowsPath("data/sft/v1/xlam-baymax-auth-filtered-v14")
    for field in (
        "train_path",
        "validation_path",
        "test_path",
        "split_manifest_path",
        "validation_report_path",
    ):
        assert PureWindowsPath(getattr(current.data, field)).parent == directory
    assert current.runtime.output_dir != previous.runtime.output_dir
    assert PureWindowsPath(current.runtime.expected_adapter_path) == (
        PureWindowsPath(current.runtime.output_dir) / "adapter"
    )


@pytest.mark.parametrize(
    "config_name,directory_name,revision",
    [
        (
            "qwen2_5_1_5b_lora_target_compatible_v14_act200_val200",
            "xlam-baymax-target-compatible-v14-act200-val200",
            "e69265988209fac83d47d8e1b19e7d572b1bb0205bd1417e7e2ef106f50a665a",
        ),
        (
            "qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200",
            "xlam-baymax-target-compatible-v14-act200-anchored-val200",
            "329fab4d8f1604a4085d3ec50e05881e7c069717bb1428db7e592743bb07f03c",
        ),
    ],
)
def test_target_compatible_control_changes_data_identity_not_act200_settings(
    config_name: str, directory_name: str, revision: str
) -> None:
    previous = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_act200_val200.json")
    )
    current = load_training_config(Path("configs/sft") / f"{config_name}.json")
    assert current.base_model == previous.base_model
    assert current.lora == previous.lora
    assert current.training == previous.training
    assert current.output_adapter == previous.output_adapter
    assert current.runtime.seed == previous.runtime.seed == 1
    assert current.data.system_prompt_sha256 == previous.data.system_prompt_sha256
    assert current.data.eval_scenarios_paths == previous.data.eval_scenarios_paths
    assert current.data.revision == revision
    directory = PureWindowsPath("data/sft/v1") / directory_name
    for field in (
        "train_path",
        "validation_path",
        "test_path",
        "split_manifest_path",
        "validation_report_path",
    ):
        assert PureWindowsPath(getattr(current.data, field)).parent == directory
    assert current.runtime.output_dir != previous.runtime.output_dir
    assert current.runtime.expected_adapter_path != previous.runtime.expected_adapter_path
    assert PureWindowsPath(current.runtime.expected_adapter_path) == (
        PureWindowsPath(current.runtime.output_dir) / "adapter"
    )


def test_validation_cap_config_changes_no_training_or_prompt_settings() -> None:
    previous = load_training_config(Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14.json"))
    current = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_val200.json")
    )
    assert current.base_model == previous.base_model
    assert current.lora == previous.lora
    assert current.training == previous.training
    assert current.runtime.seed == previous.runtime.seed
    assert current.data.system_prompt_sha256 == previous.data.system_prompt_sha256
    assert current.data.revision != previous.data.revision
    assert current.data.train_path != previous.data.train_path
    assert current.runtime.output_dir != previous.runtime.output_dir


def test_act200_config_preserves_settings_and_versions_data_and_output() -> None:
    previous = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_val200.json")
    )
    current = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_act200_val200.json")
    )
    assert current.base_model == previous.base_model
    assert current.lora == previous.lora
    assert current.training == previous.training
    assert current.runtime.seed == previous.runtime.seed == 1
    assert current.output_adapter == previous.output_adapter
    assert current.data.system_prompt_sha256 == previous.data.system_prompt_sha256
    assert current.data.eval_scenarios_paths == previous.data.eval_scenarios_paths
    assert current.data.revision != previous.data.revision
    directory = PureWindowsPath("data/sft/v1/xlam-baymax-auth-filtered-v14-act200-val200")
    for field in (
        "train_path",
        "validation_path",
        "test_path",
        "split_manifest_path",
        "validation_report_path",
    ):
        assert PureWindowsPath(getattr(current.data, field)).parent == directory
    assert current.runtime.output_dir != previous.runtime.output_dir
    assert current.runtime.run_name == current.config_name
    assert PureWindowsPath(current.runtime.expected_adapter_path) == (
        PureWindowsPath(current.runtime.output_dir) / "adapter"
    )


def test_matched_config_changes_no_model_prompt_or_training_settings() -> None:
    previous = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_act200_val200.json")
    )
    current = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_auth_filtered_v14_matched_act200_val200.json")
    )
    assert current.base_model == previous.base_model
    assert current.lora == previous.lora
    assert current.training == previous.training
    assert current.runtime.seed == previous.runtime.seed
    assert current.output_adapter == previous.output_adapter
    assert current.data.system_prompt_sha256 == previous.data.system_prompt_sha256
    assert current.data.eval_scenarios_paths == previous.data.eval_scenarios_paths
    assert current.data.revision != previous.data.revision
    assert current.runtime.output_dir != previous.runtime.output_dir


@pytest.mark.parametrize("epochs", [2, 4])
def test_anchored_epoch_configs_change_only_epochs_and_run_identity(epochs: int) -> None:
    previous = load_training_config(
        Path("configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200.json")
    )
    current = load_training_config(
        Path("configs/sft") / f"{previous.config_name}_epochs{epochs}.json"
    )

    assert previous.training.num_train_epochs == 1.0
    assert current.training.num_train_epochs == float(epochs)
    assert current.training.model_dump(exclude={"num_train_epochs"}) == (
        previous.training.model_dump(exclude={"num_train_epochs"})
    )
    assert current.base_model == previous.base_model
    assert current.data == previous.data
    assert current.lora == previous.lora
    assert current.output_adapter == previous.output_adapter
    changed_runtime_fields = {"run_name", "output_dir", "expected_adapter_path"}
    assert current.runtime.model_dump(exclude=changed_runtime_fields) == (
        previous.runtime.model_dump(exclude=changed_runtime_fields)
    )
    assert current.config_name == f"{previous.config_name}_epochs{epochs}"
    assert current.runtime.run_name == current.config_name
    assert PureWindowsPath(current.runtime.output_dir) == (
        PureWindowsPath("models") / current.config_name
    )
    assert PureWindowsPath(current.runtime.expected_adapter_path) == (
        PureWindowsPath(current.runtime.output_dir) / "adapter"
    )
    assert current.config_hash == training_config_hash(current)
    assert current.config_hash != previous.config_hash


def _training_config(*, config_hash: str | None = None) -> SFTTrainingConfig:
    return SFTTrainingConfig(
        config_name="test_config",
        config_hash=config_hash,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        data=TrainingDataConfig(
            dataset_id="Salesforce/xLAM-function-calling-60k",
            revision="26d14ebfe18b1f7b524bd39b404b50af5dc97866",
            format_version="qwen-chat-messages-v2",
            system_prompt_sha256=PROMPT_SHA256,
            train_path=Path("missing/train.jsonl"),
            validation_path=Path("missing/validation.jsonl"),
            test_path=Path("missing/test.jsonl"),
            split_manifest_path=Path("missing/split-manifest.json"),
            validation_report_path=Path("missing/validation-report.json"),
            eval_scenarios_paths=[Path("missing/scenarios")],
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
            output_dir=Path("models/test_config"),
            model_registry_path=Path("models/models.json"),
            expected_adapter_path=Path("models/test_config/adapter"),
            report_to=[],
        ),
        output_adapter=OutputAdapterConfig(
            native_output_format="xlam_function_calling",
            baymax_boundary="AgentResponse",
            adapter_owner="Marv",
        ),
    )


def _training_config_with_manifest(tmp_path: Path) -> SFTTrainingConfig:
    config = _training_config()
    data = config.data.model_copy(
        update={
            "train_path": tmp_path / "train.jsonl",
            "validation_path": tmp_path / "validation.jsonl",
            "test_path": tmp_path / "test.jsonl",
            "split_manifest_path": tmp_path / "split-manifest.json",
        }
    )
    config = config.model_copy(update={"data": data})
    _write_split_manifest(config)
    return config


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
