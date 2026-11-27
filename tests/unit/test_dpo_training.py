from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from baymax.models import dpo_training
from baymax.models.baymax_sft import (
    generate_synthetic_baymax_examples,
    synthetic_baymax_example_to_sft_example,
)
from baymax.models.dpo_dataset import build_preference_pairs
from baymax.models.dpo_training import (
    DPOTrainingConfig,
    DPOTrainingOverrides,
    check_pair_tokenization,
    load_preference_data,
    run_dpo_training,
    verify_reference_adapter,
)
from baymax.models.sft_dataset import DEFAULT_SFT_SYSTEM_PROMPT
from baymax.models.sft_mix import _file_sha256
from baymax.models.training_config import (
    load_training_config,
    training_config_with_hash,
    write_training_config,
)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    ref = load_training_config(
        root
        / "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json"
    )
    adapter = tmp_path / "reference-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": ref.base_model, "peft_type": "LORA", "bias": "none"})
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"frozen-test-weights")
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    for split in ("validation", "test"):
        (heldout / f"{split}.jsonl").write_text(split)
    ref = training_config_with_hash(
        ref.model_copy(
            update={
                "runtime": ref.runtime.model_copy(update={"expected_adapter_path": adapter}),
                "data": ref.data.model_copy(
                    update={
                        "validation_path": heldout / "validation.jsonl",
                        "test_path": heldout / "test.jsonl",
                    }
                ),
            }
        )
    )
    ref_path = tmp_path / "sft-configs" / "ref.json"
    ref_path.parent.mkdir()
    write_training_config(ref_path, ref)
    sources = [
        s
        for s in generate_synthetic_baymax_examples()
        if s.contrast_group == "act_clarify_refuse_triplet_001"
    ]
    pool = {
        s.id: synthetic_baymax_example_to_sft_example(s, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT)
        for s in sources
    }
    pairs, _ = build_preference_pairs(
        pool, pool, {s.id: str(s.contrast_group) for s in sources}, protected_inputs=set(), seed=1
    )
    data_dir = tmp_path / "preferences"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        "".join(
            json.dumps(p.model_dump(mode="json", include={"prompt", "chosen", "rejected"})) + "\n"
            for p in pairs
        ),
        encoding="utf-8",
    )
    (data_dir / "pairs-review.jsonl").write_text(
        "".join(p.model_dump_json() + "\n" for p in pairs), encoding="utf-8"
    )
    manifest = {
        "format_version": "baymax-decision-preferences-v1",
        "reference_config_sha256": _file_sha256(ref_path),
        "reference_config_hash": ref.config_hash,
        "reference_adapter": str(adapter),
        "heldout_sha256": {
            str(path): _file_sha256(path) for path in (ref.data.validation_path, ref.data.test_path)
        },
        "train_sha256": _file_sha256(data_dir / "train.jsonl"),
        "pair_count": len(pairs),
        "unique_prompts": 3,
        "direction_counts": dict(
            Counter(f"{p.chosen_decision}>{p.rejected_decision}" for p in pairs)
        ),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    config = DPOTrainingConfig(
        reference_config=ref_path,
        preference_dir=data_dir,
        preference_manifest_sha256=_file_sha256(data_dir / "manifest.json"),
        preference_review_sha256=_file_sha256(data_dir / "pairs-review.jsonl"),
        adapter_config_sha256=_file_sha256(adapter / "adapter_config.json"),
        adapter_weights_sha256=_file_sha256(adapter / "adapter_model.safetensors"),
        output_dir=tmp_path / "new-run",
    )
    path = tmp_path / "dpo-configs" / "config.json"
    path.parent.mkdir()
    path.write_text(config.model_dump_json(), encoding="utf-8")
    return path


def _config(path: Path) -> DPOTrainingConfig:
    return DPOTrainingConfig.model_validate_json(path.read_text())


def _alternate_start(config_path: Path) -> DPOTrainingConfig:
    config = _config(config_path)
    ref = load_training_config(config.reference_config)
    adapter = config_path.parent.parent / "alternate-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_bytes(
        (ref.runtime.expected_adapter_path / "adapter_config.json").read_bytes()
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"alternate-sft-test-weights")
    initial = training_config_with_hash(
        ref.model_copy(
            update={"runtime": ref.runtime.model_copy(update={"expected_adapter_path": adapter})}
        )
    )
    path = config_path.parent / "alternate-sft.json"
    write_training_config(path, initial)
    return config.model_copy(
        update={
            "initial_sft_config": path,
            "initial_sft_config_sha256": _file_sha256(path),
            "adapter_config_sha256": _file_sha256(adapter / "adapter_config.json"),
            "adapter_weights_sha256": _file_sha256(adapter / "adapter_model.safetensors"),
        }
    )


def test_alternate_start_changes_checkpoint_not_preferences(config_path: Path) -> None:
    config = _config(config_path)
    original_pairs, original_adapter, _ = load_preference_data(config)
    alternate = _alternate_start(config_path)
    pairs, adapter, _ = load_preference_data(alternate)
    assert pairs == original_pairs
    assert adapter != original_adapter and adapter.name == "alternate-adapter"
    assert (
        _file_sha256(config.preference_dir / "manifest.json") == config.preference_manifest_sha256
    )


def test_alternate_start_requires_immutable_config_identity(config_path: Path) -> None:
    alternate = _alternate_start(config_path)
    with pytest.raises(ValueError, match="requires initial_sft_config_sha256"):
        load_preference_data(alternate.model_copy(update={"initial_sft_config_sha256": None}))
    assert alternate.initial_sft_config is not None
    alternate.initial_sft_config.write_bytes(alternate.initial_sft_config.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_preference_data(alternate)


@pytest.mark.parametrize("change", ["prompt", "base_model", "heldout"])
def test_alternate_start_rejects_incompatible_prompt_model_or_heldout(
    config_path: Path, change: str
) -> None:
    alternate = _alternate_start(config_path)
    assert alternate.initial_sft_config is not None
    initial = load_training_config(alternate.initial_sft_config)
    if change == "prompt":
        initial = initial.model_copy(
            update={"data": initial.data.model_copy(update={"system_prompt_sha256": "0" * 64})}
        )
    elif change == "base_model":
        initial = initial.model_copy(update={"base_model": "different/model"})
    else:
        initial.data.validation_path.write_text("changed heldout")
    write_training_config(alternate.initial_sft_config, training_config_with_hash(initial))
    alternate = alternate.model_copy(
        update={"initial_sft_config_sha256": _file_sha256(alternate.initial_sft_config)}
    )
    with pytest.raises(ValueError, match="same base model and system prompt|SHA256 mismatch"):
        load_preference_data(alternate)


def test_dry_run_requires_no_ml_dependencies_or_outputs(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dpo_training, "_load_dependencies", lambda: pytest.fail("ML imported"))
    result = run_dpo_training(
        config_path=config_path, overrides=DPOTrainingOverrides(dry_run=True, max_pairs=2)
    )
    assert result["pairs"] == 2 and not result["trained"]
    assert not _config(config_path).output_dir.exists()


@pytest.mark.parametrize(
    "asset",
    [
        "train.jsonl",
        "pairs-review.jsonl",
        "manifest.json",
        "adapter_model.safetensors",
        "adapter_config.json",
    ],
)
def test_frozen_artifact_drift_is_rejected(config_path: Path, asset: str) -> None:
    config = _config(config_path)
    adapter = load_training_config(config.reference_config).runtime.expected_adapter_path
    path = (adapter if asset.startswith("adapter_") else config.preference_dir) / asset
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_preference_data(config)


def test_semantic_review_required_before_ml_imports(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dpo_training, "_load_dependencies", lambda: pytest.fail("ML imported"))
    with pytest.raises(ValueError, match="--reviewed"):
        run_dpo_training(config_path=config_path, overrides=DPOTrainingOverrides())


@pytest.mark.parametrize("answer", ["chosen", "rejected"])
def test_invalid_output_contract_is_rejected_even_with_valid_file_hashes(
    config_path: Path, answer: str
) -> None:
    config = _config(config_path)
    data = config.preference_dir
    rows = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
    review = [json.loads(line) for line in (data / "pairs-review.jsonl").read_text().splitlines()]
    malformed = '[{"name":"request_clarification","arguments":{"reason":"wrong field"}}]'
    rows[0][answer][0]["content"] = malformed
    review[0][answer][0]["content"] = malformed
    for name, contents in (("train.jsonl", rows), ("pairs-review.jsonl", review)):
        (data / name).write_text("".join(json.dumps(row) + "\n" for row in contents))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["train_sha256"] = _file_sha256(data / "train.jsonl")
    (data / "manifest.json").write_text(json.dumps(manifest))
    config = config.model_copy(
        update={
            "preference_manifest_sha256": _file_sha256(data / "manifest.json"),
            "preference_review_sha256": _file_sha256(data / "pairs-review.jsonl"),
        }
    )
    with pytest.raises(ValueError, match="Preference 1:"):
        load_preference_data(config)


def test_unavailable_tool_negatives_still_require_valid_canonical_arguments(
    config_path: Path,
) -> None:
    config = _config(config_path)
    data = config.preference_dir
    rows = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
    review = [json.loads(line) for line in (data / "pairs-review.jsonl").read_text().splitlines()]
    index = next(i for i, pair in enumerate(review) if pair["chosen_decision"] != "act")
    malformed = '[{"name":"calendar.update_event","arguments":{"title":"guessed"}}]'
    rows[index]["rejected"][0]["content"] = malformed
    review[index]["rejected"][0]["content"] = malformed
    review[index]["rejected_decision"] = "act"
    review[index]["negative_kind"] = "unavailable_tool"
    for name, contents in (("train.jsonl", rows), ("pairs-review.jsonl", review)):
        (data / name).write_text("".join(json.dumps(row) + "\n" for row in contents))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["train_sha256"] = _file_sha256(data / "train.jsonl")
    (data / "manifest.json").write_text(json.dumps(manifest))
    config = config.model_copy(
        update={
            "preference_manifest_sha256": _file_sha256(data / "manifest.json"),
            "preference_review_sha256": _file_sha256(data / "pairs-review.jsonl"),
        }
    )
    with pytest.raises(ValueError, match="arguments violate"):
        load_preference_data(config)


@pytest.mark.parametrize("version", ["0.1.0", "1.15.0"])
def test_unsupported_trl_version_fails_before_library_imports(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    monkeypatch.setattr(dpo_training.metadata, "version", lambda name: version)
    monkeypatch.setattr(dpo_training, "import_module", lambda name: pytest.fail("ML imported"))
    with pytest.raises(dpo_training.TrainingDependencyError, match="pinned TRL"):
        dpo_training._load_dependencies()


def test_output_cannot_overwrite_reference_or_previous_run(config_path: Path) -> None:
    config = _config(config_path)
    adapter = load_training_config(config.reference_config).runtime.expected_adapter_path
    for output in (adapter, adapter.parent, config.preference_dir / "new"):
        with pytest.raises(ValueError, match="overlap"):
            run_dpo_training(
                config_path=config_path,
                overrides=DPOTrainingOverrides(dry_run=True, output_dir=output),
            )
    config.output_dir.mkdir()
    (config.output_dir / "old.json").write_text("old")
    with pytest.raises(ValueError, match="not overwritten"):
        run_dpo_training(config_path=config_path, overrides=DPOTrainingOverrides(dry_run=True))


class Tokenizer:
    pad_token = "pad"
    eos_token = "eos"

    def apply_chat_template(self, messages, **kwargs):
        prefix = [1, 2, 3, 4]
        ids = prefix if len(messages) == 2 else prefix + [5, 6, 7]
        # Transformers 5.15 returns a BatchEncoding/dictionary by default.
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def save_pretrained(self, path):
        pass


def test_tokenization_reports_effective_directions_without_truncating(config_path: Path) -> None:
    pairs, _, _ = load_preference_data(_config(config_path))
    report = check_pair_tokenization(pairs, Tokenizer(), 7)
    assert report["pairs"] == 6 and report["unique_prompts"] == 3
    assert report["chosen_completion_tokens"] == 18
    assert report["truncated_pairs"] == 0 and len(report["direction_counts"]) == 6
    with pytest.raises(ValueError, match="no truncation"):
        check_pair_tokenization(pairs, Tokenizer(), 6)
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = [{"input_ids": [1, 2]}, {"input_ids": [1, 3, 4]}]
    with pytest.raises(ValueError, match="prompt/completion mismatch"):
        check_pair_tokenization(pairs, tokenizer, 10)


def test_tokenization_explicitly_requests_ids_and_correct_generation_headers(
    config_path: Path,
) -> None:
    pairs, _, _ = load_preference_data(_config(config_path))
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = [
        {"input_ids": [1, 2], "attention_mask": [1, 1]},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [1, 2, 4, 5], "attention_mask": [1, 1, 1, 1]},
    ]
    report = check_pair_tokenization(pairs[:1], tokenizer, 4)
    calls = tokenizer.apply_chat_template.call_args_list
    assert all(call.kwargs["return_dict"] is True for call in calls)
    assert [call.kwargs["add_generation_prompt"] for call in calls] == [True, False, False]
    assert report["chosen_completion_tokens"] == 1
    assert report["rejected_completion_tokens"] == 2
    assert report["longest_sequence"] == 4


class Parameter:
    def __init__(self, value: int, trainable: bool):
        self.value = value
        self.requires_grad = trainable

    def detach(self):
        return self

    def requires_grad_(self, value):
        self.requires_grad = value


def _model():
    model = MagicMock()
    model.peft_config = {"default": object(), "ref": object()}
    model.active_adapter = "default"
    policy, ref = Parameter(42, True), Parameter(42, False)
    model.named_parameters.return_value = [
        ("q.lora_A.default.weight", policy),
        ("q.lora_A.ref.weight", ref),
    ]
    model.get_parameter.return_value = ref
    return model, policy, ref


def test_reference_is_sft_copy_not_base_and_is_frozen() -> None:
    model, _, ref = _model()
    torch = SimpleNamespace(equal=lambda a, b: a.value == b.value)
    verify_reference_adapter(model, torch)
    assert not ref.requires_grad
    ref.value = 0
    with pytest.raises(ValueError, match="do not match"):
        verify_reference_adapter(model, torch)
    model.peft_config = {"default": object()}
    with pytest.raises(ValueError, match="reference adapters"):
        verify_reference_adapter(model, torch)


@pytest.mark.parametrize("alternate_start", [False, True])
def test_training_uses_existing_adapter_and_saves_only_policy(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, alternate_start: bool
) -> None:
    if alternate_start:
        config_path.write_text(_alternate_start(config_path).model_dump_json(), encoding="utf-8")
    expected_adapter = load_preference_data(_config(config_path))[1]
    model, _, _ = _model()
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.is_bf16_supported.return_value = True
    torch.equal.side_effect = lambda a, b: a.value == b.value
    transformers, peft, datasets, trl = (MagicMock() for _ in range(4))
    transformers.AutoTokenizer.from_pretrained.return_value = Tokenizer()
    peft.PeftModel.from_pretrained.return_value = model
    trl.DPOTrainer.return_value.train_dataset = [None] * 6
    deps = dict(
        zip(
            ("torch", "transformers", "peft", "datasets", "trl"),
            (torch, transformers, peft, datasets, trl),
            strict=True,
        )
    )
    monkeypatch.setattr(dpo_training, "_load_dependencies", lambda: deps)
    monkeypatch.setattr(dpo_training.metadata, "version", lambda name: "test-version")
    result = run_dpo_training(
        config_path=config_path, overrides=DPOTrainingOverrides(reviewed=True, max_steps=1)
    )
    assert result["trained"] and result["artifact"]["pairs"] == 6
    assert result["loss_backend"] == "pytorch_full_logits"
    assert result["trl_version"] == "1.14.0"
    assert peft.PeftModel.from_pretrained.call_args.kwargs == {"is_trainable": True}
    assert peft.PeftModel.from_pretrained.call_args.args[1] == str(expected_adapter)
    trainer_args = trl.DPOTrainer.call_args.kwargs
    assert trainer_args["ref_model"] is None and "peft_config" not in trainer_args
    config_args = trl.DPOConfig.call_args.kwargs
    assert config_args["precompute_ref_log_probs"] and not config_args["sync_ref_model"]
    assert config_args["use_liger_kernel"] is False
    assert config_args["max_steps"] == 1 and config_args["dataloader_num_workers"] == 0
    model.save_pretrained.assert_called_once_with(
        str(_config(config_path).output_dir / "adapter"), selected_adapters=["default"]
    )
    assert load_preference_data(_config(config_path))[1] == expected_adapter
