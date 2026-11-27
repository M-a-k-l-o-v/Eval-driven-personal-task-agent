from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from baymax.models.sft_dataset import SFTSourceRowRef
from baymax.models.sft_loss_weighting import (
    WeightedDataCollator,
    first_tool_name_weights,
    weighted_causal_loss,
    weighted_trainer_class,
)
from baymax.models.sft_token_spans import token_components
from baymax.models.sft_training import _tokenize_chat_example, tokenize_sft_rows_for_training


class CharacterTokenizer:
    is_fast = True
    padding_side = "right"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        text = "".join(f"<{message['role']}>{message['content']}<end>" for message in messages)
        return text + "<assistant>" if kwargs["add_generation_prompt"] else text

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids = list(map(ord, text))
        if kwargs.get("truncation"):
            ids = ids[: kwargs["max_length"]]
        result: dict[str, Any] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(index, index + 1) for index in range(len(ids))]
        return result


def _row(label: str) -> dict[str, Any]:
    calls = [{"name": "unit.store", "arguments": {"text": "request_clarification"}}]
    if label == "clarify":
        calls = [{"name": "request_clarification", "arguments": {"question": "Which unit?"}}]
    elif label == "refuse":
        calls = [{"name": "refuse_request", "arguments": {"reason": "Unsupported operation."}}]
    elif label == "multi":
        calls *= 2
    return {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "unit request"},
            {"role": "assistant", "content": json.dumps(calls)},
        ]
    }


@pytest.mark.parametrize("label", ["act", "multi", "clarify", "refuse"])
@pytest.mark.parametrize("weight", [2.0, 3.0])
def test_only_first_tool_name_gets_extra_weight_and_inputs_stay_identical(
    label: str, weight: float
) -> None:
    row = _row(label)
    original = json.dumps(row)
    tokenizer = CharacterTokenizer()
    baseline = tokenize_sft_rows_for_training(
        rows=[row], tokenizer=tokenizer, max_sequence_length=4096, split_name="train"
    )
    weighted = tokenize_sft_rows_for_training(
        rows=[row],
        tokenizer=tokenizer,
        max_sequence_length=4096,
        split_name="train",
        first_tool_name_weight=weight,
    )
    assert {key: value for key, value in weighted[0].items() if key != "loss_weights"} == baseline[
        0
    ]
    parts = token_components(row, tokenizer, baseline[0], 4096)
    assert any(part == "first_tool_name" for part in parts)
    assert weighted[0]["loss_weights"] == [
        0 if part is None else weight if part == "first_tool_name" else 1 for part in parts
    ]
    assert json.dumps(row) == original


def test_weighting_preserves_no_label_filtering_and_does_not_require_offsets_by_default() -> None:
    row = _row("clarify")
    tokenizer = CharacterTokenizer()
    tokenizer.is_fast = False
    default = tokenize_sft_rows_for_training(
        rows=[row], tokenizer=tokenizer, max_sequence_length=4096, split_name="train"
    )
    assert "loss_weights" not in default[0]
    with pytest.raises(ValueError, match="train row 1.*fast tokenizer"):
        tokenize_sft_rows_for_training(
            rows=[row],
            tokenizer=tokenizer,
            max_sequence_length=4096,
            split_name="train",
            first_tool_name_weight=3,
        )
    with pytest.raises(ValueError, match="no rows with trainable"):
        tokenize_sft_rows_for_training(
            rows=[row],
            tokenizer=tokenizer,
            max_sequence_length=1,
            split_name="train",
            first_tool_name_weight=3,
        )


@pytest.mark.parametrize("weight", [0.5, float("nan"), float("inf")])
def test_weight_builder_rejects_invalid_weights(weight: float) -> None:
    row = _row("act")
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="finite and at least 1"):
        first_tool_name_weights(
            row, tokenizer, _tokenize_chat_example(row, tokenizer, 4096), 4096, weight
        )


def _torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("factor", [1.0, 2.0, 3.0])
def test_weighted_loss_and_gradients_match_explicit_shifted_cross_entropy(factor: float) -> None:
    torch = _torch()
    logits = torch.tensor(
        [[[1.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 2.0], [3.0, 1.0, 0.0]]], requires_grad=True
    )
    reference = logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[-100, 1, 2, -100]])
    weights = torch.tensor([[99.0, factor, 1.0, 99.0]])
    actual = weighted_causal_loss(logits, labels, weights, torch)
    nll = torch.nn.functional.cross_entropy(
        reference[0, :-1], labels[0, 1:], reduction="none", ignore_index=-100
    )
    expected = (factor * nll[0] + nll[1]) / (factor + 1)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("side", ["left", "right"])
def test_collator_pads_weights_with_zero_and_validation_defaults_to_unweighted(side: str) -> None:
    torch = _torch()
    features = [
        {"labels": [-100, 1, 2], "loss_weights": [0.0, 3.0, 1.0]},
        {"labels": [-100, 2]},
    ]
    before = json.dumps(features)

    def base(items):
        assert all("loss_weights" not in item for item in items)

        def pad(values):
            return (
                values + [-100] * (3 - len(values))
                if side == "right"
                else [-100] * (3 - len(values)) + values
            )

        return {"labels": torch.tensor([pad(item["labels"]) for item in items])}

    batch = WeightedDataCollator(base, torch, side)(features)
    expected = [0.0, 1.0, 0.0] if side == "right" else [0.0, 0.0, 1.0]
    assert batch["loss_weights"].tolist() == [[0.0, 3.0, 1.0], expected]
    assert json.dumps(features) == before


class FakeBaseTrainer:
    def __init__(self, *, args: Any = None, **kwargs: Any) -> None:
        self.args = args or SimpleNamespace(world_size=1, n_gpu=0)

    def compute_loss(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
        raise AssertionError("The weighted trainer must use its own loss.")

    def _get_num_items_in_batch(self, batch_samples: Any, device: Any) -> Any:
        return None


def test_accumulation_window_normalization_matches_combined_loss_and_gradient() -> None:
    torch = _torch()
    trainer = weighted_trainer_class(FakeBaseTrainer, torch)()
    assert trainer.model_accepts_loss_kwargs is True
    batches = [
        {"labels": torch.tensor([[-100, 0, 1]]), "loss_weights": torch.tensor([[0.0, 3.0, 1.0]])},
        {
            "labels": torch.tensor([[-100, 1, -100]]),
            "loss_weights": torch.tensor([[0.0, 1.0, 99.0]]),
        },
    ]
    denominator = trainer._get_num_items_in_batch(batches, "cpu")
    assert float(denominator) == 5.0
    assert trainer._get_num_items_in_batch([], "cpu") is None
    parameter = torch.tensor(0.2, requires_grad=True)
    logits = torch.stack([parameter.expand(3), -parameter.expand(3)], dim=-1).unsqueeze(0)
    losses = [
        weighted_causal_loss(
            logits, batch["labels"], batch["loss_weights"], torch, normalizer=denominator
        )
        for batch in batches
    ]
    combined = weighted_causal_loss(
        logits.expand(2, -1, -1),
        torch.cat([batch["labels"] for batch in batches]),
        torch.cat([batch["loss_weights"] for batch in batches]),
        torch,
    )
    torch.testing.assert_close(sum(losses), combined)
    accumulated_grad = torch.autograd.grad(sum(losses), parameter, retain_graph=True)[0]
    combined_grad = torch.autograd.grad(combined, parameter)[0]
    torch.testing.assert_close(accumulated_grad, combined_grad)
    assert float(trainer._get_num_items_in_batch(batches[:1], "cpu")) == 4.0


def test_trainer_strips_loss_metadata_and_validation_remains_plain_cross_entropy() -> None:
    torch = _torch()
    trainer = weighted_trainer_class(FakeBaseTrainer, torch)()
    batch = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "labels": torch.tensor([[-100, 1, 2]]),
        "loss_weights": torch.tensor([[0.0, 1.0, 1.0]]),
    }
    output = SimpleNamespace(logits=torch.zeros(1, 3, 3))
    model = Mock(return_value=output)
    loss, returned = trainer.compute_loss(model, batch, return_outputs=True)
    assert returned is output
    assert float(loss) == pytest.approx(math.log(3))
    model.assert_called_once_with(input_ids=batch["input_ids"], use_cache=False, return_dict=True)
    assert "labels" in batch and "loss_weights" in batch
    with pytest.raises(ValueError, match="one training device"):
        weighted_trainer_class(FakeBaseTrainer, torch)(args=SimpleNamespace(world_size=2, n_gpu=1))


def test_weighted_training_wiring_uses_fresh_adapter_and_saves_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from baymax.models import sft_training
    from baymax.models.training_config import load_training_config, training_config_with_hash

    torch = _torch()
    reference = load_training_config(
        Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first_tool_weight3.json"
        )
    )
    config = training_config_with_hash(
        reference.model_copy(
            update={
                "runtime": reference.runtime.model_copy(
                    update={
                        "output_dir": tmp_path / "model",
                        "expected_adapter_path": tmp_path / "model/adapter",
                    }
                )
            }
        )
    )
    overrides = sft_training.SFTTrainingOverrides(allow_cpu=True)
    plan = sft_training.build_training_plan(
        config=config, overrides=overrides, train_rows=1, eval_rows=1
    )
    tokenizer: Any = CharacterTokenizer()
    tokenizer.pad_token = "pad"
    tokenizer.save_pretrained = Mock()
    model = Mock()
    captured = {}

    class StubTrainer(FakeBaseTrainer):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            captured.update(kwargs)

        def train(self) -> None:
            captured["train_called"] = True

    transformers = SimpleNamespace(
        set_seed=Mock(),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        TrainingArguments=lambda **_: SimpleNamespace(world_size=1, n_gpu=0),
        Trainer=StubTrainer,
        DataCollatorForSeq2Seq=Mock(return_value=Mock()),
    )
    dependencies = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            **{
                key: getattr(torch, key)
                for key in ("float32", "tensor", "stack", "nn", "as_tensor", "isfinite")
            },
        ),
        "transformers": transformers,
        "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
        "peft": SimpleNamespace(
            LoraConfig=Mock(),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=Mock(return_value=model),
        ),
    }
    monkeypatch.setattr(sft_training, "_load_training_dependencies", lambda: dependencies)
    source = [SFTSourceRowRef(source_index=0, source_row_id="unit:0", row_hash="unit")]
    sft_training._train_with_huggingface(
        config=config,
        overrides=overrides,
        plan=plan,
        train_rows=[_row("clarify")],
        eval_rows=[_row("clarify")],
        train_source_rows=source,
        eval_source_rows=source,
    )
    assert "loss_weights" in captured["train_dataset"][0]
    assert "loss_weights" not in captured["eval_dataset"][0]
    assert isinstance(captured["data_collator"], WeightedDataCollator)
    assert captured["train_called"] is True
    transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
        config.base_model, trust_remote_code=False
    )
    dependencies["peft"].get_peft_model.assert_called_once()
    metadata = json.loads((plan.output_dir / "weighted-loss-run.json").read_text())
    assert metadata["config"]["config_hash"] == config.config_hash
    assert metadata["plan"]["first_tool_name_weight"] == 3.0
    assert metadata["weighted_first_tool_name_tokens"] == len("request_clarification")
    assert metadata["validation_loss"].startswith("unweighted")
    model.save_pretrained.assert_called_once_with(plan.adapter_path)
