from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from baymax.models.sft_dataset import SFTSourceRowRef, convert_xlam_row_to_sft_example
from baymax.models.sft_training import _tokenize_chat_example
from baymax.models.training_config import load_training_config
from scripts import report_sft_loss as diagnostic


def _row(label: str, *, prompt: str = "unit request", detail: str = "detail") -> dict[str, Any]:
    calls = [{"name": "unit.store", "arguments": {"title": detail}}]
    if label == "clarify":
        calls = [{"name": "request_clarification", "arguments": {"question": detail}}]
    elif label == "refuse":
        calls = [{"name": "refuse_request", "arguments": {"reason": detail}}]
    elif label == "multi":
        calls *= 2
    return {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(calls)},
        ]
    }


class CharacterTokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        ids = list(map(ord, text))
        if kwargs.get("truncation"):
            ids = ids[: kwargs["max_length"]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeTorch:
    long = "long"
    active = False

    @contextmanager
    def inference_mode(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def tensor(self, values: list[list[int]], *, dtype: str, device: str) -> list[list[int]]:
        assert dtype == self.long and device == "test-device"
        return values


class FakeModel:
    def __init__(self, torch: FakeTorch, losses: list[float]) -> None:
        self.torch = torch
        self.losses = iter(losses)
        self.calls: list[dict[str, Any]] = []
        self.evaluating = False
        self.config = SimpleNamespace(_commit_hash="unit-model-commit")

    def eval(self) -> None:
        self.evaluating = True

    def parameters(self):
        yield SimpleNamespace(device="test-device", dtype="test-dtype")

    def __call__(self, **kwargs: Any) -> Any:
        assert self.evaluating and self.torch.active
        assert kwargs.pop("use_cache") is False
        assert kwargs.pop("return_dict") is True
        self.calls.append(kwargs)
        loss = next(self.losses)
        return SimpleNamespace(loss=SimpleNamespace(item=lambda: loss))


def _score(rows: list[dict[str, Any]], losses: list[float], cap: int = 4096):
    torch = FakeTorch()
    model = FakeModel(torch, losses)
    report = diagnostic.score_rows(
        rows,
        tokenizer=CharacterTokenizer(),
        model=model,
        torch=torch,
        max_sequence_length=cap,
    )
    return report, model


def test_scores_masked_answers_in_eval_mode_and_weights_class_loss_by_tokens() -> None:
    rows = [_row("act"), _row("multi"), _row("clarify"), _row("refuse")]
    report, model = _score(rows, [1.0, 3.0, 4.0, 2.0])
    for row, call in zip(rows, model.calls, strict=True):
        expected = _tokenize_chat_example(row, CharacterTokenizer(), 4096)
        assert call == {key: [values] for key, values in expected.items()}
        assert -100 in call["labels"][0]
        assert call["labels"][0][-1] == call["input_ids"][0][-1]
    stats = report["by_class"]["act"]
    first, second = report["rows"][:2]
    expected_loss = (first["scored_tokens"] + 3 * second["scored_tokens"]) / stats["scored_tokens"]
    assert stats["scored_rows"] == 2
    assert stats["token_mean_loss"] == pytest.approx(expected_loss)
    assert stats["row_mean_loss"] == 2.0
    assert stats["token_mean_loss"] != stats["row_mean_loss"]
    assert report["by_class"]["clarify"]["token_mean_loss"] == 4.0
    assert report["by_class"]["refuse"]["token_mean_loss"] == 2.0
    assert report["overall"]["scored_rows"] == 4


def test_skips_zero_label_rows_but_reports_partially_truncated_answers() -> None:
    rows = [_row("clarify"), _row("act", prompt="x" * 1000), _row("refuse", detail="y" * 1000)]
    tokenizer = CharacterTokenizer()
    cap = len(
        tokenizer.apply_chat_template(
            rows[0]["messages"], tokenize=False, add_generation_prompt=False
        )
    )
    report, model = _score(rows, [1.0, 2.0], cap)
    assert len(model.calls) == 2
    assert report["rows"][1]["loss"] is None
    assert report["rows"][1]["scored_tokens"] == 0
    assert report["rows"][2]["loss"] == 2.0
    assert report["overall"]["truncated_rows"] == 2
    assert report["by_class"]["act"]["skipped_rows"] == 1
    assert report["by_class"]["act"]["token_mean_loss"] is None


def test_token_weighting_excludes_unpredictable_position_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostic,
        "_tokenize_chat_example",
        lambda *_: {"input_ids": [1, 2, 3], "labels": [1, 2, -100]},
    )
    report, _ = _score([_row("clarify")], [2.0])
    assert report["overall"]["scored_tokens"] == 1
    assert report["overall"]["total_negative_log_likelihood"] == 2.0


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -0.1])
def test_rejects_invalid_numeric_losses(loss: float) -> None:
    with pytest.raises(ValueError, match="Row 1: non-finite or negative loss"):
        _score([_row("act")], [loss])


def test_rejects_all_masked_data_and_malformed_targets() -> None:
    with pytest.raises(ValueError, match="No assistant tokens"):
        _score([_row("clarify")], [], cap=1)
    with pytest.raises(ValueError, match="malformed"):
        _score([{"messages": []}], [])


class OffsetTokenizer(CharacterTokenizer):
    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        encoded: dict[str, Any] = super().__call__(text, **kwargs)
        if kwargs.get("return_offsets_mapping"):
            encoded["offset_mapping"] = [
                (index, index + 1) for index in range(len(encoded["input_ids"]))
            ]
        return encoded


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_json_spans_follow_field_paths_with_escapes_unicode_and_either_key_order(
    compact: bool, ensure_ascii: bool
) -> None:
    question = 'Which "route"\\unit?\nCaf\u00e9?'
    calls = [{"name": "request_clarification", "arguments": {"question": question}}]
    target = json.dumps(
        calls,
        ensure_ascii=ensure_ascii,
        sort_keys=not compact,
        separators=(",", ":") if compact else None,
    )
    spans = diagnostic.target_component_spans(target)
    assert [(part, target[start:end]) for part, start, end in spans] == [
        ("first_tool_name", "request_clarification"),
        ("question_text", json.dumps(question, ensure_ascii=ensure_ascii)[1:-1]),
    ]


def test_json_spans_do_not_confuse_argument_names_or_tool_names_mentioned_in_text() -> None:
    calls = [
        {
            "arguments": {"name": "refuse_request", "question": "not a meta-tool"},
            "name": "unit.first",
        },
        {"name": "unit.second", "arguments": {"reason": "not refusal text"}},
    ]
    target = json.dumps(calls)
    assert [
        (part, target[start:end]) for part, start, end in diagnostic.target_component_spans(target)
    ] == [("first_tool_name", "unit.first"), ("later_tool_name", "unit.second")]
    refusal = _row("refuse", detail="request_clarification is not the chosen tool")
    text = refusal["messages"][-1]["content"]
    assert [
        (part, text[start:end]) for part, start, end in diagnostic.target_component_spans(text)
    ] == [
        ("first_tool_name", "refuse_request"),
        ("reason_text", "request_clarification is not the chosen tool"),
    ]


@pytest.mark.parametrize("cap", [4096, 120, 1])
def test_component_offsets_partition_only_visible_shifted_assistant_labels(cap: int) -> None:
    row = _row("clarify", detail="Which unit?")
    tokenizer = OffsetTokenizer()
    tokenized = _tokenize_chat_example(row, tokenizer, cap)
    parts = diagnostic.token_components(row, tokenizer, tokenized, cap)
    assert len(parts) == len(tokenized["labels"])
    assert sum(part is not None for part in parts) == sum(
        label != -100 for label in tokenized["labels"][1:]
    )
    for index, label in enumerate(tokenized["labels"]):
        if not index or label == -100:
            assert parts[index] is None
    if cap == 4096:
        assert parts.count("first_tool_name") == len("request_clarification")
        assert parts.count("question_text") == len("Which unit?")
        assert parts[-1] == "other"


def test_tokens_crossing_json_quotes_are_separate_not_counted_as_pure_tool_name() -> None:
    class MergedTokenizer(OffsetTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            joined = text.find('"request_clarification"')
            offsets: list[tuple[int, int]] = []
            ids: list[int] = []
            cursor = 0
            while cursor < len(text):
                end = cursor + (2 if cursor == joined else 1)
                offsets.append((cursor, end))
                ids.append(1000 if cursor == joined else ord(text[cursor]))
                cursor = end
            if kwargs.get("truncation"):
                ids, offsets = ids[: kwargs["max_length"]], offsets[: kwargs["max_length"]]
            result: dict[str, Any] = {"input_ids": ids, "attention_mask": [1] * len(ids)}
            if kwargs.get("return_offsets_mapping"):
                result["offset_mapping"] = offsets
            return result

    row = _row("clarify")
    tokenizer = MergedTokenizer()
    tokenized = _tokenize_chat_example(row, tokenizer, 4096)
    parts = diagnostic.token_components(row, tokenizer, tokenized, 4096)
    assert parts.count("mixed_boundary") == 1
    assert parts.count("first_tool_name") == len("request_clarification") - 1


def test_component_mapping_rejects_unavailable_or_changed_offsets() -> None:
    row = _row("act")
    tokenizer = CharacterTokenizer()
    tokenized = _tokenize_chat_example(row, tokenizer, 4096)
    with pytest.raises(ValueError, match="fast tokenizer"):
        diagnostic.token_components(row, tokenizer, tokenized, 4096)

    class ChangedTokenizer(OffsetTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            encoded = super().__call__(text, **kwargs)
            if kwargs.get("return_offsets_mapping"):
                encoded["input_ids"][0] += 1
            return encoded

    with pytest.raises(ValueError, match="actual training input IDs"):
        diagnostic.token_components(row, ChangedTokenizer(), tokenized, 4096)
    duplicate = '[{"name":"unit.first","name":"unit.second","arguments":{}}]'
    with pytest.raises(ValueError, match="unambiguous"):
        diagnostic.target_component_spans(duplicate)


def test_component_cross_entropy_matches_shifted_reference_across_chunks() -> None:
    torch = pytest.importorskip("torch")
    length = 150
    logits = torch.arange(length * 3, dtype=torch.float32).reshape(1, length, 3) / 100
    labels = [-100 if index % 11 == 0 else index % 3 for index in range(length)]
    parts = [
        None
        if not index or label == -100
        else ("first_tool_name" if index % 2 else "question_text")
        for index, label in enumerate(labels)
    ]
    result = diagnostic.component_losses(logits, {"labels": labels}, parts, torch)
    reference = torch.nn.functional.cross_entropy(
        logits[0, :-1], torch.tensor(labels[1:]), reduction="none", ignore_index=-100
    )
    for part in diagnostic.COMPONENTS:
        selected = [
            float(reference[index - 1]) for index in range(1, length) if parts[index] == part
        ]
        assert result[part]["scored_tokens"] == len(selected)
        if selected:
            assert result[part]["loss"] == pytest.approx(math.fsum(selected) / len(selected))
        else:
            assert result[part]["loss"] is None


def test_component_mode_preserves_overall_loss_and_reconciles_all_token_counts() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = OffsetTokenizer()
    rows = [_row("act"), _row("multi"), _row("clarify"), _row("refuse")]

    class UniformModel:
        def eval(self) -> None:
            pass

        def parameters(self):
            yield torch.empty(0)

        def __call__(self, **kwargs: Any) -> Any:
            assert kwargs["use_cache"] is False and kwargs["return_dict"] is True
            assert not torch.is_grad_enabled()
            labels = kwargs["labels"]
            logits = torch.zeros(1, labels.shape[1], 128)
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1], labels[0, 1:], ignore_index=-100
            )
            return SimpleNamespace(logits=logits, loss=loss)

    model = UniformModel()
    plain = diagnostic.score_rows(
        rows, tokenizer=tokenizer, model=model, torch=torch, max_sequence_length=4096
    )
    report = diagnostic.score_rows(
        rows,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        max_sequence_length=4096,
        by_component=True,
    )
    assert plain["overall"] == report["overall"]
    assert report["overall"]["token_mean_loss"] == pytest.approx(math.log(128))
    assert (
        sum(part["scored_tokens"] for part in report["by_component"].values())
        == report["overall"]["scored_tokens"]
    )
    for label, stats in report["by_class"].items():
        assert (
            sum(part["scored_tokens"] for part in stats["components"].values())
            == stats["scored_tokens"]
        )
        for part in stats["components"].values():
            if part["scored_tokens"]:
                assert part["token_mean_loss"] == pytest.approx(math.log(128))
    assert report["by_class"]["act"]["components"]["later_tool_name"]["scored_tokens"] > 0


@pytest.fixture
def setup_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reference = load_training_config(diagnostic.DEFAULT_CONFIG)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    example = convert_xlam_row_to_sft_example(
        {
            "query": "unit request",
            "tools": [{"name": "unit.store", "parameters": {"type": "object"}}],
            "answers": [{"name": "unit.store", "arguments": {}}],
        },
        system_prompt="policy",
    ).model_dump(mode="json")
    for split in ("train", "validation", "test"):
        (dataset / f"{split}.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")
    manifest = dataset / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    adapter = tmp_path / "model" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": reference.base_model,
                "peft_type": "LORA",
                "r": reference.lora.r,
                "lora_alpha": reference.lora.lora_alpha,
                "target_modules": reference.lora.target_modules,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"unit-test weights")
    config = reference.model_copy(
        update={
            "data": reference.data.model_copy(
                update={
                    "split_manifest_path": manifest,
                    "system_prompt_sha256": hashlib.sha256(b"policy").hexdigest(),
                    **{
                        f"{split}_path": dataset / f"{split}.jsonl"
                        for split in ("train", "validation", "test")
                    },
                }
            ),
            "runtime": reference.runtime.model_copy(
                update={"expected_adapter_path": adapter, "output_dir": adapter.parent}
            ),
        }
    )
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic, "validate_training_config", lambda *_, **__: SimpleNamespace(valid=True)
    )
    monkeypatch.setattr(
        diagnostic,
        "load_split_source_rows",
        lambda *_, **__: [SFTSourceRowRef(source_index=0, source_row_id="unit:0", row_hash="unit")],
    )
    monkeypatch.setattr(
        diagnostic, "_load_model", lambda *_, **__: pytest.fail("No model in preflight")
    )
    return path, config, tmp_path / "report.json"


@pytest.mark.parametrize("by_component", [False, True])
def test_dry_run_preserves_inputs_and_records_hashes_without_loading_model(
    setup_report, by_component: bool
) -> None:
    path, config, output = setup_report
    before = config.data.train_path.read_bytes()
    report = diagnostic.run_report(
        config_path=path, output=output, max_rows=1, dry_run=True, by_component=by_component
    )
    assert report["measured"] is False
    assert report["raw_class_counts"] == {"act": 1}
    assert "overall" not in report
    assert len(report["sha256"]["adapter_weights"]) == 64
    assert json.loads(output.read_text()) == report
    assert config.data.train_path.read_bytes() == before
    assert ("component_definition" in report) is by_component
    assert report["diagnostic"] == (
        "sft-class-component-loss-v1" if by_component else "sft-class-loss-v1"
    )


def test_measured_report_saves_per_row_losses_sources_and_checkpoint_identity(
    setup_report, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, config, output = setup_report
    torch = FakeTorch()
    model = FakeModel(torch, [2.5])
    monkeypatch.setattr(
        diagnostic, "_load_model", lambda *_, **__: (torch, model, CharacterTokenizer())
    )
    monkeypatch.setattr(diagnostic, "version", lambda _: "unit-version")
    before = config.data.train_path.read_bytes()
    report = diagnostic.run_report(
        config_path=path, output=output, max_rows=1, max_sequence_length=4096
    )
    assert report["measured"] is True
    assert report["overall"]["token_mean_loss"] == 2.5
    assert report["by_class"]["act"]["scored_rows"] == 1
    assert report["by_class"]["clarify"]["scored_rows"] == 0
    assert report["rows"][0]["source"]["source_row_id"] == "unit:0"
    assert report["model_revision"] == "unit-model-commit"
    assert report["model_dtype"] == "test-dtype"
    assert json.loads(output.read_text()) == report
    assert config.data.train_path.read_bytes() == before


@pytest.mark.parametrize("failure", ["existing", "protected", "hash", "adapter", "short", "prompt"])
def test_preflight_rejects_invalid_inputs_before_loading_model(
    setup_report, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path, config, output = setup_report
    expected = ""
    cap = 1
    if failure == "existing":
        output.write_text("preserve", encoding="utf-8")
        expected = "already exists"
    elif failure == "protected":
        output = config.runtime.expected_adapter_path / "new.json"
        expected = "outside dataset and model"
    elif failure == "hash":
        monkeypatch.setattr(
            diagnostic, "validate_training_config", lambda *_, **__: SimpleNamespace(valid=False)
        )
        expected = "config/hash/manifest"
    elif failure == "adapter":
        (config.runtime.expected_adapter_path / "adapter_config.json").write_text(
            "{}", encoding="utf-8"
        )
        expected = "Adapter metadata"
    elif failure == "short":
        cap = 2
        expected = "shorter"
    elif failure == "prompt":
        row = json.loads(config.data.train_path.read_text())
        row["messages"][0]["content"] = "wrong policy"
        config.data.train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        expected = "system prompt hash"
    with pytest.raises(ValueError, match=expected):
        diagnostic.run_report(config_path=path, output=output, max_rows=cap, dry_run=True)
    if failure == "existing":
        assert output.read_text() == "preserve"
    else:
        assert not output.exists()
