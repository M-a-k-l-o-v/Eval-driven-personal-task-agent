from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from baymax.eval.lora_runner import (
    LocalLoraAgent,
    LocalLoraConfig,
    LocalLoraInvocation,
    parse_lora_agent_response,
)
from baymax.eval.scenario_loader import Scenario
from baymax.models.sft_dataset import (
    SFTExample,
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SFTSplitRatios,
    convert_xlam_row_to_sft_example,
)
from baymax.models.training_config import (
    load_training_config,
    training_config_with_hash,
    write_training_config,
)
from scripts.probe_sft_clarification import benchmark_json_example, main, run_probe

REFERENCE = Path(__file__).resolve().parents[2] / (
    "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json"
)


def _example(query: str, decision: str) -> dict[str, Any]:
    calls = [{"name": "unit.store", "arguments": {"title": "unit item"}}]
    if decision == "clarify":
        calls = [{"name": "request_clarification", "arguments": {"question": f"Which {query}?"}}]
    elif decision == "refuse":
        calls = [
            {"name": "refuse_request", "arguments": {"reason": "That operation is unsupported."}}
        ]
    elif decision == "multi-tool":
        calls *= 2
    return convert_xlam_row_to_sft_example(
        {
            "query": query,
            "tools": [
                {
                    "name": "unit.store",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                }
            ],
            "answers": calls,
        },
        system_prompt="Unit test prompt",
    ).model_dump(mode="json")


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    reference = load_training_config(REFERENCE)
    examples = {
        "train": [_example("training one", "act"), _example("training two", "act")],
        "validation": [
            _example("heldout action", "act"),
            _example("heldout one", "clarify"),
            _example("heldout two", "clarify"),
            _example("heldout one", "clarify"),
            _example("heldout refusal", "refuse"),
        ],
        "test": [_example("untouched test", "refuse")],
    }
    entries = {}
    source_index = 0
    for split, rows in examples.items():
        path = dataset / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        refs = [
            SFTSourceRowRef(
                source_index=source_index + i,
                source_row_id=f"baymax:unit-{source_index + i}",
                row_hash=str(i),
            )
            for i in range(len(rows))
        ]
        source_index += len(rows)
        entries[split] = SFTSplitManifestEntry(path=str(path), count=len(rows), source_rows=refs)
    manifest = SFTSplitManifest(
        format_version=reference.data.format_version,
        config_hash="unit-source",
        dataset_id=reference.data.dataset_id,
        revision="unit-revision",
        source_split="mixed",
        seed=1,
        max_rows=None,
        split_ratios=SFTSplitRatios(),
        system_prompt_sha256=hashlib.sha256(b"Unit test prompt").hexdigest(),
        output_dir=str(dataset),
        splits=entries,
        loaded_rows=source_index,
        converted_rows=source_index,
        skipped_rows=[],
    )
    manifest_path = dataset / "split-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    model_dir = tmp_path / "model"
    adapter = model_dir / "adapter"
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
    (adapter / "adapter_model.safetensors").write_bytes(b"Fake weights for unit test only")
    data = reference.data.model_copy(
        update={
            "revision": manifest.revision,
            "system_prompt_sha256": manifest.system_prompt_sha256,
            "split_manifest_path": manifest_path,
            "validation_report_path": dataset / "validation-report.json",
            **{f"{split}_path": Path(entry.path) for split, entry in entries.items()},
        }
    )
    config = training_config_with_hash(
        reference.model_copy(
            update={
                "data": data,
                "runtime": reference.runtime.model_copy(
                    update={
                        "output_dir": model_dir,
                        "expected_adapter_path": adapter,
                    }
                ),
            }
        )
    )
    config_path = tmp_path / "probe-config.json"
    write_training_config(config_path, config)
    return config_path, dataset, tmp_path / "probe.json"


def _probe(config: Path, output: Path, **kwargs: Any) -> dict[str, Any]:
    return run_probe(
        config_path=config, output=output, limit=2, validation_cap=5, train_cap=2, **kwargs
    )


def test_dry_run_selects_unique_heldout_rows_without_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset, output = _setup(tmp_path)
    before = {path.name: path.read_bytes() for path in dataset.iterdir()}
    monkeypatch.setattr(
        "scripts.probe_sft_clarification.LocalLoraAgent", lambda _: pytest.fail("No model")
    )
    report = _probe(config, output, dry_run=True)
    assert not report["generated"] and report["summary"] is None
    assert report["available_unique_clarification_rows"] == 2
    assert [row["validation_row"] for row in report["rows"]] == [2, 3]
    assert all(row["expected_decision"] == "clarify" for row in report["rows"])
    assert {path.name: path.read_bytes() for path in dataset.iterdir()} == before
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("behavior", ["refuse", "multi-tool"])
def test_new_behavior_modes_select_baymax_targets_without_loading_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: str
) -> None:
    config, dataset, output = _setup(tmp_path)
    rows = [
        _example("external input", behavior),
        _example("heldout one", behavior),
        _example("heldout two", behavior),
        _example("heldout one", behavior),
        _example("single action", "act"),
    ]
    (dataset / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = dataset / "split-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["validation"]["source_rows"][0]["source_row_id"] = 123
    manifest_path.write_text(json.dumps(manifest))
    before = {path.name: path.read_bytes() for path in dataset.iterdir()}
    monkeypatch.setattr(
        "scripts.probe_sft_clarification.LocalLoraAgent", lambda _: pytest.fail("No model")
    )
    report = _probe(config, output, behavior=behavior, dry_run=True)
    assert report["summary"] is None and not report["generated"]
    assert report["source_scope"] == "baymax"
    assert [row["validation_row"] for row in report["rows"]] == [2, 3]
    assert all(
        row["expected_decision"] == ("act" if behavior == "multi-tool" else "refuse")
        for row in report["rows"]
    )
    if behavior == "multi-tool":
        assert all(
            row["expected_tool_sequence"] == ["unit.store", "unit.store"] for row in report["rows"]
        )
    assert {path.name: path.read_bytes() for path in dataset.iterdir()} == before


@pytest.mark.parametrize("variant", ["complete", "missing_second", "mixed_meta"])
def test_multi_tool_probe_distinguishes_routing_from_sequence_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    config, dataset, output = _setup(tmp_path)
    example = _example("heldout sequence", "multi-tool")
    rows = [example, *[_example(f"other {i}", "act") for i in range(4)]]
    (dataset / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    calls = json.loads(example["messages"][-1]["content"])
    if variant == "missing_second":
        calls = calls[:1]
    elif variant == "mixed_meta":
        calls[1] = {"name": "request_clarification", "arguments": {"question": "Which unit?"}}
    raw = json.dumps(calls)
    inputs = []

    class FakeAgent:
        def __init__(self, settings: LocalLoraConfig):
            assert (
                settings.adapter_path == load_training_config(config).runtime.expected_adapter_path
            )

        def invoke_messages(self, messages: list[dict[str, str]]) -> LocalLoraInvocation:
            inputs.append(messages)
            return LocalLoraInvocation(raw_text=raw, response=parse_lora_agent_response(raw))

    monkeypatch.setattr("scripts.probe_sft_clarification.LocalLoraAgent", FakeAgent)
    report = run_probe(
        config_path=config,
        output=output,
        behavior="multi-tool",
        limit=1,
        validation_cap=5,
        train_cap=2,
    )
    row = report["rows"][0]
    assert inputs == [example["messages"][:-1]]
    assert row["expected_tool_sequence"] == ["unit.store", "unit.store"]
    assert row["tool_sequence_matches"] == (variant == "complete")
    assert report["summary"]["matching_tool_sequences"] == int(variant == "complete")
    assert row["correct_decision"] == (variant != "mixed_meta")
    assert (row["agent_response"]["parse_error"] is not None) == (variant == "mixed_meta")
    assert row["raw_model_output"] == raw


def test_refusal_probe_counts_decisions_without_requiring_exact_reason_wording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset, output = _setup(tmp_path)
    example = _example("heldout refusal", "refuse")
    rows = [example, *[_example(f"other {i}", "act") for i in range(4)]]
    (dataset / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    raw = '[{"name":"refuse_request","arguments":{"reason":"A different explanation."}}]'

    class FakeAgent:
        def __init__(self, _: LocalLoraConfig):
            pass

        def invoke_messages(self, messages: list[dict[str, str]]) -> LocalLoraInvocation:
            assert messages == example["messages"][:-1]
            return LocalLoraInvocation(raw_text=raw, response=parse_lora_agent_response(raw))

    monkeypatch.setattr("scripts.probe_sft_clarification.LocalLoraAgent", FakeAgent)
    report = run_probe(
        config_path=config, output=output, behavior="refuse", limit=1, validation_cap=5, train_cap=2
    )
    assert report["rows"][0]["correct_decision"]
    assert report["summary"]["correct_refusal_decisions"] == 1
    assert report["summary"]["raw_outputs_mentioning_refusal"] == 1
    assert "task_success" not in report["summary"]


@pytest.mark.parametrize("behavior", ["refuse", "multi-tool"])
def test_new_behavior_modes_reject_exact_training_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: str
) -> None:
    config, dataset, output = _setup(tmp_path)
    example = _example("overlap", behavior)
    (dataset / "train.jsonl").write_text(
        json.dumps(example) + "\n" + json.dumps(_example("other training", "act")) + "\n"
    )
    (dataset / "validation.jsonl").write_text(
        json.dumps(example)
        + "\n"
        + "".join(json.dumps(_example(f"other {i}", "act")) + "\n" for i in range(4))
    )
    monkeypatch.setattr(
        "scripts.probe_sft_clarification.LocalLoraAgent", lambda _: pytest.fail("No model")
    )
    with pytest.raises(ValueError, match="prompt overlaps training"):
        run_probe(
            config_path=config,
            output=output,
            behavior=behavior,
            limit=1,
            validation_cap=5,
            train_cap=2,
        )
    assert not output.exists()


def test_probe_uses_exact_inputs_and_counts_decisions_without_tools_or_wording_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset, output = _setup(tmp_path)
    inputs = []
    settings = []
    raw_outputs = iter(
        [
            '[{"name":"request_clarification","arguments":{"question":"Another question?"}}]',
            '[{"name":"request_clarification","arguments":{"reason":"Wrong field"}}]',
        ]
    )

    class FakeAgent:
        def __init__(self, settings_: LocalLoraConfig):
            settings.append(settings_)

        def invoke_messages(self, messages: list[dict[str, str]]) -> LocalLoraInvocation:
            inputs.append(messages)
            raw = next(raw_outputs)
            return LocalLoraInvocation(raw_text=raw, response=parse_lora_agent_response(raw))

    monkeypatch.setattr("scripts.probe_sft_clarification.LocalLoraAgent", FakeAgent)
    report = _probe(config, output)
    rows = [json.loads(line) for line in (dataset / "validation.jsonl").read_text().splitlines()]
    assert inputs == [rows[1]["messages"][:-1], rows[2]["messages"][:-1]]
    assert all([message["role"] for message in prompt] == ["system", "user"] for prompt in inputs)
    assert settings[0].adapter_path == load_training_config(config).runtime.expected_adapter_path
    assert settings[0].max_new_tokens == 256 and settings[0].temperature == 0
    assert report["summary"]["decision_counts"] == {
        "act": 0,
        "clarify": 1,
        "refuse": 0,
        "invalid/unknown": 1,
    }
    assert report["summary"]["correct_clarification_decisions"] == 1
    assert report["summary"]["raw_outputs_mentioning_clarification"] == 2
    assert report["rows"][1]["agent_response"]["parse_error"] is not None
    assert report["rows"][1]["raw_model_output"].endswith('"Wrong field"}}]')


@pytest.mark.parametrize(
    "defect",
    [
        "output_exists",
        "source_overlap",
        "prompt_hash",
        "adapter_metadata",
        "config_hash",
        "output_in_dataset",
        "insufficient_rows",
        "wrong_meta_field",
    ],
)
def test_preflight_fails_before_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    config, dataset, output = _setup(tmp_path)
    rows = [json.loads(line) for line in (dataset / "validation.jsonl").read_text().splitlines()]
    if defect == "output_exists":
        output.write_text("preserve prior report")
    elif defect == "source_overlap":
        training = [rows[1], _example("training two", "act")]
        (dataset / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in training))
    elif defect == "prompt_hash":
        rows[1]["messages"][0]["content"] = "Unexpected prompt"
    elif defect == "wrong_meta_field":
        rows[1]["messages"][-1]["content"] = (
            '[{"name":"request_clarification","arguments":{"reason":"Wrong"}}]'
        )
    elif defect == "insufficient_rows":
        rows[2] = rows[1]
    elif defect == "adapter_metadata":
        path = load_training_config(config).runtime.expected_adapter_path / "adapter_config.json"
        payload = json.loads(path.read_text())
        payload["r"] = 16
        path.write_text(json.dumps(payload))
    elif defect == "config_hash":
        payload = json.loads(config.read_text())
        payload["config_hash"] = "stale"
        config.write_text(json.dumps(payload))
    else:
        output = dataset / "must-not-write.json"
    (dataset / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        "scripts.probe_sft_clarification.LocalLoraAgent", lambda _: pytest.fail("No model")
    )
    with pytest.raises(ValueError):
        _probe(config, output)
    if defect == "output_exists":
        assert output.read_text() == "preserve prior report"
    else:
        assert not output.exists()


def test_training_probe_is_explicitly_insample_and_never_sends_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset, output = _setup(tmp_path)
    rows = [_example("seen one", "clarify"), _example("seen two", "clarify")]
    (dataset / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = {path.name: path.read_bytes() for path in dataset.iterdir()}
    inputs = []

    class FakeAgent:
        def __init__(self, settings: LocalLoraConfig):
            assert settings.max_new_tokens == 256 and settings.temperature == 0

        def invoke_messages(self, messages: list[dict[str, str]]) -> LocalLoraInvocation:
            inputs.append(messages)
            raw = '[{"name":"request_clarification","arguments":{"question":"Which item?"}}]'
            return LocalLoraInvocation(raw_text=raw, response=parse_lora_agent_response(raw))

    monkeypatch.setattr("scripts.probe_sft_clarification.LocalLoraAgent", FakeAgent)
    report = _probe(config, output, training_rows=[2, 1])
    assert report["diagnostic"] == "insample-sft-clarification-probe-v1"
    assert report["data_split"] == "train"
    assert report["requested_training_rows"] == [1, 2]
    assert report["training_cap"] == 2
    assert "validation_sha256" not in report
    assert [row["training_row"] for row in report["rows"]] == [1, 2]
    assert all("validation_row" not in row for row in report["rows"])
    assert inputs == [row["messages"][:-1] for row in rows]
    assert all([message["role"] for message in prompt] == ["system", "user"] for prompt in inputs)
    assert report["summary"]["correct_clarification_decisions"] == 2
    assert {path.name: path.read_bytes() for path in dataset.iterdir()} == before


@pytest.mark.parametrize(
    "defect",
    [
        "count",
        "duplicate_line",
        "zero",
        "outside_cap",
        "not_clarify",
        "duplicate_input",
        "reference",
        "presentation",
    ],
)
def test_training_probe_rejects_invalid_selection_or_mode_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    config, dataset, output = _setup(tmp_path)
    rows = [_example("seen one", "clarify"), _example("seen two", "clarify")]
    indices = {"count": [1], "duplicate_line": [1, 1], "zero": [0, 1], "outside_cap": [1, 3]}.get(
        defect, [1, 2]
    )
    if defect == "not_clarify":
        rows[1] = _example("seen action", "act")
    elif defect == "duplicate_input":
        rows[1] = rows[0]
    (dataset / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        "scripts.probe_sft_clarification.LocalLoraAgent", lambda _: pytest.fail("No model")
    )
    with pytest.raises(ValueError):
        _probe(
            config,
            output,
            training_rows=indices,
            reference_report=tmp_path / "unused.json" if defect == "reference" else None,
            benchmark_json=defect == "presentation",
        )
    assert not output.exists()


def test_training_probe_cli_dry_run_is_labelled_insample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, dataset, output = _setup(tmp_path)
    rows = [_example("seen one", "clarify"), _example("seen two", "clarify")]
    (dataset / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        "scripts.probe_sft_clarification.LocalLoraAgent", lambda _: pytest.fail("No model")
    )
    assert (
        main(
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--limit",
                "2",
                "--train-cap",
                "2",
                "--training-rows",
                "1",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "in-sample training" in capsys.readouterr().out
    report = json.loads(output.read_text())
    assert report["data_split"] == "train"
    assert not report["generated"] and report["summary"] is None


def test_cli_dry_run_does_not_claim_measured_decisions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _, output = _setup(tmp_path)
    assert (
        main(
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--limit",
                "2",
                "--validation-cap",
                "5",
                "--train-cap",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "Selected 2 held-out" in text and "no model/tokenizer loaded" in text
    assert "correct_clarification_decisions" not in text


def test_benchmark_json_preserves_values_time_tools_order_and_targets() -> None:
    row = _example("Caf\u00e9", "clarify")
    header, separator, text = row["messages"][1]["content"].partition("\n\nUser request:\n")
    context = json.loads(text)
    context["current_time"] = "2026-06-01T09:00:00Z"
    context["initial_state"] = {"clipboard": {"text": "Caf\u00e9", "items": [3, 1]}}
    row["messages"][1]["content"] = (
        header
        + separator
        + json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    example = SFTExample.model_validate(row)
    before = example.model_dump_json()
    presented = benchmark_json_example(example)
    assert example.model_dump_json() == before
    assert presented.messages[0] == example.messages[0]
    assert presented.messages[-1] == example.messages[-1]
    old_header, _, old_context = example.messages[1].content.partition(separator)
    new_header, _, new_context = presented.messages[1].content.partition(separator)
    title, _, old_tools = old_header.partition("\n")
    new_title, _, new_tools = new_header.partition("\n")
    assert title == new_title
    assert json.loads(old_tools) == json.loads(new_tools)
    assert json.loads(old_context) == json.loads(new_context)
    assert json.loads(new_context)["current_time"] == "2026-06-01T09:00:00Z"
    assert json.loads(new_context)["initial_state"]["clipboard"]["items"] == [3, 1]
    assert "\\u00e9" in new_context and "\u00e9" in old_context
    assert new_tools == json.dumps(json.loads(old_tools), sort_keys=True)
    assert new_context == json.dumps(json.loads(old_context), sort_keys=True)


def _completed_reference(
    config: Path, output: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], list[list[dict[str, str]]]]:
    inputs = []

    class FakeAgent:
        def __init__(self, _: LocalLoraConfig):
            pass

        def invoke_messages(self, messages: list[dict[str, str]]) -> LocalLoraInvocation:
            inputs.append(messages)
            raw = '[{"name":"request_clarification","arguments":{"question":"Which item?"}}]'
            return LocalLoraInvocation(raw_text=raw, response=parse_lora_agent_response(raw))

    monkeypatch.setattr("scripts.probe_sft_clarification.LocalLoraAgent", FakeAgent)
    return _probe(config, output), inputs


def test_presentation_probe_matches_reference_and_changes_only_model_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dataset, reference_path = _setup(tmp_path)
    reference, inputs = _completed_reference(config, reference_path, monkeypatch)
    before = reference_path.read_bytes()
    report = _probe(
        config, tmp_path / "paired.json", benchmark_json=True, reference_report=reference_path
    )
    assert reference_path.read_bytes() == before
    assert report["reference_summary"] == reference["summary"]
    assert report["input_presentation"] == "benchmark-json"
    assert len(inputs) == 4
    for i in range(2):
        original = inputs[i]
        changed = inputs[i + 2]
        assert original != changed
        assert original[0] == changed[0]
        for original_text, changed_text in zip(
            original[1]["content"].split("\n\nUser request:\n"),
            changed[1]["content"].split("\n\nUser request:\n"),
            strict=True,
        ):
            if original_text.startswith("Available tools"):
                original_text = original_text.split("\n", 1)[1]
                changed_text = changed_text.split("\n", 1)[1]
            assert json.loads(original_text) == json.loads(changed_text)
        assert report["rows"][i]["input_sha256"] == reference["rows"][i]["input_sha256"]
        assert report["rows"][i]["model_input_sha256"] != report["rows"][i]["input_sha256"]
        assert report["rows"][i]["expected_target"] == reference["rows"][i]["expected_target"]
        assert report["rows"][i]["validation_row"] == reference["rows"][i]["validation_row"]
    assert report["summary"]["correct_clarification_decisions"] == 2
    assert (dataset / "validation.jsonl").exists()


@pytest.mark.parametrize("defect", ["weights", "selection", "dry_reference", "missing_reference"])
def test_presentation_probe_rejects_uncontrolled_reference_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    config, _, reference_path = _setup(tmp_path)
    reference, _ = _completed_reference(config, reference_path, monkeypatch)
    if defect == "weights":
        path = (
            load_training_config(config).runtime.expected_adapter_path / "adapter_model.safetensors"
        )
        path.write_bytes(b"Different unit checkpoint")
    elif defect == "selection":
        reference["rows"][0]["validation_row"] = 99
    elif defect == "dry_reference":
        reference["generated"] = False
        reference["summary"] = None
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.probe_sft_clarification.LocalLoraAgent", lambda _: pytest.fail("No model")
    )
    output = tmp_path / "must-not-write.json"
    with pytest.raises(ValueError):
        _probe(
            config,
            output,
            benchmark_json=True,
            reference_report=None if defect == "missing_reference" else reference_path,
        )
    assert not output.exists()


def test_message_inference_reuses_greedy_generation_and_strict_parser() -> None:
    recorded = {}
    messages = [
        {"role": "system", "content": "Original prompt"},
        {"role": "user", "content": "Original input"},
    ]

    class InputIds:
        shape = (1, 3)

        def to(self, device: str) -> InputIds:
            assert device == "unit-device"
            return self

    class Tokenizer:
        eos_token_id = 7

        def apply_chat_template(self, values: Any, **kwargs: Any) -> str:
            recorded["messages"] = values
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "Unit rendered prompt"

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, InputIds]:
            assert prompt == "Unit rendered prompt" and kwargs == {"return_tensors": "pt"}
            return {"input_ids": InputIds()}

        def decode(self, values: list[int], **kwargs: Any) -> str:
            assert values == [42] and kwargs == {"skip_special_tokens": True}
            return '[{"name":"request_clarification","arguments":{"question":"Which item?"}}]'

    def generate(**kwargs: Any) -> list[list[int]]:
        recorded["generation"] = kwargs
        return [[1, 2, 3, 42]]

    agent = object.__new__(LocalLoraAgent)
    agent._config = LocalLoraConfig()
    agent._tokenizer = Tokenizer()
    agent._torch = SimpleNamespace(no_grad=nullcontext)
    agent._model = SimpleNamespace(
        generate=generate, parameters=lambda: iter([SimpleNamespace(device="unit-device")])
    )
    invocation = agent.invoke_messages(messages)
    assert recorded["messages"] == messages
    assert recorded["generation"]["max_new_tokens"] == 256
    assert recorded["generation"]["do_sample"] is False
    assert "temperature" not in recorded["generation"]
    assert invocation.response.decision_type == "clarify"
    with pytest.raises(ValueError, match="no assistant target"):
        agent.invoke_messages([*messages, {"role": "assistant", "content": "Must not leak"}])


@pytest.mark.parametrize("compact_json", [False, True])
def test_scenario_invocation_delegates_to_the_same_message_path(
    monkeypatch: pytest.MonkeyPatch,
    compact_json: bool,
) -> None:
    messages = [{"role": "system", "content": "same"}, {"role": "user", "content": "same"}]
    agent = object.__new__(LocalLoraAgent)
    agent._config = LocalLoraConfig(compact_json=compact_json)
    result = LocalLoraInvocation(raw_text="[]", response=parse_lora_agent_response("[]"))
    seen = []

    def build_messages(_: Scenario, **kwargs: Any) -> list[dict[str, str]]:
        assert kwargs == {"compact_json": compact_json}
        return messages

    monkeypatch.setattr("baymax.eval.lora_runner.build_lora_scenario_messages", build_messages)
    monkeypatch.setattr(agent, "invoke_messages", lambda values: seen.append(values) or result)
    assert agent.invoke(cast(Scenario, object())) is result
    assert seen == [messages]
