from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from baymax.models.baymax_sft import SyntheticBaymaxExample, synthetic_baymax_example_to_sft_example
from baymax.models.sft_dataset import (
    DEFAULT_SFT_SYSTEM_PROMPT,
    SFTExample,
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SFTSplitRatios,
    convert_xlam_row_to_sft_example,
)
from baymax.models.sft_expansion import expand_training_dataset
from baymax.models.sft_expansion_examples import generate_expansion_examples
from baymax.models.sft_mix import _sft_input_hash
from baymax.models.sft_serialization import name_first_example
from baymax.models.sft_validation import validate_sft_dataset, validate_sft_target_contract
from baymax.models.training_config import (
    load_training_config,
    training_config_with_hash,
    validate_training_config,
    write_training_config,
)
from scripts.prepare_sft_expansion import prepare_experiment


@pytest.fixture(scope="module")
def examples() -> list[SyntheticBaymaxExample]:
    return generate_expansion_examples(seed=1)


def _sft(example: SyntheticBaymaxExample) -> SFTExample:
    return name_first_example(
        synthetic_baymax_example_to_sft_example(example, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT),
        location=example.id,
    )


def test_new_pool_has_exact_unique_class_counts_and_matched_groups(examples) -> None:
    assert len(examples) == 400
    assert Counter(row.behavior_type for row in examples) == {
        "direct_tool_call": 175,
        "multi_tool_sequence": 25,
        "request_clarification": 100,
        "refuse_request": 100,
    }
    groups = Counter(row.contrast_group for row in examples)
    assert len(groups) == 100 and set(groups.values()) == {4}
    inputs = set()
    for row in examples:
        example = _sft(row)
        assert not validate_sft_target_contract(example, location=row.id)
        assert example.messages[-1].content.startswith('[{"name":')
        digest = _sft_input_hash(example)
        assert digest not in inputs
        inputs.add(digest)
    assert generate_expansion_examples(seed=1) == examples
    assert generate_expansion_examples(seed=2) != examples


def test_examples_bind_state_and_ask_for_all_missing_calendar_fields(examples) -> None:
    for row in examples:
        if row.behavior_type == "multi_tool_sequence":
            assert row.assistant_calls[0] == {"name": "clipboard.read", "arguments": {}}
            arguments = row.assistant_calls[1]["arguments"]
            assert (
                arguments.get("title", arguments.get("body"))
                == row.initial_state["clipboard"]["text"]
            )
        if row.id.endswith(":clarify") and row.available_tools[0] == "calendar.create_event":
            question = row.assistant_calls[0]["arguments"]["question"]
            if not re.search(r"\bfor \d+ minutes\b", row.user_input):
                assert "duration" in question or "minutes" in question
            if " at " not in row.user_input:
                assert "time" in question
        if "without assigning a deadline" in row.user_input:
            assert row.behavior_type == "direct_tool_call"
            assert "due_date" not in row.assistant_calls[0]["arguments"]
        if row.behavior_type == "request_clarification" and "due soon" in row.user_input:
            assert "due date" in row.assistant_calls[0]["arguments"]["question"]
        if "indiscriminately to every contact" in row.user_input:
            assert row.behavior_type == "refuse_request"
            assert len(row.initial_state["gmail_contacts"]) >= 3


def _external(index: int) -> SFTExample:
    return name_first_example(
        convert_xlam_row_to_sft_example(
            {
                "query": f"External lookup {index}",
                "tools": [{"name": "lookup", "parameters": {"n": {"type": "int"}}}],
                "answers": [{"name": "lookup", "arguments": {"n": index}}],
            },
            system_prompt=DEFAULT_SFT_SYSTEM_PROMPT,
        ),
        location="fixture external",
    )


def _parent(tmp_path: Path, examples) -> Path:
    directory = tmp_path / "parent"
    directory.mkdir()
    train = [_sft(row) for row in examples[4:8]] + [_external(i) for i in range(20)]
    entries = {}
    for split, rows in (
        ("train", train),
        ("validation", [_external(100)]),
        ("test", [_external(101)]),
    ):
        path = directory / f"{split}.jsonl"
        path.write_bytes(b"".join((row.model_dump_json() + "\r\n").encode() for row in rows))
        entries[split] = SFTSplitManifestEntry(
            path=str(path),
            count=len(rows),
            source_rows=[
                SFTSourceRowRef(
                    source_index=i,
                    source_row_id=(
                        f"{'baymax' if split == 'train' and i < 4 else 'xlam'}:{split}:{i}:r0"
                    ),
                    row_hash=hashlib.sha256(f"{split}:{i}".encode()).hexdigest(),
                )
                for i in range(len(rows))
            ],
        )
    manifest = SFTSplitManifest(
        format_version="qwen-chat-messages-v2",
        config_hash="fixture",
        dataset_id="mixed:fixture",
        revision="pinned",
        source_split="synthetic",
        seed=1,
        max_rows=None,
        split_ratios=SFTSplitRatios(),
        system_prompt_sha256=hashlib.sha256(DEFAULT_SFT_SYSTEM_PROMPT.encode()).hexdigest(),
        output_dir=str(directory),
        splits=entries,
        loaded_rows=26,
        converted_rows=26,
        skipped_rows=[],
    )
    (directory / "split-manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return directory


def _patch_additions(monkeypatch, examples) -> None:
    monkeypatch.setattr(
        "baymax.models.sft_expansion.generate_expansion_examples", lambda *, seed: examples[12:16]
    )


def test_expansion_preserves_prefix_parent_sources_and_heldout_bytes(
    tmp_path, monkeypatch, examples
) -> None:
    parent = _parent(tmp_path, examples)
    before = {path.name: path.read_bytes() for path in parent.iterdir()}
    _patch_additions(monkeypatch, examples)
    output = tmp_path / "expanded"
    manifest = expand_training_dataset(dataset_dir=parent, output_dir=output, train_cap=10)
    assert before == {path.name: path.read_bytes() for path in parent.iterdir()}
    for split in ("validation", "test"):
        assert (output / f"{split}.jsonl").read_bytes() == before[f"{split}.jsonl"]
        original = SFTSplitManifest.model_validate_json(before["split-manifest.json"])
        assert manifest.splits[split].source_rows == original.splits[split].source_rows
    old_lines = before["train.jsonl"].splitlines(keepends=True)
    new_lines = (output / "train.jsonl").read_bytes().splitlines(keepends=True)
    assert new_lines[:10] == old_lines[:10]
    assert Counter(new_lines) >= Counter(old_lines)
    assert len(new_lines) == len(old_lines) + 4
    report = json.loads((output / "expansion-report.json").read_text())
    assert report["after_cap"]["source"] == {"baymax": 8, "xlam": 12}
    assert report["after_cap"]["baymax_classes"] == {
        "direct_tool_call": 4,
        "request_clarification": 2,
        "refuse_request": 2,
    }
    assert report["after_cap"]["unique_inputs"] == 20
    assert report["after_cap"]["exact_duplicate_rows"] == 0
    assert len(report["added_xlam"]) == 6
    assert validate_sft_dataset(output).valid
    repeat = tmp_path / "repeat"
    second = expand_training_dataset(dataset_dir=parent, output_dir=repeat, train_cap=10)
    assert manifest.revision == second.revision
    assert new_lines == (repeat / "train.jsonl").read_bytes().splitlines(keepends=True)
    with pytest.raises(ValueError, match="new nonoverlapping"):
        expand_training_dataset(dataset_dir=parent, output_dir=output, train_cap=10)


@pytest.mark.parametrize("external_index,blocked", [(0, True), (6, False)])
def test_same_input_different_answer_is_blocked_in_prefix_or_excluded_from_tail(
    tmp_path, monkeypatch, examples, external_index, blocked
) -> None:
    parent = _parent(tmp_path, examples)
    _patch_additions(monkeypatch, examples)
    heldout = _external(external_index)
    calls = json.loads(heldout.messages[-1].content)
    calls[0]["arguments"]["n"] = 900
    heldout.messages[-1].content = json.dumps(calls, separators=(",", ":"))
    (parent / "validation.jsonl").write_text(heldout.model_dump_json() + "\n", encoding="utf-8")
    # Historical validation compares complete examples, so it misses this input-only overlap.
    assert validate_sft_dataset(parent).valid
    before = {path.name: path.read_bytes() for path in parent.iterdir()}
    output = tmp_path / "expanded"
    if blocked:
        with pytest.raises(
            ValueError, match="Frozen training prefix line 5 overlaps validation line 1"
        ):
            expand_training_dataset(dataset_dir=parent, output_dir=output, train_cap=10)
        assert not output.exists()
        return
    manifest = expand_training_dataset(dataset_dir=parent, output_dir=output, train_cap=10)
    assert before == {path.name: path.read_bytes() for path in parent.iterdir()}
    assert (output / "validation.jsonl").read_bytes() == before["validation.jsonl"]
    old_lines = before["train.jsonl"].splitlines(keepends=True)
    new_lines = (output / "train.jsonl").read_bytes().splitlines(keepends=True)
    assert new_lines[:10] == old_lines[:10]
    assert len(new_lines) == len(old_lines) + 4 - 1
    assert not any(
        _sft_input_hash(SFTExample.model_validate_json(line)) == _sft_input_hash(heldout)
        for line in new_lines
    )
    assert len(manifest.skipped_rows) == 1
    assert manifest.loaded_rows - len(manifest.skipped_rows) == manifest.converted_rows
    report = json.loads((output / "expansion-report.json").read_text())
    assert report["after_cap"]["source"] == {"baymax": 8, "xlam": 12}
    assert report["excluded_parent_tail"][0]["train_line"] == 11
    assert report["excluded_parent_tail"][0]["heldout_line"] == 1
    assert validate_sft_dataset(output).valid


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate-new",
        "heldout-overlap",
        "training-overlap",
        "wrong-classes",
        "too-few-external",
        "bad-count",
        "bad-prompt",
    ],
)
def test_invalid_expansions_never_publish(tmp_path, monkeypatch, examples, problem) -> None:
    parent = _parent(tmp_path, examples)
    additions = list(examples[12:16])
    if problem == "duplicate-new":
        additions[1] = additions[0]
    elif problem == "wrong-classes":
        additions = list(examples[:4])  # Contains a multi-tool ACT, unlike the parent quota.
    elif problem in ("heldout-overlap", "training-overlap"):
        additions[0] = examples[4] if problem == "training-overlap" else additions[0]
        if problem == "heldout-overlap":
            (parent / "validation.jsonl").write_text(
                _sft(additions[0]).model_dump_json() + "\n", encoding="utf-8"
            )
    elif problem == "too-few-external":
        path = parent / "train.jsonl"
        path.write_bytes(b"".join(path.read_bytes().splitlines(keepends=True)[:13]))
        manifest_path = parent / "split-manifest.json"
        manifest = SFTSplitManifest.model_validate_json(manifest_path.read_bytes())
        manifest.splits["train"].count = 13
        manifest.splits["train"].source_rows = manifest.splits["train"].source_rows[:13]
        manifest.loaded_rows = manifest.converted_rows = 15
        manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    elif problem == "bad-count":
        manifest = SFTSplitManifest.model_validate_json(
            (parent / "split-manifest.json").read_bytes()
        )
        manifest.loaded_rows += 1
        (parent / "split-manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    elif problem == "bad-prompt":
        path = parent / "train.jsonl"
        path.write_bytes(
            path.read_bytes().replace(b"BAYMAX's tool-calling planner", b"different planner")
        )
    monkeypatch.setattr(
        "baymax.models.sft_expansion.generate_expansion_examples", lambda *, seed: additions
    )
    with pytest.raises(ValueError):
        expand_training_dataset(dataset_dir=parent, output_dir=tmp_path / "output", train_cap=10)
    assert not (tmp_path / "output").exists()


def test_config_preserves_training_recipe_and_uses_new_identity(
    tmp_path, monkeypatch, examples
) -> None:
    parent = _parent(tmp_path, examples)
    _patch_additions(monkeypatch, examples)
    base = load_training_config(
        Path(
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo_name_first.json"
        )
    )
    payload = base.model_dump(mode="json")
    payload["data"].update(
        {
            "dataset_id": "mixed:fixture",
            "revision": "pinned",
            "split_manifest_path": str(parent / "split-manifest.json"),
            "validation_report_path": str(parent / "validation-report.json"),
            **{
                f"{split}_path": str(parent / f"{split}.jsonl")
                for split in ("train", "validation", "test")
            },
        }
    )
    (parent / "validation-report.json").write_text(
        validate_sft_dataset(parent).model_dump_json(), encoding="utf-8"
    )
    reference = training_config_with_hash(type(base).model_validate(payload))
    reference_path = tmp_path / "reference.json"
    write_training_config(reference_path, reference)
    config_path = tmp_path / "coverage.json"
    config = prepare_experiment(
        reference_config=reference_path,
        output_dir=tmp_path / "output",
        config_path=config_path,
        train_cap=10,
    )
    assert config.training == reference.training
    assert config.lora == reference.lora
    assert config.base_model == reference.base_model
    assert config.data.system_prompt_sha256 == reference.data.system_prompt_sha256
    assert config.runtime.seed == reference.runtime.seed
    assert config.config_hash != reference.config_hash
    assert config.data.revision != reference.data.revision
    (tmp_path / "output" / "validation-report.json").write_text(
        validate_sft_dataset(tmp_path / "output").model_dump_json(), encoding="utf-8"
    )
    assert validate_training_config(config_path, check_paths=True).valid
