from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from baymax.models.sft_dataset import (
    SFTExample,
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SFTSplitRatios,
    convert_xlam_row_to_sft_example,
)
from baymax.models.sft_serialization import name_first_example, serialize_sft_name_first
from baymax.models.sft_validation import validate_sft_dataset, validate_sft_target_contract
from baymax.models.training_config import (
    load_training_config,
    training_config_hash,
    training_config_with_hash,
    validate_training_config,
    write_training_config,
)
from scripts.prepare_sft_name_first import prepare_experiment

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / (
    "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2_qkvo.json"
)


def _example(split: str, kind: str) -> SFTExample:
    if kind == "clarify":
        calls = [
            {"name": "request_clarification", "arguments": {"question": f"Which {split} item?"}}
        ]
    elif kind == "refuse":
        calls = [
            {"name": "refuse_request", "arguments": {"reason": f"Cannot do {split} operation."}}
        ]
    else:
        calls = [
            {
                "name": "test.capture",
                "arguments": {
                    "payload": {
                        "z": [1, {"name": "nested", "arguments": {"n": 2.5}}],
                        "question": "keep this key",
                        "reason": "keep this too",
                    }
                },
            },
            {"name": "test.capture", "arguments": {"payload": {"name": "second"}}},
        ]
    return convert_xlam_row_to_sft_example(
        {
            "query": f"{split} {kind} independent training request",
            "tools": [
                {
                    "name": "test.capture",
                    "parameters": {
                        "type": "object",
                        "properties": {"payload": {"type": "object"}},
                        "required": ["payload"],
                        "additionalProperties": False,
                    },
                }
            ],
            "answers": calls,
        },
        system_prompt="Test prompt",
    )


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, output = tmp_path / "source", tmp_path / "name-first"
    source.mkdir()
    reference = load_training_config(REFERENCE)
    entries = {}
    index = 0
    for split in ("train", "validation", "test"):
        examples = [_example(split, kind) for kind in ("act", "clarify", "refuse")]
        refs = []
        for example in examples:
            refs.append(
                SFTSourceRowRef(
                    source_index=index,
                    source_row_id=f"source-{index}",
                    row_hash=f"identity-{index}",
                )
            )
            index += 1
        path = source / f"{split}.jsonl"
        path.write_bytes(b"".join((row.model_dump_json() + "\r\n").encode() for row in examples))
        entries[split] = SFTSplitManifestEntry(path=str(path), count=3, source_rows=refs)
    manifest = SFTSplitManifest(
        format_version=reference.data.format_version,
        config_hash="parent-build-hash",
        dataset_id=reference.data.dataset_id,
        revision="parent-revision",
        source_split="mixed",
        seed=reference.runtime.seed,
        max_rows=None,
        split_ratios=SFTSplitRatios(),
        system_prompt_sha256=hashlib.sha256(b"Test prompt").hexdigest(),
        output_dir=str(source),
        splits=entries,
        loaded_rows=9,
        converted_rows=9,
        skipped_rows=[],
    )
    manifest_path = source / "split-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    data = reference.data.model_copy(
        update={
            "revision": manifest.revision,
            "system_prompt_sha256": manifest.system_prompt_sha256,
            "split_manifest_path": manifest_path,
            "validation_report_path": source / "validation-report.json",
            **{f"{split}_path": Path(entry.path) for split, entry in entries.items()},
        }
    )
    reference_path = tmp_path / "reference.json"
    write_training_config(
        reference_path, training_config_with_hash(reference.model_copy(update={"data": data}))
    )
    return source, output, reference_path


@pytest.mark.parametrize("kind", ["act", "clarify", "refuse"])
def test_reorders_only_outer_call_keys(kind: str) -> None:
    example = _example("train", kind)
    before = example.model_dump_json()
    result = name_first_example(example, location="training fixture")
    assert example.model_dump_json() == before
    assert result.messages[:-1] == example.messages[:-1]
    original = json.loads(example.messages[-1].content)
    transformed = json.loads(result.messages[-1].content)
    assert transformed == original
    assert all(list(call) == ["name", "arguments"] for call in transformed)
    for old, new in zip(original, transformed, strict=True):
        assert json.dumps(new["arguments"]) == json.dumps(old["arguments"])
    assert not validate_sft_target_contract(result, location="transformed fixture")


def test_preserves_rows_caps_splits_refs_and_parent_bytes(tmp_path: Path) -> None:
    source, output, _ = _setup(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    parent = SFTSplitManifest.model_validate_json(before["split-manifest.json"])
    derived = serialize_sft_name_first(dataset_dir=source, output_dir=output)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    for split, entry in derived.splits.items():
        assert entry.source_rows == parent.splits[split].source_rows
        assert entry.count == parent.splits[split].count
        old_rows = [json.loads(line) for line in before[f"{split}.jsonl"].splitlines()]
        new_rows = [
            json.loads(line) for line in (output / f"{split}.jsonl").read_bytes().splitlines()
        ]
        for old, new in zip(old_rows, new_rows, strict=True):
            assert old["messages"][:-1] == new["messages"][:-1]
            assert json.loads(old["messages"][-1]["content"]) == json.loads(
                new["messages"][-1]["content"]
            )
        # A prefix cap must still select the very same examples, not resample them.
        assert new_rows[:2] == [
            name_first_example(
                SFTExample.model_validate(row), location="capped fixture"
            ).model_dump(mode="json")
            for row in old_rows[:2]
        ]
    ignored = {"revision", "config_hash", "output_dir", "splits"}
    assert derived.model_dump(exclude=ignored) == parent.model_dump(exclude=ignored)
    assert derived.revision != parent.revision
    assert validate_sft_dataset(output).valid
    provenance = json.loads((output / "serialization-provenance.json").read_text())
    assert provenance["parent_revision"] == parent.revision
    assert (
        provenance["parent_manifest_sha256"]
        == hashlib.sha256(before["split-manifest.json"]).hexdigest()
    )
    for split, counts in provenance["files"].items():
        assert counts["rows"] == 3 and counts["calls"] == 4
        assert counts["parent_sha256"] == hashlib.sha256(before[f"{split}.jsonl"]).hexdigest()
        assert (
            counts["output_sha256"]
            == hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest()
        )
    second = tmp_path / "second-copy"
    repeated = serialize_sft_name_first(dataset_dir=source, output_dir=second)
    assert repeated.revision == derived.revision
    for split in derived.splits:
        assert (second / f"{split}.jsonl").read_bytes() == (output / f"{split}.jsonl").read_bytes()


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_meta_field",
        "malformed_json",
        "duplicate_key",
        "noncanonical",
        "extra_row",
        "missing_row",
        "escaped_path",
        "prompt_hash",
        "provenance_count",
    ],
)
def test_rejects_bad_sources_without_publishing(tmp_path: Path, defect: str) -> None:
    source, output, _ = _setup(tmp_path)
    manifest_path = source / "split-manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    path = source / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "wrong_meta_field":
        rows[1]["messages"][-1]["content"] = (
            '[{"arguments":{"reason":"Wrong field"},"name":"request_clarification"}]'
        )
    elif defect == "malformed_json":
        rows[1]["messages"][-1]["content"] = '[{"arguments":'
    elif defect == "duplicate_key":
        rows[1]["messages"][-1]["content"] = (
            '[{"arguments":{"question":"First","question":"Second"},"name":"request_clarification"}]'
        )
    elif defect == "noncanonical":
        rows[1]["messages"][-1]["content"] = json.dumps(
            json.loads(rows[1]["messages"][-1]["content"])
        )
    elif defect == "extra_row":
        rows.append(rows[-1])
    elif defect == "missing_row":
        rows.pop()
    elif defect == "escaped_path":
        manifest["splits"]["train"]["path"] = str(tmp_path / "outside.jsonl")
    elif defect == "provenance_count":
        manifest["splits"]["train"]["source_rows"].pop()
    else:
        manifest["system_prompt_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError):
        serialize_sft_name_first(dataset_dir=source, output_dir=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".name-first-*"))
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_preparation_keeps_E_recipe_and_generates_valid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source, output, reference_path = _setup(tmp_path)
    reference = load_training_config(reference_path)
    config_path = tmp_path / "configs" / "qkvo_name_first.json"
    result = prepare_experiment(
        reference_config=reference_path, output_dir=output, config_path=config_path
    )
    assert result.base_model == reference.base_model
    assert result.lora == reference.lora
    assert result.training == reference.training
    assert result.output_adapter == reference.output_adapter
    assert result.runtime.model_dump(
        exclude={"run_name", "output_dir", "expected_adapter_path"}
    ) == reference.runtime.model_dump(exclude={"run_name", "output_dir", "expected_adapter_path"})
    assert result.data.model_dump(
        exclude={
            "revision",
            "train_path",
            "validation_path",
            "test_path",
            "split_manifest_path",
            "validation_report_path",
        }
    ) == reference.data.model_dump(
        exclude={
            "revision",
            "train_path",
            "validation_path",
            "test_path",
            "split_manifest_path",
            "validation_report_path",
        }
    )
    assert result.config_hash == training_config_hash(result)
    assert result.config_hash != reference.config_hash
    assert load_training_config(config_path) == result
    assert validate_training_config(config_path).valid
    assert result.runtime.output_dir != reference.runtime.output_dir
    assert result.data.train_path.parent == output
    assert source != output


@pytest.mark.parametrize("blocked", ["config", "model", "hash", "dataset", "overlap"])
def test_preparation_rejects_overwrites_and_stale_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: str
) -> None:
    monkeypatch.chdir(tmp_path)
    source, output, reference_path = _setup(tmp_path)
    config_path = tmp_path / "candidate.json"
    if blocked == "config":
        config_path.write_text("preserve me")
    elif blocked == "model":
        (tmp_path / "models" / "candidate").mkdir(parents=True)
    elif blocked == "hash":
        reference = load_training_config(reference_path).model_copy(update={"config_hash": "stale"})
        write_training_config(reference_path, reference)
    elif blocked == "dataset":
        output.mkdir()
        (output / "keep.txt").write_text("preserve me")
    else:
        output = source / "nested"
    with pytest.raises(ValueError):
        prepare_experiment(
            reference_config=reference_path, output_dir=output, config_path=config_path
        )
    if blocked == "config":
        assert config_path.read_text() == "preserve me"
    else:
        assert not config_path.exists()
    if blocked == "dataset":
        assert (output / "keep.txt").read_text() == "preserve me"


def test_direct_cli_runs_on_tiny_fixture_without_model_loading(tmp_path: Path) -> None:
    _, output, reference_path = _setup(tmp_path)
    config_path = tmp_path / "cli-config.json"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_sft_name_first.py"),
            "--reference-config",
            str(reference_path),
            "--output-dir",
            str(output),
            "--config-path",
            str(config_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    assert "Only assistant call-key order changed" in process.stdout
    assert validate_training_config(config_path).valid
