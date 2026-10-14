from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from baymax.models.baymax_sft import BaymaxSFTBuildConfig, build_baymax_sft_dataset
from baymax.models.sft_dataset import (
    SFTBuildConfig,
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SplitName,
    build_sft_dataset_from_rows,
    convert_xlam_row_to_sft_example,
)
from baymax.models.sft_filter import filter_sft_dataset
from baymax.models.sft_mix import (
    SFTMixConfig,
    _contrast_reference_train_order,
    _mixed_split_rows,
    _read_contrast_groups,
    _read_full_reference_rows,
    _read_reference_sampling,
    _rebalance_reference_train_order,
    _reference_cap_order,
    mix_sft_datasets,
)
from baymax.models.sft_validation import validate_sft_dataset
from scripts.audit_sft_numeric_arguments import audit_dataset


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _setup(tmp_path: Path, *, real_control: bool = False) -> SFTMixConfig:
    xlam, baymax, reference, filtered = [
        tmp_path / name for name in ("xlam", "baymax", "reference", "filtered")
    ]
    tools = [{"name": "lookup", "parameters": {"n": {"type": "int"}}}]
    build_sft_dataset_from_rows(
        [
            {
                "id": i,
                "query": f"External lookup {i}",
                "tools": tools,
                "answers": [{"name": "lookup", "arguments": {"n": str(i) if i % 2 else i}}],
            }
            for i in range(2000 if real_control else 100)
        ],
        config=SFTBuildConfig(
            dataset_id="example/xlam", revision="pinned", output_dir=xlam, seed=1
        ),
    )
    build_sft_dataset_from_rows(
        [
            {
                "id": i,
                "query": f"BAYMAX request {i}",
                "tools": tools,
                "answers": (
                    [{"name": "request_clarification", "arguments": {"question": "Which?"}}]
                    if i % 3 == 0
                    else [{"name": "refuse_request", "arguments": {"reason": "Unavailable."}}]
                    if i % 3 == 1
                    else [{"name": "lookup", "arguments": {"n": i}}]
                ),
            }
            for i in range(90)
        ],
        config=SFTBuildConfig(
            dataset_id="example/baymax", revision="pinned", output_dir=baymax, seed=1
        ),
    )
    if real_control:
        build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax, seed=1))
    config = SFTMixConfig(
        xlam_dir=xlam,
        baymax_dir=baymax,
        output_dir=reference,
        seed=1,
        train_cap_row_count=1000 if real_control else 30,
        train_cap_baymax_row_count=400 if real_control else 6,
        train_cap_baymax_clarification_count=100 if real_control else 2,
        train_cap_baymax_refusal_count=100 if real_control else 2,
        validation_cap_row_count=200 if real_control else 12,
        validation_cap_baymax_row_count=90 if real_control else 6,
        validation_cap_baymax_clarification_count=30 if real_control else 2,
        validation_cap_baymax_refusal_count=30 if real_control else 2,
    )
    mix_sft_datasets(config)
    audit = audit_dataset(xlam, checks="targets")
    assert audit["row_parse_errors"] == []
    audit_path, review_path = tmp_path / "audit.json", tmp_path / "review.json"
    _write(audit_path, audit)
    _write(
        review_path,
        {
            "format_version": "target-filter-review-v1",
            "policy": "sft-target-compatible-v1",
            "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "manifest_sha256": audit["manifest_sha256"],
            "approved_rows": audit["proposed_exclusions"],
            "schema_reviews": [],
            "retained_coverage_gaps": [],
        },
    )
    filter_sft_dataset(
        dataset_dir=xlam, audit_path=audit_path, review_path=review_path, output_dir=filtered
    )
    return config.model_copy(
        update={
            "xlam_dir": filtered,
            "reference_mix_dir": reference,
            "output_dir": tmp_path / "anchored",
        }
    )


def _prefix(path: Path, count: int) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:count]]


def test_reference_sampling_preserves_every_compatible_position_and_baymax_row(
    tmp_path: Path,
) -> None:
    config = _setup(tmp_path)
    assert config.reference_mix_dir is not None
    before = {p.name: p.read_bytes() for p in config.reference_mix_dir.iterdir()}
    manifest = mix_sft_datasets(config)
    report = json.loads((config.output_dir / "sampling-report.json").read_text())
    review = json.loads((config.xlam_dir / "filter-review.json").read_text())
    exclusions = {(r["split"], r["example_sha256"]) for r in review["approved_rows"]}
    caps: dict[SplitName, int] = {"train": 30, "validation": 12}
    for split, cap in caps.items():
        old = _prefix(config.reference_mix_dir / f"{split}.jsonl", cap)
        new = _prefix(config.output_dir / f"{split}.jsonl", cap)
        replaced = 0
        for position, (old_row, new_row) in enumerate(zip(old, new, strict=True), 1):
            if (split, _hash(old_row)) in exclusions:
                assert old_row != new_row
                assert any(
                    r["position"] == position for r in report["splits"][split]["replacements"]
                )
                replaced += 1
            else:
                assert old_row == new_row
        assert replaced > 0
        assert report["splits"][split]["kept_baymax"] == 6
        assert report["splits"][split]["replaced_xlam"] == replaced
        assert len({_hash(row) for row in new}) == cap
        assert len(manifest.splits[split].source_rows) == manifest.splits[split].count
    assert {p.name: p.read_bytes() for p in config.reference_mix_dir.iterdir()} == before
    assert validate_sft_dataset(config.output_dir).valid
    train = {_hash(row) for row in _prefix(config.output_dir / "train.jsonl", 1000)}
    validation = {_hash(row) for row in _prefix(config.output_dir / "validation.jsonl", 1000)}
    assert train.isdisjoint(validation)


def test_reference_prefix_is_independent_of_combined_shuffle_and_output_path(
    tmp_path: Path,
) -> None:
    config = _setup(tmp_path)
    first = mix_sft_datasets(config)
    repeat = config.model_copy(update={"output_dir": tmp_path / "repeat"})
    second = mix_sft_datasets(repeat)
    unshuffled = config.model_copy(update={"output_dir": tmp_path / "unshuffled", "shuffle": False})
    mix_sft_datasets(unshuffled)
    assert first.revision == second.revision
    for split, cap in (("train", 30), ("validation", 12)):
        assert _prefix(config.output_dir / f"{split}.jsonl", cap) == _prefix(
            unshuffled.output_dir / f"{split}.jsonl", cap
        )
    assert (config.output_dir / "sampling-report.json").read_bytes() == (
        repeat.output_dir / "sampling-report.json"
    ).read_bytes()


def test_reference_sampling_cannot_use_protected_rows_as_replacement_fillers(
    tmp_path: Path,
) -> None:
    config = _setup(tmp_path)
    prefixes, exclusions, _ = _read_reference_sampling(config)
    rows, _ = _mixed_split_rows(
        split_name="train", config=config, baymax_repeat_count=1, output_source_index=0
    )
    protected = {_hash(row.example.model_dump(mode="json")) for row in prefixes["train"]}
    only_protected = [
        row for row in rows if _hash(row.example.model_dump(mode="json")) in protected
    ]
    with pytest.raises(ValueError, match="same-split replacement"):
        _reference_cap_order(only_protected, prefixes["train"], exclusions, config, "train")


def test_pool_size_changes_do_not_reselect_compatible_reference_examples(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    prefixes, exclusions, _ = _read_reference_sampling(config)
    rows, _ = _mixed_split_rows(
        split_name="train", config=config, baymax_repeat_count=1, output_source_index=0
    )
    template = next(row for row in rows if str(row.source.source_row_id).startswith("xlam:"))
    additions = []
    for index in range(25):
        example = convert_xlam_row_to_sft_example(
            {
                "query": f"Additional independent item {index}",
                "tools": [{"name": "lookup", "parameters": {"n": {"type": "int"}}}],
                "answers": [{"name": "lookup", "arguments": {"n": index}}],
            },
            system_prompt=template.example.messages[0].content,
        )
        additions.append(template.model_copy(update={"example": example}))
    expanded, report = _reference_cap_order(
        additions + list(reversed(rows)), prefixes["train"], exclusions, config, "train"
    )
    assert report["kept_baymax"] == 6
    for position, row in enumerate(prefixes["train"]):
        if ("train", _hash(row.example.model_dump(mode="json"))) not in exclusions:
            assert expanded[position].example == row.example


def test_reference_cli_reports_preserved_and_replaced_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.mix_sft_datasets import main

    config = _setup(tmp_path)
    arguments = {
        "xlam-dir": config.xlam_dir,
        "baymax-dir": config.baymax_dir,
        "reference-mix-dir": config.reference_mix_dir,
        "output-dir": config.output_dir,
        "train-cap-row-count": 30,
        "train-cap-baymax-row-count": 6,
        "train-cap-baymax-clarification-count": 2,
        "train-cap-baymax-refusal-count": 2,
        "validation-cap-row-count": 12,
        "validation-cap-baymax-row-count": 6,
        "validation-cap-baymax-clarification-count": 2,
        "validation-cap-baymax-refusal-count": 2,
    }
    argv = ["mix_sft_datasets.py"]
    for name, value in arguments.items():
        argv.extend(["--" + name, str(value)])
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    output = capsys.readouterr().out
    assert "Reference train: kept BAYMAX=6" in output
    assert "Reference validation: kept BAYMAX=6" in output
    assert "replaced xLAM=" in output and "sampling-report.json" in output


@pytest.mark.parametrize(
    "fault",
    [
        "baymax_changed",
        "reference_changed",
        "reference_duplicate",
        "wrong_quota",
        "filtered_changed",
        "review_changed",
        "existing_output",
        "wrong_seed",
        "missing_provenance",
    ],
)
def test_reference_sampling_rejects_unexplained_changes_before_publishing(
    tmp_path: Path, fault: str
) -> None:
    config = _setup(tmp_path)
    assert config.reference_mix_dir is not None
    if fault == "baymax_changed":
        path = config.baymax_dir / "train.jsonl"
        path.write_bytes(path.read_bytes().replace(b"BAYMAX request", b"Edited BAYMAX request"))
    elif fault in {"reference_changed", "reference_duplicate"}:
        path = config.reference_mix_dir / "train.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        if fault == "reference_changed":
            lines[0] = lines[0].replace(b"BAYMAX request", b"Edited BAYMAX request")
        else:
            lines[1] = lines[0]
        path.write_bytes(b"".join(lines))
    elif fault == "wrong_quota":
        config = config.model_copy(update={"train_cap_baymax_refusal_count": 1})
    elif fault == "filtered_changed":
        path = config.xlam_dir / "train.jsonl"
        path.write_bytes(path.read_bytes().replace(b"External lookup", b"Edited lookup"))
    elif fault == "review_changed":
        path = config.xlam_dir / "filter-review.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "wrong_seed":
        config = config.model_copy(update={"seed": 2})
    elif fault == "missing_provenance":
        _write(config.xlam_dir / "filter-provenance.json", {})
    else:
        config.output_dir.mkdir()
        (config.output_dir / "sentinel.txt").write_text("preserve")
    with pytest.raises(ValueError):
        mix_sft_datasets(config)
    assert not (config.output_dir / "split-manifest.json").exists()
    assert not (config.output_dir / "train.jsonl").exists()
    if fault == "existing_output":
        assert (config.output_dir / "sentinel.txt").read_text() == "preserve"


@pytest.mark.parametrize(
    "change",
    [
        {"train_cap_row_count": None},
        {"validation_cap_row_count": None},
        {"preserve_contrast_groups": True},
        {"train_baymax_repeat_count": 2},
        {"train_baymax_clarification_extra_repeat_count": 1},
    ],
)
def test_reference_sampling_rejects_incompatible_modes(change: dict[str, Any]) -> None:
    kwargs = {
        "reference_mix_dir": Path("reference"),
        "train_cap_row_count": 1000,
        "validation_cap_row_count": 200,
        **change,
    }
    with pytest.raises(ValueError, match="Reference sampling"):
        SFTMixConfig(**kwargs)


def _matched_setup(tmp_path: Path) -> SFTMixConfig:
    config = _setup(tmp_path)
    mix_sft_datasets(config)
    source_dir = tmp_path / "matched-source"
    source_dir.mkdir()
    old_manifest = SFTSplitManifest.model_validate_json(
        (config.baymax_dir / "split-manifest.json").read_text()
    )
    examples = []
    sources = []
    groups = {}
    for group in range(4):
        members = []
        for role in ("act", "clarify", "refuse"):
            source_id = f"matched_{group}_{role}"
            members.append(source_id)
            name = {
                "act": "lookup",
                "clarify": "request_clarification",
                "refuse": "refuse_request",
            }[role]
            arguments = {
                "act": {"n": group},
                "clarify": {"question": "Which item?"},
                "refuse": {"reason": "Lookup is unavailable."},
            }[role]
            examples.append(
                convert_xlam_row_to_sft_example(
                    {
                        "query": f"Independent matched item {group}, {role}",
                        "tools": [{"name": "lookup", "parameters": {"n": {"type": "int"}}}],
                        "answers": [{"name": name, "arguments": arguments}],
                    },
                    system_prompt=_prefix(config.baymax_dir / "train.jsonl", 1)[0]["messages"][0][
                        "content"
                    ],
                )
            )
            sources.append(
                SFTSourceRowRef(
                    source_index=len(sources), source_row_id=source_id, row_hash=_hash(source_id)
                )
            )
        groups[f"matched_{group}"] = members
    manifest = old_manifest.model_copy(
        update={
            "output_dir": str(source_dir),
            "revision": _hash(groups),
            "loaded_rows": len(examples),
            "converted_rows": len(examples),
            "splits": {
                split: SFTSplitManifestEntry(
                    path=str(source_dir / f"{split}.jsonl"),
                    count=len(examples) if split == "train" else 0,
                    source_rows=sources if split == "train" else [],
                )
                for split in ("train", "validation", "test")
            },
        }
    )
    (source_dir / "train.jsonl").write_text(
        "".join(example.model_dump_json() + "\n" for example in examples), encoding="utf-8"
    )
    for split in ("validation", "test"):
        (source_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
    (source_dir / "split-manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    _write(source_dir / "contrast-groups.json", {"revision": manifest.revision, "groups": groups})
    return config.model_copy(
        update={
            "reference_mix_dir": config.output_dir,
            "baymax_dir": source_dir,
            "output_dir": tmp_path / "matched-mix",
            "preserve_contrast_groups": True,
            "replace_reference_baymax_train": True,
        }
    )


def test_matched_reference_changes_only_baymax_train_and_is_reproducible(tmp_path: Path) -> None:
    config = _matched_setup(tmp_path)
    assert config.reference_mix_dir is not None
    before = {p.name: p.read_bytes() for p in config.reference_mix_dir.iterdir()}
    original_manifest = SFTSplitManifest.model_validate_json(
        (config.reference_mix_dir / "split-manifest.json").read_text()
    )
    first = mix_sft_datasets(config)
    second_config = config.model_copy(update={"output_dir": tmp_path / "matched-repeat"})
    second = mix_sft_datasets(second_config)
    assert first.revision == second.revision
    for split in ("train", "validation", "test"):
        assert (config.output_dir / f"{split}.jsonl").read_bytes() == (
            second_config.output_dir / f"{split}.jsonl"
        ).read_bytes()
    for split in ("validation", "test"):
        assert (config.output_dir / f"{split}.jsonl").read_bytes() == before[f"{split}.jsonl"]
        assert first.splits[split].source_rows == original_manifest.splits[split].source_rows
    old = _prefix(config.reference_mix_dir / "train.jsonl", 1000)
    new = _prefix(config.output_dir / "train.jsonl", 1000)
    assert len(old) == len(new)
    for source, old_row, new_row in zip(
        original_manifest.splits["train"].source_rows, old, new, strict=True
    ):
        if str(source.source_row_id).startswith("xlam:"):
            assert old_row == new_row
    report = json.loads((config.output_dir / "sampling-report.json").read_text())
    train = report["splits"]["train"]
    assert train["kept_xlam"] == 24 and train["replaced_xlam"] == 0
    assert train["baymax_class_counts"] == {
        "productive": 2,
        "request_clarification": 2,
        "refuse_request": 2,
    }
    assert len(train["complete_contrast_groups"]) == 2
    assert len({_hash(row) for row in new}) == len(new)
    assert validate_sft_dataset(config.output_dir).valid
    assert {p.name: p.read_bytes() for p in config.reference_mix_dir.iterdir()} == before


def test_matched_reference_excludes_whole_group_on_heldout_input_overlap(tmp_path: Path) -> None:
    config = _matched_setup(tmp_path)
    assert config.reference_mix_dir is not None
    path = config.baymax_dir / "train.jsonl"
    rows = _prefix(path, 1000)
    heldout = _prefix(config.reference_mix_dir / "validation.jsonl", 1)[0]
    rows[0]["messages"][:-1] = heldout["messages"][:-1]
    assert rows[0] != heldout
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mix_sft_datasets(config)
    report = json.loads((config.output_dir / "sampling-report.json").read_text())["splits"]["train"]
    assert report["excluded_heldout_groups"] == ["matched_0"]
    assert "matched_0" not in report["complete_contrast_groups"]
    source_manifest = SFTSplitManifest.model_validate_json(
        (config.baymax_dir / "split-manifest.json").read_text()
    )
    result = SFTSplitManifest.model_validate_json(
        (config.output_dir / "split-manifest.json").read_text()
    )
    for source in result.splits["train"].source_rows:
        if str(source.source_row_id).startswith("baymax:"):
            index = int(str(source.source_row_id).split(":")[2])
            assert source_manifest.splits["train"].source_rows[index].source_row_id not in {
                "matched_0_act",
                "matched_0_clarify",
                "matched_0_refuse",
            }


def test_matched_reference_rejects_missing_xlam_instead_of_reselecting(tmp_path: Path) -> None:
    config = _matched_setup(tmp_path)
    reference_rows = _read_full_reference_rows(config)
    rows, _ = _mixed_split_rows(
        split_name="train", config=config, baymax_repeat_count=1, output_source_index=0
    )
    missing = next(
        row for row in reference_rows["train"] if str(row.source.source_row_id).startswith("xlam:")
    )
    rows = [row for row in rows if row.example != missing.example]
    groups = _read_contrast_groups(
        config.baymax_dir,
        SFTSplitManifest.model_validate_json(
            (config.baymax_dir / "split-manifest.json").read_text()
        ),
    )
    with pytest.raises(ValueError, match="preserve every reference xLAM"):
        _contrast_reference_train_order(rows, reference_rows, {}, config, groups)


@pytest.mark.parametrize(
    "fields",
    [
        {"replace_reference_baymax_train": True},
        {
            "replace_reference_baymax_train": True,
            "reference_mix_dir": Path("reference"),
            "train_cap_row_count": 1000,
            "validation_cap_row_count": 200,
        },
    ],
)
def test_matched_reference_requires_explicit_group_and_reference_modes(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="requires a reference mix and contrast groups"):
        SFTMixConfig(**fields)


def test_matched_reference_real_generator_keeps_act200_quotas(tmp_path: Path) -> None:
    config = _setup(tmp_path, real_control=True)
    mix_sft_datasets(config)
    source = tmp_path / "matched-real-source"
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=source, matched_contrasts=True))
    matched = config.model_copy(
        update={
            "reference_mix_dir": config.output_dir,
            "baymax_dir": source,
            "output_dir": tmp_path / "matched-real-mix",
            "preserve_contrast_groups": True,
            "replace_reference_baymax_train": True,
        }
    )
    mix_sft_datasets(matched)
    assert validate_sft_dataset(matched.output_dir).valid
    report = json.loads((matched.output_dir / "sampling-report.json").read_text())["splits"][
        "train"
    ]
    assert report["baymax_class_counts"] == {
        "productive": 200,
        "request_clarification": 100,
        "refuse_request": 100,
    }
    assert report["kept_xlam"] == 600 and report["replaced_xlam"] == 0
    assert len(report["complete_contrast_groups"]) >= 50


def test_matched_reference_cannot_fill_quotas_from_heldout_groups(tmp_path: Path) -> None:
    config = _matched_setup(tmp_path)
    assert config.reference_mix_dir is not None
    path = config.baymax_dir / "train.jsonl"
    rows = _prefix(path, 1000)
    heldout = _prefix(config.reference_mix_dir / "validation.jsonl", 1)[0]
    for index in (0, 3, 6):
        rows[index]["messages"][:-1] = heldout["messages"][:-1]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="selectable unique"):
        mix_sft_datasets(config)
    assert not (config.output_dir / "train.jsonl").exists()
    assert not (config.output_dir / "split-manifest.json").exists()


def test_matched_reference_detects_heldout_file_drift_before_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _matched_setup(tmp_path)
    assert config.reference_mix_dir is not None
    path = config.reference_mix_dir / "validation.jsonl"
    original_loader = _mixed_split_rows

    def changed_loader(**kwargs: Any) -> Any:
        rows = original_loader(**kwargs)
        path.write_bytes(path.read_bytes() + b"\n")
        return rows

    monkeypatch.setattr("baymax.models.sft_mix._mixed_split_rows", changed_loader)
    with pytest.raises(ValueError, match="inputs changed during the build"):
        mix_sft_datasets(config)
    assert not (config.output_dir / "train.jsonl").exists()
    assert not (config.output_dir / "validation.jsonl").exists()


def _rebalance_setup(tmp_path: Path) -> SFTMixConfig:
    reference = _setup(tmp_path)
    mix_sft_datasets(reference)
    return reference.model_copy(
        update={
            "reference_mix_dir": reference.output_dir,
            "output_dir": tmp_path / "rebalanced",
            "rebalance_reference_train": True,
            "train_cap_baymax_row_count": 12,
            "train_cap_baymax_clarification_count": 5,
            "train_cap_baymax_refusal_count": 5,
        }
    )


def test_rebalance_preserves_controls_and_heldout_files_without_duplicates(tmp_path: Path) -> None:
    config = _rebalance_setup(tmp_path)
    assert config.reference_mix_dir is not None
    original = _read_full_reference_rows(config)
    before = {path.name: path.read_bytes() for path in config.reference_mix_dir.iterdir()}
    manifest = mix_sft_datasets(config)
    prefix = _prefix(config.output_dir / "train.jsonl", 30)
    report = json.loads((config.output_dir / "sampling-report.json").read_text())["splits"]["train"]
    assert report["baymax_class_counts"] == {
        "productive": 2,
        "request_clarification": 5,
        "refuse_request": 5,
    }
    assert report["added_boundary_rows"] == 6
    assert report["kept_baymax"] == 6 and report["kept_xlam"] == 18
    assert len({_hash(row) for row in prefix}) == 30
    for index, row in enumerate(original["train"][:30]):
        if str(row.source.source_row_id).startswith("baymax:"):
            assert prefix[index] == row.example.model_dump(mode="json")
    old_hashes = {_hash(row.example.model_dump(mode="json")) for row in original["train"]}
    all_new = _prefix(config.output_dir / "train.jsonl", 1000)
    assert old_hashes.issubset({_hash(row) for row in all_new})
    assert len({_hash(row) for row in all_new}) == len(all_new)
    assert manifest.splits["train"].count == len(original["train"]) + 6
    for split in ("validation", "test"):
        assert (config.output_dir / f"{split}.jsonl").read_bytes() == before[f"{split}.jsonl"]
    assert {path.name: path.read_bytes() for path in config.reference_mix_dir.iterdir()} == before
    assert validate_sft_dataset(config.output_dir).valid


def test_rebalance_is_deterministic_across_output_paths_and_shuffle(tmp_path: Path) -> None:
    config = _rebalance_setup(tmp_path)
    first = mix_sft_datasets(config)
    repeat = config.model_copy(update={"output_dir": tmp_path / "repeat-ratio"})
    second = mix_sft_datasets(repeat)
    assert first.revision == second.revision
    assert (config.output_dir / "train.jsonl").read_bytes() == (
        repeat.output_dir / "train.jsonl"
    ).read_bytes()
    unshuffled = config.model_copy(
        update={"output_dir": tmp_path / "no-shuffle-ratio", "shuffle": False}
    )
    mix_sft_datasets(unshuffled)
    assert _prefix(config.output_dir / "train.jsonl", 30) == _prefix(
        unshuffled.output_dir / "train.jsonl", 30
    )


def test_rebalance_rejects_reference_tail_rows_missing_from_source(tmp_path: Path) -> None:
    config = _rebalance_setup(tmp_path)
    reference = _read_full_reference_rows(config)
    rows, _ = _mixed_split_rows(
        split_name="train",
        config=config,
        baymax_repeat_count=1,
        output_source_index=0,
    )
    missing = reference["train"][30].example
    rows = [row for row in rows if row.example != missing]
    with pytest.raises(ValueError, match="missing from the source"):
        _rebalance_reference_train_order(rows, reference, {}, config)


def test_rebalance_rejects_heldout_input_even_with_a_different_target(tmp_path: Path) -> None:
    config = _rebalance_setup(tmp_path)
    reference = _read_full_reference_rows(config)
    rows, _ = _mixed_split_rows(
        split_name="train",
        config=config,
        baymax_repeat_count=1,
        output_source_index=0,
    )
    heldout = reference["validation"][0].example
    candidates = []
    prefix_hashes = {_hash(row.example.model_dump(mode="json")) for row in reference["train"][:30]}
    for row in rows:
        if (
            str(row.source.source_row_id).startswith("baymax:")
            and _hash(row.example.model_dump(mode="json")) not in prefix_hashes
        ):
            payload = row.example.model_dump(mode="json")
            payload["messages"][:-1] = heldout.model_dump(mode="json")["messages"][:-1]
            row = row.model_copy(update={"example": type(row.example).model_validate(payload)})
        candidates.append(row)
    with pytest.raises(ValueError, match="unique same-split"):
        _rebalance_reference_train_order(candidates, reference, {}, config)


@pytest.mark.parametrize(
    "changes",
    [
        {"train_cap_baymax_row_count": 13},
        {"train_cap_baymax_row_count": 8, "train_cap_baymax_clarification_count": 1},
        {"train_cap_baymax_row_count": 40, "train_cap_baymax_clarification_count": 33},
    ],
)
def test_rebalance_rejects_changed_productive_counts_or_impossible_quotas(
    tmp_path: Path,
    changes: dict[str, int],
) -> None:
    config = _rebalance_setup(tmp_path).model_copy(update=changes)
    with pytest.raises(ValueError):
        mix_sft_datasets(config)
    assert not (config.output_dir / "train.jsonl").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"reference_mix_dir": Path("reference"), "preserve_contrast_groups": True},
    ],
)
def test_rebalance_requires_an_explicit_reference_without_group_sampling(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="rebalancing requires a reference"):
        SFTMixConfig(rebalance_reference_train=True, **changes)


def test_rebalance_accepts_existing_empty_output_directory(tmp_path: Path) -> None:
    config = _rebalance_setup(tmp_path)
    config.output_dir.mkdir()
    manifest = mix_sft_datasets(config)
    assert (config.output_dir / "split-manifest.json").exists()
    assert manifest.splits["train"].count > 30


def test_rebalance_does_not_overwrite_nonempty_output_directory(tmp_path: Path) -> None:
    config = _rebalance_setup(tmp_path)
    config.output_dir.mkdir()
    sentinel = config.output_dir / "partial-build.jsonl"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty output directory"):
        mix_sft_datasets(config)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (config.output_dir / "split-manifest.json").exists()
