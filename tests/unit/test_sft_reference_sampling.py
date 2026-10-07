from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from baymax.models.sft_dataset import (
    SFTBuildConfig,
    SplitName,
    build_sft_dataset_from_rows,
    convert_xlam_row_to_sft_example,
)
from baymax.models.sft_filter import filter_sft_dataset
from baymax.models.sft_mix import (
    SFTMixConfig,
    _mixed_split_rows,
    _read_reference_sampling,
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


def _setup(tmp_path: Path) -> SFTMixConfig:
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
            for i in range(100)
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
    config = SFTMixConfig(
        xlam_dir=xlam,
        baymax_dir=baymax,
        output_dir=reference,
        seed=1,
        train_cap_row_count=30,
        train_cap_baymax_row_count=6,
        train_cap_baymax_clarification_count=2,
        train_cap_baymax_refusal_count=2,
        validation_cap_row_count=12,
        validation_cap_baymax_row_count=6,
        validation_cap_baymax_clarification_count=2,
        validation_cap_baymax_refusal_count=2,
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
