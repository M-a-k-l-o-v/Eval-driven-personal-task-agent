from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from baymax.models.baymax_sft import BaymaxSFTBuildConfig, build_baymax_sft_dataset
from baymax.models.sft_dataset import SFTBuildConfig, build_sft_dataset_from_rows
from baymax.models.sft_mix import SFTMixConfig, mix_sft_datasets
from baymax.models.sft_validation import validate_sft_dataset


def test_group_preserving_cap_keeps_act200_quotas_and_complete_groups(tmp_path: Path) -> None:
    xlam_dir, baymax_dir = tmp_path / "xlam", tmp_path / "baymax"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(1600)],
        config=SFTBuildConfig(dataset_id="example/xlam", revision="pinned", output_dir=xlam_dir),
    )
    source_manifest = build_baymax_sft_dataset(
        BaymaxSFTBuildConfig(output_dir=baymax_dir, matched_contrasts=True)
    )
    groups = json.loads((baymax_dir / "contrast-groups.json").read_text(encoding="utf-8"))["groups"]
    source_behaviors = {
        source.source_row_id: next(iter(_behavior_counts([row])))
        for source, row in zip(
            source_manifest.splits["train"].source_rows,
            _read_jsonl(baymax_dir / "train.jsonl"),
            strict=True,
        )
    }
    outputs = []
    manifests = []
    for name, cap in (("first", 1000), ("same", 1000), ("larger", 1500)):
        config = SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=tmp_path / name,
            train_cap_row_count=cap,
            preserve_contrast_groups=True,
            validation_cap_row_count=200,
        )
        manifest = mix_sft_datasets(config)
        rows = _read_jsonl(config.output_dir / "train.jsonl")[:cap]
        sources = [row.model_dump() for row in manifest.splits["train"].source_rows[:cap]]
        assert _source_counts(sources) == {"baymax": 400, "xlam": cap - 400}
        behavior = _behavior_counts(rows)
        assert behavior["request_clarification"] == behavior["refuse_request"] == 100
        assert len({json.dumps(row, sort_keys=True) for row in rows}) == cap
        selected = {
            source_manifest.splits["train"]
            .source_rows[int(row["source_row_id"].split(":")[2])]
            .source_row_id
            for row in sources
            if row["source_row_id"].startswith("baymax:")
        }
        complete_triplets = 0
        for group, members in groups.items():
            selected_boundary = [
                source.source_row_id
                for source in source_manifest.splits["train"].source_rows
                if source.source_row_id in selected & set(members)
                and source_behaviors[source.source_row_id]
                in {"request_clarification", "refuse_request"}
            ]
            assert not selected_boundary or set(members) <= selected
            complete_triplets += int(
                group.startswith("act_clarify_refuse_triplet_") and set(members) <= selected
            )
        assert complete_triplets >= 50
        outputs.append(config.output_dir)
        manifests.append(manifest)
    assert (outputs[0] / "train.jsonl").read_bytes() == (outputs[1] / "train.jsonl").read_bytes()
    assert (
        _read_jsonl(outputs[0] / "train.jsonl")[:1000]
        == _read_jsonl(outputs[2] / "train.jsonl")[:1000]
    )
    for split in ("validation", "test"):
        assert (outputs[0] / f"{split}.jsonl").read_bytes() == (
            outputs[2] / f"{split}.jsonl"
        ).read_bytes()
    assert not (
        set(row.source_row_id for row in manifests[0].splits["train"].source_rows)
        & set(row.source_row_id for row in manifests[0].splits["validation"].source_rows)
    )


@pytest.mark.parametrize("fault", ["revision", "unknown", "repeated", "cross_split"])
def test_group_sampler_rejects_unbound_metadata(tmp_path: Path, fault: str) -> None:
    from baymax.models.sft_mix import _read_contrast_groups

    manifest = build_baymax_sft_dataset(
        BaymaxSFTBuildConfig(output_dir=tmp_path, matched_contrasts=True)
    )
    path = tmp_path / "contrast-groups.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if fault == "revision":
        metadata["revision"] = "wrong"
        message = "source revision"
    elif fault == "unknown":
        metadata["groups"] = {"bad": ["unknown", "also_unknown"]}
        message = "Unknown or repeated"
    elif fault == "repeated":
        member = manifest.splits["train"].source_rows[0].source_row_id
        metadata["groups"] = {"bad": [member, member]}
        message = "Unknown or repeated"
    else:
        metadata["groups"] = {
            "bad": [
                manifest.splits["train"].source_rows[0].source_row_id,
                manifest.splits["validation"].source_rows[0].source_row_id,
            ]
        }
        message = "crosses source split"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _read_contrast_groups(tmp_path, manifest)


def test_group_sampler_requires_a_cap() -> None:
    with pytest.raises(ValueError, match="requires a training cap"):
        SFTMixConfig(preserve_contrast_groups=True)


def test_mix_sft_datasets_combines_xlam_and_repeated_baymax_train_rows(
    tmp_path: Path,
) -> None:
    xlam_dir = tmp_path / "xlam"
    baymax_dir = tmp_path / "baymax-control"
    mixed_dir = tmp_path / "mixed"
    xlam_manifest = build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(10)],
        config=SFTBuildConfig(
            dataset_id="example/xlam",
            revision="xlam123",
            output_dir=xlam_dir,
            seed=1,
        ),
    )
    baymax_manifest = build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))

    config = SFTMixConfig(
        xlam_dir=xlam_dir,
        baymax_dir=baymax_dir,
        output_dir=mixed_dir,
        seed=1,
        train_baymax_repeat_count=2,
        eval_baymax_repeat_count=1,
    )
    mixed_manifest = mix_sft_datasets(config)

    assert mixed_manifest.dataset_id == "mixed:example/xlam+baymax/synthetic-control-v1"
    expected_baymax_train_count = _weighted_baymax_train_count(
        baymax_dir,
        base=2,
        clarification_extra=config.train_baymax_clarification_extra_repeat_count,
        refusal_extra=config.train_baymax_refusal_extra_repeat_count,
    )
    assert mixed_manifest.splits["train"].count == (
        xlam_manifest.splits["train"].count + expected_baymax_train_count
    )
    assert mixed_manifest.splits["validation"].count == (
        xlam_manifest.splits["validation"].count + baymax_manifest.splits["validation"].count
    )
    assert mixed_manifest.splits["test"].count == (
        xlam_manifest.splits["test"].count + baymax_manifest.splits["test"].count
    )

    report = validate_sft_dataset(mixed_dir)
    assert report.valid is True


def test_mix_sft_datasets_writes_baymax_meta_tool_examples(tmp_path: Path) -> None:
    xlam_dir = tmp_path / "xlam"
    baymax_dir = tmp_path / "baymax-control"
    mixed_dir = tmp_path / "mixed"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(10)],
        config=SFTBuildConfig(
            dataset_id="example/xlam",
            revision="xlam123",
            output_dir=xlam_dir,
            seed=1,
        ),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))

    mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=mixed_dir,
            seed=1,
            train_baymax_repeat_count=1,
            eval_baymax_repeat_count=1,
            shuffle=False,
        )
    )

    assistant_payloads = _assistant_payloads(_read_jsonl(mixed_dir / "train.jsonl"))

    assert any(payload[0]["name"] == "request_clarification" for payload in assistant_payloads)
    assert any(payload[0]["name"] == "refuse_request" for payload in assistant_payloads)


def test_mix_sft_datasets_only_adds_meta_tool_repeats_to_train_split(tmp_path: Path) -> None:
    xlam_dir = tmp_path / "xlam"
    baymax_dir = tmp_path / "baymax-control"
    mixed_dir = tmp_path / "mixed"
    xlam_manifest = build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(10)],
        config=SFTBuildConfig(
            dataset_id="example/xlam",
            revision="xlam123",
            output_dir=xlam_dir,
            seed=1,
        ),
    )
    baymax_manifest = build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))

    mixed_manifest = mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=mixed_dir,
            seed=1,
            train_baymax_repeat_count=1,
            train_baymax_clarification_extra_repeat_count=2,
            train_baymax_refusal_extra_repeat_count=4,
            eval_baymax_repeat_count=1,
        )
    )

    assert mixed_manifest.splits["train"].count == (
        xlam_manifest.splits["train"].count
        + _weighted_baymax_train_count(
            baymax_dir,
            base=1,
            clarification_extra=2,
            refusal_extra=4,
        )
    )
    assert mixed_manifest.splits["validation"].count == (
        xlam_manifest.splits["validation"].count + baymax_manifest.splits["validation"].count
    )
    assert mixed_manifest.splits["test"].count == (
        xlam_manifest.splits["test"].count + baymax_manifest.splits["test"].count
    )


def test_mix_sft_datasets_shuffles_train_split_before_training_cap(tmp_path: Path) -> None:
    xlam_dir = tmp_path / "xlam"
    baymax_dir = tmp_path / "baymax-control"
    mixed_dir = tmp_path / "mixed"
    same_seed_dir = tmp_path / "same-seed"
    different_seed_dir = tmp_path / "different-seed"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(20)],
        config=SFTBuildConfig(
            dataset_id="example/xlam",
            revision="xlam123",
            output_dir=xlam_dir,
            seed=1,
        ),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))

    mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=mixed_dir,
            seed=1,
            train_baymax_repeat_count=1,
            eval_baymax_repeat_count=1,
        )
    )
    mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=same_seed_dir,
            seed=1,
            train_baymax_repeat_count=1,
            eval_baymax_repeat_count=1,
        )
    )
    mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=different_seed_dir,
            seed=2,
            train_baymax_repeat_count=1,
            eval_baymax_repeat_count=1,
        )
    )

    first_source_ids = _first_train_source_ids(mixed_dir, limit=10)
    same_seed_source_ids = _first_train_source_ids(same_seed_dir, limit=10)
    different_seed_source_ids = _first_train_source_ids(different_seed_dir, limit=10)

    assert first_source_ids != [f"xlam:train:{index}:r0" for index in range(10)]
    assert first_source_ids == same_seed_source_ids
    assert first_source_ids != different_seed_source_ids


def test_mix_sft_datasets_can_write_cap_aware_train_prefix(tmp_path: Path) -> None:
    xlam_dir = tmp_path / "xlam"
    baymax_dir = tmp_path / "baymax-control"
    mixed_dir = tmp_path / "mixed"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(200)],
        config=SFTBuildConfig(
            dataset_id="example/xlam",
            revision="xlam123",
            output_dir=xlam_dir,
            seed=1,
        ),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))

    mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=mixed_dir,
            seed=1,
            train_baymax_repeat_count=1,
            train_baymax_clarification_extra_repeat_count=6,
            train_baymax_refusal_extra_repeat_count=12,
            eval_baymax_repeat_count=1,
            train_cap_row_count=100,
            train_cap_baymax_row_count=15,
            train_cap_baymax_clarification_count=5,
            train_cap_baymax_refusal_count=6,
        )
    )

    rows = _read_jsonl(mixed_dir / "train.jsonl")[:100]
    manifest = json.loads((mixed_dir / "split-manifest.json").read_text(encoding="utf-8"))
    source_rows = manifest["splits"]["train"]["source_rows"][:100]
    source_counts = _source_counts(source_rows)
    behavior_counts = _behavior_counts(rows)
    duplicate_counts = _duplicate_counts_by_behavior(rows)

    assert source_counts == {"baymax": 15, "xlam": 85}
    assert behavior_counts["request_clarification"] == 5
    assert behavior_counts["refuse_request"] == 6
    assert duplicate_counts.get("request_clarification", 0) == 0
    assert duplicate_counts.get("refuse_request", 0) == 0


@pytest.mark.parametrize(
    "baymax_count,clarify_count,refuse_count", [(400, 100, 100), (450, 150, 150)]
)
def test_larger_training_cap_keeps_fixed_unique_baymax_boundary_pool(
    tmp_path: Path,
    baymax_count: int,
    clarify_count: int,
    refuse_count: int,
) -> None:
    xlam_dir = tmp_path / "xlam"
    baymax_dir = tmp_path / "baymax-control"
    mixed_dir = tmp_path / "mixed"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(2200)],
        config=SFTBuildConfig(
            dataset_id="example/xlam",
            revision="xlam123",
            output_dir=xlam_dir,
            seed=1,
        ),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))

    mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=mixed_dir,
            seed=1,
            train_cap_row_count=1000,
            train_cap_baymax_row_count=baymax_count,
            train_cap_baymax_clarification_count=clarify_count,
            train_cap_baymax_refusal_count=refuse_count,
        )
    )

    all_rows = _read_jsonl(mixed_dir / "train.jsonl")
    manifest = json.loads((mixed_dir / "split-manifest.json").read_text(encoding="utf-8"))
    all_source_rows = manifest["splits"]["train"]["source_rows"]
    rows_at_1000 = all_rows[:1000]
    rows_at_1500 = all_rows[:1500]
    sources_at_1000 = all_source_rows[:1000]
    sources_at_1500 = all_source_rows[:1500]

    assert _source_counts(sources_at_1000) == {"baymax": baymax_count, "xlam": 1000 - baymax_count}
    assert _source_counts(sources_at_1500) == {"baymax": baymax_count, "xlam": 1500 - baymax_count}
    assert validate_sft_dataset(mixed_dir).valid

    for rows in (rows_at_1000, rows_at_1500):
        baymax_rows = rows[:baymax_count]
        behavior_counts = _behavior_counts(baymax_rows)
        unique_counts = _unique_counts_by_behavior(baymax_rows)
        duplicate_counts = _duplicate_counts_by_behavior(rows)

        assert behavior_counts["request_clarification"] == clarify_count
        assert behavior_counts["refuse_request"] == refuse_count
        assert (
            behavior_counts["direct_tool_call"] + behavior_counts.get("multi_tool_sequence", 0)
            == baymax_count - clarify_count - refuse_count
        )
        assert unique_counts["request_clarification"] == clarify_count
        assert unique_counts["refuse_request"] == refuse_count
        assert (
            unique_counts["direct_tool_call"] + unique_counts.get("multi_tool_sequence", 0)
            == baymax_count - clarify_count - refuse_count
        )
        assert duplicate_counts == {}


@pytest.mark.parametrize("missing", ["clarification", "refusal", "productive", "xlam"])
def test_cap_fails_instead_of_silently_substituting_for_missing_unique_rows(
    tmp_path: Path,
    missing: str,
) -> None:
    xlam_dir, baymax_dir = tmp_path / "xlam", tmp_path / "baymax"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(10)],
        config=SFTBuildConfig(dataset_id="example/xlam", revision="123", output_dir=xlam_dir),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))
    config = SFTMixConfig(
        xlam_dir=xlam_dir,
        baymax_dir=baymax_dir,
        output_dir=tmp_path / "mixed",
        train_cap_row_count=1000,
        train_cap_baymax_row_count=1000 if missing != "xlam" else 450,
        train_cap_baymax_clarification_count=1000 if missing == "clarification" else 150,
        train_cap_baymax_refusal_count=850 if missing == "refusal" else 150,
    )
    message = "unique rows available" if missing == "xlam" else f"unique BAYMAX {missing} rows"
    with pytest.raises(ValueError, match=message):
        mix_sft_datasets(config)
    assert not (tmp_path / "mixed" / "split-manifest.json").exists()


def test_validation_cap_preserves_training_test_and_split_membership(tmp_path: Path) -> None:
    xlam_dir, baymax_dir = tmp_path / "xlam", tmp_path / "baymax"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(200)],
        config=SFTBuildConfig(dataset_id="example/xlam", revision="123", output_dir=xlam_dir),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))
    original_files = {
        p: p.read_bytes() for directory in (xlam_dir, baymax_dir) for p in directory.iterdir()
    }
    baseline = SFTMixConfig(
        xlam_dir=xlam_dir,
        baymax_dir=baymax_dir,
        output_dir=tmp_path / "baseline",
        train_cap_row_count=100,
        train_cap_baymax_row_count=15,
        train_cap_baymax_clarification_count=5,
        train_cap_baymax_refusal_count=6,
    )
    before = mix_sft_datasets(baseline)
    balanced = baseline.model_copy(
        update={
            "output_dir": tmp_path / "balanced",
            "validation_cap_row_count": 20,
            "validation_cap_baymax_row_count": 10,
            "validation_cap_baymax_clarification_count": 3,
            "validation_cap_baymax_refusal_count": 4,
        }
    )
    after = mix_sft_datasets(balanced)
    for split in ("train", "test"):
        assert (baseline.output_dir / f"{split}.jsonl").read_bytes() == (
            balanced.output_dir / f"{split}.jsonl"
        ).read_bytes()
        assert before.splits[split].source_rows == after.splits[split].source_rows
    old_rows = (baseline.output_dir / "validation.jsonl").read_bytes().splitlines()
    new_rows = (balanced.output_dir / "validation.jsonl").read_bytes().splitlines()
    assert Counter(old_rows) == Counter(new_rows)
    assert Counter(ref.source_row_id for ref in before.splits["validation"].source_rows) == Counter(
        ref.source_row_id for ref in after.splits["validation"].source_rows
    )
    for split in before.splits:
        assert before.splits[split].count == after.splits[split].count
    prefix = _read_jsonl(balanced.output_dir / "validation.jsonl")[:20]
    assert _source_counts(
        [ref.model_dump() for ref in after.splits["validation"].source_rows[:20]]
    ) == {
        "baymax": 10,
        "xlam": 10,
    }
    behaviors = _behavior_counts(prefix)
    assert behaviors["request_clarification"] == 3
    assert behaviors["refuse_request"] == 4
    assert _duplicate_counts_by_behavior(prefix) == {}
    assert validate_sft_dataset(balanced.output_dir).valid
    assert {p: p.read_bytes() for p in original_files} == original_files
    assert before.revision != after.revision
    identical = balanced.model_copy(update={"output_dir": tmp_path / "balanced-again"})
    repeated = mix_sft_datasets(identical)
    assert repeated.revision == after.revision
    assert (identical.output_dir / "validation.jsonl").read_bytes() == (
        balanced.output_dir / "validation.jsonl"
    ).read_bytes()
    disabled = baseline.model_copy(
        update={
            "output_dir": tmp_path / "disabled",
            "validation_cap_baymax_clarification_count": 99,
        }
    )
    unchanged = mix_sft_datasets(disabled)
    assert unchanged.revision == before.revision
    assert (disabled.output_dir / "validation.jsonl").read_bytes() == (
        baseline.output_dir / "validation.jsonl"
    ).read_bytes()


def test_act200_mix_changes_training_quota_without_changing_validation_or_test(
    tmp_path: Path,
) -> None:
    xlam_dir, baymax_dir = tmp_path / "xlam", tmp_path / "baymax"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(200)],
        config=SFTBuildConfig(dataset_id="example/xlam", revision="123", output_dir=xlam_dir),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))
    # Scale the 1,000-row experiment by ten to keep this regression small.
    baseline = SFTMixConfig(
        xlam_dir=xlam_dir,
        baymax_dir=baymax_dir,
        output_dir=tmp_path / "baseline",
        train_cap_row_count=100,
        train_cap_baymax_row_count=45,
        train_cap_baymax_clarification_count=15,
        train_cap_baymax_refusal_count=15,
        validation_cap_row_count=20,
        validation_cap_baymax_row_count=9,
        validation_cap_baymax_clarification_count=3,
        validation_cap_baymax_refusal_count=3,
    )
    before = mix_sft_datasets(baseline)
    candidate = baseline.model_copy(
        update={
            "output_dir": tmp_path / "act200",
            "train_cap_baymax_row_count": 40,
            "train_cap_baymax_clarification_count": 10,
            "train_cap_baymax_refusal_count": 10,
        }
    )
    after = mix_sft_datasets(candidate)
    prefix = _read_jsonl(candidate.output_dir / "train.jsonl")[:100]
    assert len(prefix) == 100
    assert _source_counts(
        [ref.model_dump() for ref in after.splits["train"].source_rows[:100]]
    ) == {"baymax": 40, "xlam": 60}
    behaviors = _behavior_counts(prefix)
    assert behaviors["request_clarification"] == 10
    assert behaviors["refuse_request"] == 10
    assert _duplicate_counts_by_behavior(prefix) == {}
    assert before.revision != after.revision
    for split in ("validation", "test"):
        assert (baseline.output_dir / f"{split}.jsonl").read_bytes() == (
            candidate.output_dir / f"{split}.jsonl"
        ).read_bytes()
        assert before.splits[split].source_rows == after.splits[split].source_rows
    assert validate_sft_dataset(candidate.output_dir).valid


@pytest.mark.parametrize(
    "field,value",
    [
        ("validation_cap_row_count", 0),
        ("validation_cap_baymax_row_count", 21),
        ("validation_cap_baymax_clarification_count", 8),
        ("validation_cap_baymax_refusal_count", 9),
    ],
)
def test_validation_cap_rejects_invalid_quotas(field: str, value: int) -> None:
    settings = {
        "validation_cap_row_count": 20,
        "validation_cap_baymax_row_count": 10,
        "validation_cap_baymax_clarification_count": 3,
        "validation_cap_baymax_refusal_count": 4,
    }
    settings[field] = value
    with pytest.raises(ValueError):
        SFTMixConfig.model_validate(settings)


@pytest.mark.parametrize("missing", ["clarification", "refusal", "productive", "xlam"])
def test_validation_cap_never_borrows_missing_rows_from_other_splits(
    tmp_path: Path, missing: str
) -> None:
    xlam_dir, baymax_dir = tmp_path / "xlam", tmp_path / "baymax"
    build_sft_dataset_from_rows(
        [_xlam_row(index) for index in range(10)],
        config=SFTBuildConfig(dataset_id="example/xlam", revision="123", output_dir=xlam_dir),
    )
    build_baymax_sft_dataset(BaymaxSFTBuildConfig(output_dir=baymax_dir, seed=1))
    config = SFTMixConfig(
        xlam_dir=xlam_dir,
        baymax_dir=baymax_dir,
        output_dir=tmp_path / "mixed",
        validation_cap_row_count=1000,
        validation_cap_baymax_row_count=1000 if missing != "xlam" else 10,
        validation_cap_baymax_clarification_count=1000 if missing == "clarification" else 3,
        validation_cap_baymax_refusal_count=997
        if missing == "refusal"
        else 0
        if missing == "clarification"
        else 4,
    )
    with pytest.raises(ValueError, match="Lower the quota; do not borrow from another split"):
        mix_sft_datasets(config)
    assert not (config.output_dir / "split-manifest.json").exists()


def _xlam_row(index: int) -> dict[str, object]:
    return {
        "id": index,
        "query": f"Look up item {index}.",
        "answers": json.dumps(
            [{"name": "lookup_item", "arguments": {"item_id": index}}],
            separators=(",", ":"),
        ),
        "tools": json.dumps(
            [{"name": "lookup_item", "parameters": {"item_id": {"type": "int"}}}],
            separators=(",", ":"),
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _first_train_source_ids(dataset_dir: Path, *, limit: int) -> list[str]:
    manifest = json.loads((dataset_dir / "split-manifest.json").read_text(encoding="utf-8"))
    return [
        str(source["source_row_id"])
        for source in manifest["splits"]["train"]["source_rows"][:limit]
    ]


def _source_counts(source_rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in source_rows:
        source_row_id = source["source_row_id"]
        assert isinstance(source_row_id, str)
        source_name = source_row_id.split(":", 1)[0]
        counts[source_name] = counts.get(source_name, 0) + 1

    return counts


def _behavior_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in _assistant_payloads(rows):
        behavior = _payload_behavior(payload)
        counts[behavior] = counts.get(behavior, 0) + 1

    return counts


def _duplicate_counts_by_behavior(rows: list[dict[str, object]]) -> dict[str, int]:
    seen_by_behavior: dict[str, set[str]] = {}
    duplicate_counts: dict[str, int] = {}
    for row in rows:
        payload = _assistant_payloads([row])[0]
        behavior = _payload_behavior(payload)
        row_hash = json.dumps(row, sort_keys=True)
        seen = seen_by_behavior.setdefault(behavior, set())
        if row_hash in seen:
            duplicate_counts[behavior] = duplicate_counts.get(behavior, 0) + 1
        seen.add(row_hash)

    return duplicate_counts


def _unique_counts_by_behavior(rows: list[dict[str, object]]) -> dict[str, int]:
    hashes_by_behavior: dict[str, set[str]] = {}
    for row in rows:
        payload = _assistant_payloads([row])[0]
        behavior = _payload_behavior(payload)
        hashes_by_behavior.setdefault(behavior, set()).add(json.dumps(row, sort_keys=True))

    return {behavior: len(row_hashes) for behavior, row_hashes in hashes_by_behavior.items()}


def _payload_behavior(payload: list[dict[str, object]]) -> str:
    if len(payload) > 1:
        return "multi_tool_sequence"
    name = payload[0]["name"]
    assert isinstance(name, str)
    if name == "request_clarification":
        return "request_clarification"
    if name == "refuse_request":
        return "refuse_request"
    return "direct_tool_call"


def _assistant_payloads(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    payloads: list[list[dict[str, object]]] = []
    for row in rows:
        messages = row["messages"]
        assert isinstance(messages, list)
        assistant_message = messages[-1]
        assert isinstance(assistant_message, dict)
        content = assistant_message["content"]
        assert isinstance(content, str)
        payload = json.loads(content)
        assert isinstance(payload, list)
        payloads.append(payload)

    return payloads


def _weighted_baymax_train_count(
    dataset_dir: Path,
    *,
    base: int,
    clarification_extra: int,
    refusal_extra: int,
) -> int:
    total = 0
    for payload in _assistant_payloads(_read_jsonl(dataset_dir / "train.jsonl")):
        first_call = payload[0]
        repeat_count = base
        if first_call["name"] == "request_clarification":
            repeat_count += clarification_extra
        if first_call["name"] == "refuse_request":
            repeat_count += refusal_extra
        total += repeat_count

    return total
