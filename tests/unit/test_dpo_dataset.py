from __future__ import annotations

import json
from pathlib import Path

import pytest

from baymax.models.baymax_sft import (
    generate_synthetic_baymax_examples,
    synthetic_baymax_example_to_sft_example,
)
from baymax.models.dpo_act_rejections import (
    AuthoredActRejection,
    authored_act_rejection,
)
from baymax.models.dpo_dataset import build_dpo_dataset, build_preference_pairs, decision
from baymax.models.sft_dataset import (
    DEFAULT_SFT_SYSTEM_PROMPT,
    SFTExample,
    SFTMessage,
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SFTSplitRatios,
)
from baymax.models.sft_mix import _file_sha256, _sft_input_hash
from baymax.models.training_config import (
    load_training_config,
    training_config_with_hash,
    write_training_config,
)


def _triplet() -> tuple[dict[str, SFTExample], dict[str, str]]:
    sources = [
        row
        for row in generate_synthetic_baymax_examples()
        if row.contrast_group == "act_clarify_refuse_triplet_001"
    ]
    assert len(sources) == 3
    return {
        row.id: synthetic_baymax_example_to_sft_example(
            row, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT
        )
        for row in sources
    }, {row.id: str(row.contrast_group) for row in sources}


def test_preferences_compare_different_decisions_for_the_identical_prompt() -> None:
    pool, groups = _triplet()
    pairs, missing = build_preference_pairs(pool, pool, groups, protected_inputs=set(), seed=1)
    assert len(pairs) == 6 and not missing
    assert {f"{pair.chosen_decision}>{pair.rejected_decision}" for pair in pairs} == {
        "act>clarify",
        "act>refuse",
        "clarify>act",
        "clarify>refuse",
        "refuse>act",
        "refuse>clarify",
    }
    for pair in pairs:
        assert pair.prompt == pool[pair.source_id].messages[:-1]
        assert pair.chosen == pool[pair.source_id].messages[-1:]
        assert pair.chosen != pair.rejected
        assert decision(SFTExample(messages=pair.prompt + pair.rejected)) == pair.rejected_decision
        assert all(message.role == "assistant" for message in pair.chosen + pair.rejected)
        assert json.loads(pair.rejected[0].content)
    assert all(pair.negative_kind == "wrong_decision" for pair in pairs)


def test_unavailable_tool_is_an_explicit_capability_negative_not_malformed_json() -> None:
    pool, groups = _triplet()
    source_id = next(key for key, row in pool.items() if decision(row) == "refuse")
    row = pool[source_id]
    header, separator, context = row.messages[1].content.partition("\n\nUser request:\n")
    label, _, tools_json = header.partition("\n")
    tools = [
        tool
        for tool in json.loads(tools_json)
        if tool["name"] in {"request_clarification", "refuse_request"}
    ]
    pool[source_id] = row.model_copy(
        update={
            "messages": [
                row.messages[0],
                SFTMessage(
                    role="user",
                    content=label + "\n" + json.dumps(tools) + separator + context,
                ),
                row.messages[-1],
            ]
        }
    )
    pairs, _ = build_preference_pairs(
        {source_id: pool[source_id]}, pool, groups, protected_inputs=set(), seed=1
    )
    productive = next(pair for pair in pairs if pair.rejected_decision == "act")
    assert productive.negative_kind == "unavailable_tool"
    assert json.loads(productive.rejected[0].content)[0]["name"] == "gmail.send_email"


def test_preferences_are_deterministic_and_do_not_mutate_source() -> None:
    pool, groups = _triplet()
    before = {key: row.model_dump_json() for key, row in pool.items()}
    first, _ = build_preference_pairs(pool, pool, groups, protected_inputs=set(), seed=1)
    second, _ = build_preference_pairs(
        dict(reversed(list(pool.items()))), pool, groups, protected_inputs=set(), seed=1
    )
    assert first == second
    assert {key: row.model_dump_json() for key, row in pool.items()} == before


def test_missing_productive_counterpart_is_reported_not_invented() -> None:
    pool, _ = _triplet()
    anchors = {key: row for key, row in pool.items() if decision(row) != "act"}
    pairs, missing = build_preference_pairs(anchors, anchors, {}, protected_inputs=set(), seed=1)
    assert len(pairs) == 2
    assert missing == {"clarify_without_act_counterpart": 1, "refuse_without_act_counterpart": 1}
    assert all(pair.counterpart_source_id is None for pair in pairs)


def test_authored_negatives_complete_boundary_pair_without_changing_original_preferences() -> None:
    sources = [
        row
        for row in generate_synthetic_baymax_examples()
        if row.contrast_group == "capability_boundary_send_missing_address"
    ]
    pool = {
        row.id: synthetic_baymax_example_to_sft_example(
            row, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT
        )
        for row in sources
    }
    groups = {row.id: str(row.contrast_group) for row in sources}
    authored = {
        row.id: rejection
        for row in sources
        if (rejection := authored_act_rejection(row)) is not None
    }
    before = {key: row.model_dump_json() for key, row in pool.items()}
    old, old_missing = build_preference_pairs(pool, pool, groups, protected_inputs=set(), seed=1)
    new, missing = build_preference_pairs(
        pool, pool, groups, protected_inputs=set(), seed=1, act_rejections=authored
    )
    assert len(old) == 2 and len(old_missing) == 2
    assert len(new) == 4 and not missing
    assert all(pair in new for pair in old)
    assert {key: row.model_dump_json() for key, row in pool.items()} == before
    additions = [pair for pair in new if pair not in old]
    assert all(pair.rejected_decision == "act" and pair.rejection_reason for pair in additions)
    assert {pair.negative_kind for pair in additions} == {"wrong_decision", "unavailable_tool"}
    for pair in additions:
        assert pair.prompt == pool[pair.source_id].messages[:-1]
        assert pair.chosen == pool[pair.source_id].messages[-1:]
        assert json.loads(pair.rejected[0].content)[0]["name"] == "gmail.send_email"


def test_existing_training_counterparts_take_priority_over_authored_negatives() -> None:
    pool, groups = _triplet()
    source_id = next(key for key, row in pool.items() if decision(row) == "clarify")
    override = AuthoredActRejection(
        calls=[{"name": "clipboard.write", "arguments": {"text": "guessed"}}],
        reason="Intentional guess.",
    )
    pairs, missing = build_preference_pairs(
        pool, pool, groups, protected_inputs=set(), seed=1, act_rejections={source_id: override}
    )
    assert not missing
    pair = next(p for p in pairs if p.source_id == source_id and p.rejected_decision == "act")
    assert pair.counterpart_source_id is not None and pair.rejection_reason is None
    assert json.loads(pair.rejected[0].content)[0]["name"] == "gmail.send_email"


def test_authored_negatives_do_not_bypass_heldout_protection() -> None:
    pool, groups = _triplet()
    source_id = next(key for key, row in pool.items() if decision(row) == "clarify")
    override = AuthoredActRejection(
        calls=[{"name": "clipboard.write", "arguments": {"text": "guessed"}}],
        reason="Intentional guess.",
    )
    with pytest.raises(ValueError, match="held-out"):
        build_preference_pairs(
            {source_id: pool[source_id]},
            {},
            groups,
            protected_inputs={_sft_input_hash(pool[source_id])},
            seed=1,
            act_rejections={source_id: override},
        )


def test_authored_training_families_have_schema_valid_productive_negatives() -> None:
    sources = [
        row
        for row in generate_synthetic_baymax_examples()
        if row.id.startswith(
            ("capability_boundary_", "baymax_generated_clarify_", "baymax_generated_refuse_")
        )
    ]
    assert sources
    for source in sources:
        rejected = authored_act_rejection(source)
        assert rejected is not None and rejected.reason
        assert all(
            call["name"] not in {"request_clarification", "refuse_request"}
            for call in rejected.calls
        )
    assert (
        authored_act_rejection(
            next(
                row
                for row in generate_synthetic_baymax_examples()
                if row.id.startswith("baymax_triplet_")
            )
        )
        is None
    )


@pytest.mark.parametrize(
    "calls",
    [
        [{"name": "invented.tool", "arguments": {}}],
        [{"name": "request_clarification", "arguments": {"question": "why"}}],
        [{"name": "gmail.send_email", "arguments": {"recipient": "synthetic@example.com"}}],
        [{"name": "clipboard.write", "arguments": {"text": 1}}],
    ],
)
def test_authored_negatives_cannot_hide_contract_errors(calls: list[dict]) -> None:
    with pytest.raises(ValueError):
        AuthoredActRejection(calls=calls, reason="Intentional wrong action.")


def test_duplicate_inputs_do_not_create_repeated_preferences() -> None:
    pool, groups = _triplet()
    row = next(row for row in pool.values() if decision(row) == "act")
    anchors = {"first": row, "second": row}
    pairs, _ = build_preference_pairs(anchors, pool, groups, protected_inputs=set(), seed=1)
    assert len(pairs) == 2


def test_conflicting_targets_for_the_same_prompt_fail_instead_of_silent_selection() -> None:
    pool, groups = _triplet()
    act = next(row for row in pool.values() if decision(row) == "act")
    clarify = next(row for row in pool.values() if decision(row) == "clarify")
    conflicting = SFTExample(messages=act.messages[:-1] + clarify.messages[-1:])
    with pytest.raises(ValueError, match="conflicting chosen answers"):
        build_preference_pairs(
            {"first": act, "second": conflicting}, pool, groups, protected_inputs=set(), seed=1
        )


def test_heldout_overlap_blocks_chosen_and_rejected_counterparts() -> None:
    pool, groups = _triplet()
    act_id = next(key for key, row in pool.items() if decision(row) == "act")
    protected = {_sft_input_hash(pool[act_id])}
    with pytest.raises(ValueError, match="held-out"):
        build_preference_pairs(pool, pool, groups, protected_inputs=protected, seed=1)
    anchors = {key: row for key, row in pool.items() if decision(row) != "act"}
    pairs, missing = build_preference_pairs(
        anchors, pool, groups, protected_inputs=protected, seed=1
    )
    assert len(pairs) == 2 and len(missing) == 2
    assert all(pair.counterpart_source_id != act_id for pair in pairs)


def test_invalid_chosen_contract_is_not_weakened_for_dpo() -> None:
    pool, groups = _triplet()
    row = next(row for row in pool.values() if decision(row) == "clarify")
    wrong = row.model_copy(
        update={
            "messages": row.messages[:-1]
            + [
                SFTMessage(
                    role="assistant",
                    content='[{"name":"request_clarification","arguments":{"reason":"why"}}]',
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="invalid chosen target"):
        build_preference_pairs({"bad": wrong}, pool, groups, protected_inputs=set(), seed=1)


def _dataset_setup(tmp_path: Path) -> tuple[Path, Path]:
    pool, _ = _triplet()
    source = tmp_path / "source"
    reference = tmp_path / "reference"
    root = Path(__file__).resolve().parents[2]
    config = load_training_config(
        root
        / (
            "configs/sft/qwen2_5_1_5b_lora_target_compatible_v14_act200_anchored_val200_epochs2.json"
        )
    )
    for directory in (source, reference):
        directory.mkdir()
        splits = {}
        for split in ("train", "validation", "test"):
            rows = list(pool.values()) if split == "train" else []
            path = directory / f"{split}.jsonl"
            path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
            ids = list(pool) if directory == source else [f"baymax:train:{i}:r0" for i in range(3)]
            splits[split] = SFTSplitManifestEntry(
                path=str(path),
                count=len(rows),
                source_rows=[
                    SFTSourceRowRef(source_index=i, source_row_id=ids[i], row_hash=str(i))
                    for i in range(len(rows))
                ],
            )
        manifest = SFTSplitManifest(
            format_version=config.data.format_version,
            config_hash="pinned",
            dataset_id=config.data.dataset_id,
            revision="pinned",
            source_split="synthetic",
            seed=1,
            max_rows=None,
            split_ratios=SFTSplitRatios(),
            system_prompt_sha256=config.data.system_prompt_sha256,
            output_dir=str(directory),
            splits=splits,
            loaded_rows=3,
            converted_rows=3,
            skipped_rows=[],
        )
        (directory / "split-manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={
                    "revision": "pinned",
                    "train_path": reference / "train.jsonl",
                    "validation_path": reference / "validation.jsonl",
                    "test_path": reference / "test.jsonl",
                    "split_manifest_path": reference / "split-manifest.json",
                }
            )
        }
    )
    path = tmp_path / "config.json"
    write_training_config(path, training_config_with_hash(config))
    return path, source


def test_build_writes_trainer_data_and_reviewable_lineage_without_overwriting(
    tmp_path: Path,
) -> None:
    config, source = _dataset_setup(tmp_path)
    output = tmp_path / "preferences"
    report = build_dpo_dataset(
        reference_config=config, baymax_dir=source, output_dir=output, train_cap=3
    )
    assert report["pair_count"] == 6 and report["unique_prompts"] == 3
    assert report["train_sha256"] == _file_sha256(output / "train.jsonl")
    assert b"\r\n" not in (output / "train.jsonl").read_bytes()
    assert report["chosen_counts"] == {"act": 2, "clarify": 2, "refuse": 2}
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert all(set(row) == {"prompt", "chosen", "rejected"} for row in rows)
    review = [json.loads(line) for line in (output / "pairs-review.jsonl").read_text().splitlines()]
    assert all(row["source_id"] and row["negative_kind"] for row in review)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="new or empty"):
        build_dpo_dataset(
            reference_config=config, baymax_dir=source, output_dir=output, train_cap=3
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_build_rejects_generator_source_drift_before_publication(tmp_path: Path) -> None:
    config, source = _dataset_setup(tmp_path)
    path = source / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["messages"][1]["content"] += " edited"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "preferences"
    with pytest.raises(ValueError, match="generator/source drift"):
        build_dpo_dataset(
            reference_config=config, baymax_dir=source, output_dir=output, train_cap=3
        )
    assert not output.exists()
