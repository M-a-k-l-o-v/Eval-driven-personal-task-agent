from __future__ import annotations

import json
from pathlib import Path

from baymax.models.baymax_sft import BaymaxSFTBuildConfig, build_baymax_sft_dataset
from baymax.models.sft_dataset import SFTBuildConfig, build_sft_dataset_from_rows
from baymax.models.sft_mix import SFTMixConfig, mix_sft_datasets
from baymax.models.sft_validation import validate_sft_dataset


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

    mixed_manifest = mix_sft_datasets(
        SFTMixConfig(
            xlam_dir=xlam_dir,
            baymax_dir=baymax_dir,
            output_dir=mixed_dir,
            seed=1,
            train_baymax_repeat_count=2,
            eval_baymax_repeat_count=1,
        )
    )

    assert mixed_manifest.dataset_id == "mixed:example/xlam+baymax/synthetic-control-v1"
    expected_baymax_train_count = _weighted_baymax_train_count(baymax_dir, base=2, extra=2)
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


def test_mix_sft_datasets_only_adds_boundary_repeats_to_train_split(tmp_path: Path) -> None:
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
            train_baymax_boundary_extra_repeat_count=2,
            eval_baymax_repeat_count=1,
        )
    )

    assert mixed_manifest.splits["train"].count == (
        xlam_manifest.splits["train"].count
        + _weighted_baymax_train_count(baymax_dir, base=1, extra=2)
    )
    assert mixed_manifest.splits["validation"].count == (
        xlam_manifest.splits["validation"].count + baymax_manifest.splits["validation"].count
    )
    assert mixed_manifest.splits["test"].count == (
        xlam_manifest.splits["test"].count + baymax_manifest.splits["test"].count
    )


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


def _weighted_baymax_train_count(dataset_dir: Path, *, base: int, extra: int) -> int:
    total = 0
    for payload in _assistant_payloads(_read_jsonl(dataset_dir / "train.jsonl")):
        first_call = payload[0]
        repeat_count = base
        if first_call["name"] in {"request_clarification", "refuse_request"}:
            repeat_count += extra
        total += repeat_count

    return total
