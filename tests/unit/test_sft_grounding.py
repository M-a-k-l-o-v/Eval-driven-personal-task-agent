from __future__ import annotations

import hashlib
import json
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
)
from baymax.models.sft_grounding import (
    derive_grounding_dataset,
    grounding_preview,
    revise_grounding_example,
)
from baymax.models.sft_serialization import name_first_example
from baymax.models.sft_validation import validate_sft_dataset
from baymax.models.training_config import (
    load_training_config,
    training_config_with_hash,
    validate_training_config,
    write_training_config,
)
from scripts.prepare_sft_grounding import DEFAULT_REFERENCE, prepare_experiment

REFERENCE = Path(__file__).resolve().parents[2] / DEFAULT_REFERENCE


def _example(kind: str) -> SFTExample:
    request, tools, state, calls = {
        "urgency": (
            "Create a task to review inventory soon.",
            ["notion.create_task"],
            {},
            [{"name": "request_clarification", "arguments": {"question": "What due date?"}}],
        ),
        "calendar": (
            "Schedule inventory review tonight.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [
                {
                    "name": "request_clarification",
                    "arguments": {"question": "What time and duration?"},
                }
            ],
        ),
        "clipboard": (
            "Read my clipboard and create a task from it.",
            ["clipboard.read", "notion.create_task"],
            {"clipboard": {"text": "Inspect stock"}, "notion_tasks": []},
            [
                {"name": "clipboard.read", "arguments": {}},
                {
                    "name": "notion.create_task",
                    "arguments": {"title": "Inspect stock", "status": "open"},
                },
            ],
        ),
        "dated": (
            "Create a task to review inventory due 2026-06-04.",
            ["notion.create_task"],
            {},
            [
                {
                    "name": "notion.create_task",
                    "arguments": {"title": "review inventory", "due_date": "2026-06-04"},
                }
            ],
        ),
        "entity": (
            "Move the inventory review to 13:00.",
            ["calendar.update_event"],
            {
                "calendar_events": [
                    {
                        "id": "event-unit",
                        "title": "inventory review",
                        "start_date": "2026-06-04",
                        "start_time": "10:00",
                        "duration_minutes": 30,
                    }
                ]
            },
            [
                {
                    "name": "calendar.update_event",
                    "arguments": {"event_id": "event-unit", "start_time": "13:00"},
                }
            ],
        ),
        "refuse": (
            "Order a new stock cabinet.",
            ["notion.create_task"],
            {},
            [{"name": "refuse_request", "arguments": {"reason": "No ordering tool is available."}}],
        ),
    }[kind]
    source = SyntheticBaymaxExample(
        id=f"unit-{kind}",
        behavior_type=(
            "request_clarification"
            if kind in {"urgency", "calendar"}
            else "refuse_request"
            if kind == "refuse"
            else "multi_tool_sequence"
            if kind == "clipboard"
            else "direct_tool_call"
        ),
        user_input=request,
        current_time="2026-06-01T09:00:00Z",
        available_tools=tools,
        initial_state=state,
        assistant_calls=calls,
    )
    return name_first_example(
        synthetic_baymax_example_to_sft_example(source, system_prompt=DEFAULT_SFT_SYSTEM_PROMPT),
        location="unit source",
    )


def _context(example: SFTExample) -> dict:
    return json.loads(example.messages[1].content.partition("\n\nUser request:\n")[2])


@pytest.mark.parametrize("kind", ["urgency", "calendar", "clipboard", "dated", "entity"])
def test_revision_keeps_source_immutable_and_preserves_class(kind: str) -> None:
    source = _example(kind)
    before = source.model_dump_json()
    candidate, rules = revise_grounding_example(source, index=1, seed=1)
    assert rules and source.model_dump_json() == before
    assert candidate.messages[0] == source.messages[0]
    assert (
        candidate.messages[1].content.partition("\n\nUser request:\n")[0]
        == source.messages[1].content.partition("\n\nUser request:\n")[0]
    )
    calls = json.loads(candidate.messages[-1].content)
    assert len(calls) == len(json.loads(source.messages[-1].content))
    assert all(list(call) == ["name", "arguments"] for call in calls)
    context = _context(candidate)
    if kind == "urgency":
        assert "due soon" in context["request"] and "due date" in calls[0]["arguments"]["question"]
    elif kind == "calendar":
        assert (
            "time" in calls[0]["arguments"]["question"]
            and "long" in calls[0]["arguments"]["question"]
        )
    elif kind == "clipboard":
        assert calls[1]["arguments"]["title"] == context["initial_state"]["clipboard"]["text"]
    elif kind == "dated":
        assert "due_date" not in calls[0]["arguments"]
    else:
        entities = context["initial_state"]["calendar_events"]
        assert len(entities) == 2 and len({entity["id"] for entity in entities}) == 2
        selected = next(
            entity for entity in entities if entity["id"] == calls[0]["arguments"]["event_id"]
        )
        assert selected["title"] == "inventory review"
        assert entities[0]["start_date"] != entities[1]["start_date"]


def test_preview_preserves_refusals_external_slots_and_variation_is_deterministic() -> None:
    rows = [_example(kind) for kind in ("urgency", "refuse", "clipboard")]
    output, changes = grounding_preview(
        rows, ["baymax:train:1:r0", "baymax:train:2:r0", "xlam:train:3:r0"], seed=1
    )
    assert [change["line"] for change in changes] == [1]
    assert output[1:] == rows[1:]
    assert grounding_preview(
        rows, ["baymax:train:1:r0", "baymax:train:2:r0", "xlam:train:3:r0"], seed=1
    ) == (output, changes)
    variants = [
        revise_grounding_example(_example("clipboard"), index=i, seed=1)[0] for i in range(4)
    ]
    assert len({row.messages[1].content for row in variants}) == 4


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    reference = load_training_config(REFERENCE)
    rows = [
        _example(kind) for kind in ("urgency", "calendar", "clipboard", "dated", "entity", "refuse")
    ]
    prompt = rows[0].messages[0].content
    splits = {}
    for split in ("train", "validation", "test"):
        examples = (
            rows
            if split == "train"
            else [
                _example("refuse").model_copy(
                    update={
                        "messages": [
                            rows[-1].messages[0],
                            rows[-1]
                            .messages[1]
                            .model_copy(
                                update={
                                    "content": rows[-1]
                                    .messages[1]
                                    .content.replace("stock cabinet", split + " cabinet")
                                }
                            ),
                            rows[-1].messages[2],
                        ]
                    }
                )
            ]
        )
        path = source / f"{split}.jsonl"
        path.write_bytes(b"".join((row.model_dump_json() + "\r\n").encode() for row in examples))
        refs = [
            SFTSourceRowRef(
                source_index=i,
                source_row_id=(f"xlam:{split}:{i}:r0" if i == 3 else f"baymax:{split}:{i}:r0"),
                row_hash=hashlib.sha256(f"{split}:{i}".encode()).hexdigest(),
            )
            for i in range(len(examples))
        ]
        splits[split] = SFTSplitManifestEntry(path=str(path), count=len(examples), source_rows=refs)
    parent = SFTSplitManifest(
        format_version=reference.data.format_version,
        config_hash="unit-parent",
        dataset_id=reference.data.dataset_id,
        revision="unit-parent",
        source_split="mixed",
        seed=1,
        max_rows=None,
        split_ratios=SFTSplitRatios(),
        system_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        output_dir=str(source),
        splits=splits,
        loaded_rows=8,
        converted_rows=8,
        skipped_rows=[],
    )
    manifest_path = source / "split-manifest.json"
    manifest_path.write_text(parent.model_dump_json(), encoding="utf-8")
    report_path = source / "validation-report.json"
    report_path.write_text("{}", encoding="utf-8")
    eval_dir = tmp_path / "unused-eval"
    eval_dir.mkdir()
    data = reference.data.model_copy(
        update={
            "revision": parent.revision,
            "system_prompt_sha256": parent.system_prompt_sha256,
            "split_manifest_path": manifest_path,
            "validation_report_path": report_path,
            "eval_scenarios_paths": [eval_dir],
            **{f"{split}_path": Path(entry.path) for split, entry in splits.items()},
        }
    )
    config_path = tmp_path / "reference.json"
    write_training_config(
        config_path, training_config_with_hash(reference.model_copy(update={"data": data}))
    )
    return source, tmp_path / "derived", config_path


def test_derivation_preserves_heldout_bytes_provenance_and_external_rows(tmp_path: Path) -> None:
    source, output, _ = _setup(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    parent = SFTSplitManifest.model_validate_json(before["split-manifest.json"])
    derived = derive_grounding_dataset(dataset_dir=source, output_dir=output, train_cap=6)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    for split in ("validation", "test"):
        assert (output / f"{split}.jsonl").read_bytes() == before[f"{split}.jsonl"]
    for split in parent.splits:
        assert derived.splits[split].source_rows == parent.splits[split].source_rows
        assert derived.splits[split].count == parent.splits[split].count
    assert (output / "train.jsonl").read_bytes().splitlines()[3] == before[
        "train.jsonl"
    ].splitlines()[3]
    assert validate_sft_dataset(output).valid


@pytest.mark.parametrize(
    "defect", ["output", "overlap", "heldout_duplicate", "train_duplicate", "count"]
)
def test_derivation_rejects_unsafe_artifacts(tmp_path: Path, defect: str) -> None:
    source, output, _ = _setup(tmp_path)
    if defect == "output":
        output.mkdir()
        (output / "keep").write_text("keep")
    elif defect == "overlap":
        output = source / "nested"
    elif defect in {"heldout_duplicate", "train_duplicate"}:
        candidate = revise_grounding_example(_example("urgency"), index=1, seed=1)[0]
        if defect == "heldout_duplicate":
            (source / "validation.jsonl").write_text(candidate.model_dump_json() + "\n")
        else:
            path = source / "train.jsonl"
            lines = path.read_text().splitlines()
            lines[5] = candidate.model_dump_json()
            path.write_text("\n".join(lines) + "\n")
    else:
        with (source / "train.jsonl").open("a") as stream:
            stream.write(_example("refuse").model_dump_json() + "\n")
    with pytest.raises(ValueError):
        derive_grounding_dataset(dataset_dir=source, output_dir=output, train_cap=6)
    if defect == "output":
        assert (output / "keep").read_text() == "keep"
    else:
        assert not output.exists()


def test_preparation_keeps_training_recipe_and_validates_new_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, reference_path = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "candidate.json"
    reference = load_training_config(reference_path)
    candidate = prepare_experiment(
        reference_config=reference_path, output_dir=output, config_path=config_path, train_cap=6
    )
    assert candidate is not None
    assert candidate.training == reference.training and candidate.lora == reference.lora
    assert candidate.runtime.seed == reference.runtime.seed
    assert candidate.data.train_path.parent == output
    assert candidate.data.revision != reference.data.revision
    assert (
        validate_training_config(config_path, check_paths=True).valid is False
    )  # Validation report is not built by preparation.
    assert validate_training_config(config_path).valid
    assert source != output


def test_corrected_deadline_request_is_not_rewritten_as_due_due_soon() -> None:
    corrected, _ = revise_grounding_example(_example("urgency"), index=1, seed=1)
    unchanged, rules = revise_grounding_example(corrected, index=1, seed=1)
    assert unchanged == corrected and not rules


def test_duplicate_undated_variant_keeps_later_parent_and_records_the_skip() -> None:
    first = _example("dated")
    second = SFTExample.model_validate_json(
        first.model_dump_json().replace("2026-06-04", "2026-06-05")
    )
    rows = [first, _example("refuse"), _example("refuse"), _example("refuse"), second]
    output, changes = grounding_preview(rows, [f"baymax:train:{i}:r0" for i in range(5)], seed=1)
    assert output[0] != first and output[4] == second
    assert "skipped_reason" in changes[1]
    assert changes[1]["before"] == changes[1]["after"]
    assert changes[1]["line"] == 5


def test_preparation_preview_does_not_publish_artifacts(tmp_path: Path) -> None:
    source, output, reference = _setup(tmp_path)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    config_path = tmp_path / "candidate.json"
    assert (
        prepare_experiment(
            reference_config=reference,
            output_dir=output,
            config_path=config_path,
            dry_run=True,
            train_cap=6,
        )
        is None
    )
    assert not output.exists() and not config_path.exists()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
