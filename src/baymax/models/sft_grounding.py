"""Training-only grounding revision with a frozen external pool and held-out splits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date, timedelta
from itertools import islice
from pathlib import Path
from typing import Any

from baymax.models.baymax_sft import SyntheticBaymaxExample, validate_synthetic_baymax_example
from baymax.models.sft_dataset import SFTExample, SFTSplitManifest
from baymax.models.sft_filter import SPLITS, _staging_directory
from baymax.models.sft_mix import _file_sha256, _sft_input_hash
from baymax.models.sft_serialization import _json, _unique_object
from baymax.models.sft_validation import validate_sft_target_contract

TRANSFORM_VERSION = "training-grounding-v1"
TASK_TEXTS = (
    "Check the shelf labels",
    "Reconcile the spare-parts ledger",
    "Photograph the equipment tags",
    "Inspect the storage seals",
)


def _behavior(example: SFTExample) -> str:
    calls = json.loads(example.messages[-1].content)
    name = calls[0]["name"]
    return name if name in {"request_clarification", "refuse_request"} else "act"


def revise_grounding_example(
    example: SFTExample, *, index: int, seed: int
) -> tuple[SFTExample, list[str]]:
    """Revise eligible training families, not fixture IDs or held-out examples."""
    source_issues = validate_sft_target_contract(example, location=f"source:{index}")
    if source_issues:
        raise ValueError(f"Invalid source target: {source_issues[0].message}")
    header, separator, context_text = example.messages[1].content.partition("\n\nUser request:\n")
    if not separator:
        raise ValueError("Missing SFT request wrapper.")
    tools = json.loads(header.partition("\n")[2], object_pairs_hook=_unique_object)
    context = json.loads(context_text, object_pairs_hook=_unique_object)
    calls = json.loads(example.messages[-1].content, object_pairs_hook=_unique_object)
    request, state = context["request"], context["initial_state"]
    names = [call["name"] for call in calls]
    available = [
        tool["name"]
        for tool in tools
        if tool["name"] not in {"request_clarification", "refuse_request"}
    ]
    slot = (index + seed) % 4
    changes: list[str] = []
    urgency = re.fullmatch(r"Create a task to (.+) soon\.", request)
    evening = re.fullmatch(r"Schedule (.+) tonight\.", request)
    if (
        names == ["request_clarification"]
        and urgency
        and available == ["notion.create_task"]
        and not re.search(r"\bdue\s+soon\b", request, flags=re.IGNORECASE)
    ):
        title = urgency.group(1)
        context["request"] = f"Create a task to {title} due soon."
        calls[0]["arguments"] = {"question": f"What due date should I use for {title}?"}
        changes.append("explicit-deadline-clarification")
    elif names == ["request_clarification"] and evening and available == ["calendar.create_event"]:
        title = evening.group(1)
        context["request"] = (
            f"Add {title} to this evening's calendar; the start time and length are undecided.",
            f"Tonight I need time for {title}. "
            "I have not chosen when it starts or how long it runs.",
            f"Set up a calendar entry for {title} tonight, "
            "with the time and duration still to decide.",
            f"Please put {title} on tonight's calendar. "
            "I still need to choose a start time and length.",
        )[slot]
        calls[0]["arguments"] = {
            "question": (
                f"What start time and duration should I use for {title} tonight?",
                f"When tonight should {title} start, and how many minutes should it last?",
                f"At what time tonight, and for how long, should I schedule {title}?",
                f"What time should {title} begin tonight, and what duration should I use?",
            )[slot]
        }
        changes.append("missing-fields-wording")
    elif names in (
        ["clipboard.read", "notion.create_task"],
        ["clipboard.read", "gmail.create_draft"],
    ):
        text = TASK_TEXTS[slot]
        state["clipboard"]["text"] = text
        arguments = calls[1]["arguments"]
        if names[1] == "notion.create_task":
            arguments["title"] = text
            context["request"] = (
                "Turn the text currently on my clipboard into a task; read it first.",
                "Use my current clipboard contents as the title of a new task after reading them.",
                "Read what I copied, then add that text to my task list.",
                "Make a task from the clipboard text, checking the clipboard before creating it.",
            )[slot]
            state["notion_tasks"] = [
                {
                    "id": f"grounding_other_{index}",
                    "title": "Archive the old inspection log",
                    "status": "open",
                }
            ]
        else:
            arguments["body"] = text
            arguments.pop("subject", None)
            recipient = arguments["recipient"]
            context["request"] = (
                f"Read the clipboard, then prepare a draft to {recipient} "
                "using its contents as the body.",
                f"The current clipboard text should be the body of a draft to {recipient}; "
                "read it first.",
                f"After reading what I copied, draft it to {recipient} without sending it.",
                f"Prepare an email draft to {recipient} from the text you read on my clipboard.",
            )[slot]
        changes.append("clipboard-content-binding")
    elif names == ["notion.create_task"] and slot % 2 == 0:
        arguments = calls[0]["arguments"]
        title = arguments["title"]
        if "due_date" in arguments and title.casefold() in request.casefold():
            arguments.pop("due_date")
            context["request"] = (
                f"Create a task to {title} soon; leave its due date unset.",
                f"Put {title} on my task list for when I can get to it, without a deadline.",
            )[slot // 2]
            changes.append("undated-task-act")
    elif names in (["calendar.update_event"], ["notion.update_task"]):
        key, argument = (
            ("calendar_events", "event_id")
            if names[0].startswith("calendar.")
            else ("notion_tasks", "task_id")
        )
        entities = state.get(key, [])
        if len(entities) == 1 and entities[0]["title"].casefold() in request.casefold():
            entity = entities[0]
            if entity["id"] not in request:
                entity["id"] = f"grounding_selected_{index}"
                calls[0]["arguments"][argument] = entity["id"]
                distractor = dict(
                    entity, id=f"grounding_other_{index}", title="Archive inspection notes"
                )
                if key == "calendar_events":
                    distractor["start_date"] = (
                        date.fromisoformat(entity["start_date"]) + timedelta(days=14)
                    ).isoformat()
                state[key] = [entity, distractor] if slot % 2 else [distractor, entity]
                changes.append("entity-state-distractor")
    if not changes:
        return example, []
    source = SyntheticBaymaxExample(
        id=f"{TRANSFORM_VERSION}:{index}",
        behavior_type=(
            names[0]
            if names[0] == "request_clarification"
            else ("multi_tool_sequence" if len(calls) > 1 else "direct_tool_call")
        ),
        user_input=context["request"],
        current_time=context["current_time"],
        available_tools=available,
        initial_state=state,
        assistant_calls=calls,
    )
    validate_synthetic_baymax_example(source)
    revised = example.model_copy(
        update={
            "messages": [
                example.messages[0],
                example.messages[1].model_copy(
                    update={"content": header + separator + _json(context, sort_keys=True)}
                ),
                example.messages[2].model_copy(
                    update={
                        "content": _json(
                            [
                                {
                                    "name": call["name"],
                                    "arguments": dict(sorted(call["arguments"].items())),
                                }
                                for call in calls
                            ]
                        )
                    }
                ),
            ]
        }
    )
    if _behavior(revised) != _behavior(example):
        raise ValueError("Grounding revision must not change behavioral class.")
    issues = validate_sft_target_contract(revised, location=source.id)
    if issues:
        raise ValueError(f"Invalid revised target: {issues[0].message}")
    return revised, changes


def grounding_preview(
    rows: list[SFTExample], source_ids: list[Any], *, seed: int
) -> tuple[list[SFTExample], list[dict[str, Any]]]:
    """Transform only BAYMAX source slots, preserving cap order and class counts."""
    if len(rows) != len(source_ids):
        raise ValueError("Rows and source identities differ in count.")
    revised, changes = [], []
    for index, (row, source_id) in enumerate(zip(rows, source_ids, strict=True), 1):
        candidate, rules = (
            revise_grounding_example(row, index=index, seed=seed)
            if str(source_id).startswith("baymax:")
            else (row, [])
        )
        revised.append(candidate)
        if rules:
            changes.append(
                {
                    "line": index,
                    "source_id": source_id,
                    "rules": rules,
                    "before": row.model_dump(mode="json"),
                    "after": candidate.model_dump(mode="json"),
                }
            )
    changed_lines = {change["line"] for change in changes}
    used_inputs = {
        _sft_input_hash(row) for number, row in enumerate(rows, 1) if number not in changed_lines
    }
    for change in changes:
        position = change["line"] - 1
        digest = _sft_input_hash(revised[position])
        if digest in used_inputs:
            if change["rules"] != ["undated-task-act"]:
                raise ValueError("A revised training input duplicates another capped input.")
            revised[position] = rows[position]
            change["after"] = rows[position].model_dump(mode="json")
            change["skipped_reason"] = (
                "Duplicate undated variant; retained the original dated example."
            )
        used_inputs.add(_sft_input_hash(revised[position]))
    if Counter(map(_behavior, revised)) != Counter(map(_behavior, rows)):
        raise ValueError("Grounding revision changed cap class counts.")
    input_counts = Counter(map(_sft_input_hash, revised))
    if any(
        input_counts[_sft_input_hash(revised[change["line"] - 1])] != 1
        for change in changes
        if "skipped_reason" not in change
    ):
        raise ValueError("Revised capped training inputs contain duplicates.")
    if not changes:
        raise ValueError("No eligible training examples were revised.")
    return revised, changes


def derive_grounding_dataset(
    *, dataset_dir: Path, output_dir: Path, train_cap: int = 1000
) -> SFTSplitManifest:
    """Publish revised training slots; copy every other row and held-out byte unchanged."""
    source, destination = dataset_dir.resolve(), output_dir.resolve()
    if (
        destination.exists()
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Use a new nonoverlapping dataset directory.")
    manifest_path = source / "split-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    parent = SFTSplitManifest.model_validate_json(manifest_bytes)
    if (
        set(parent.splits) != set(SPLITS)
        or train_cap < 1
        or sum(entry.count for entry in parent.splits.values()) != parent.converted_rows
        or parent.loaded_rows - len(parent.skipped_rows) != parent.converted_rows
    ):
        raise ValueError("Invalid source splits or training cap.")
    paths = {}
    for split, entry in parent.splits.items():
        path = Path(entry.path)
        if not path.is_absolute() and not path.exists():
            path = source / path.name
        path = path.resolve()
        if not path.is_relative_to(source) or len(entry.source_rows) != entry.count:
            raise ValueError("Invalid source path or provenance count.")
        paths[split] = path
    parent_hashes = {split: _file_sha256(path) for split, path in paths.items()}
    with paths["train"].open(encoding="utf-8") as stream:
        prefix = [SFTExample.model_validate_json(line) for line in islice(stream, train_cap)]
    if len(prefix) != train_cap:
        raise ValueError("Training source is shorter than the requested cap.")
    revised, changes = grounding_preview(
        prefix,
        [r.source_row_id for r in parent.splits["train"].source_rows[:train_cap]],
        seed=parent.seed,
    )
    applied = [change for change in changes if "skipped_reason" not in change]
    changed_lines = {change["line"] for change in applied}
    changed_inputs = {_sft_input_hash(revised[line - 1]) for line in changed_lines}
    recipe = {
        "transform_version": TRANSFORM_VERSION,
        "parent_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "parent_files": parent_hashes,
        "train_cap": train_cap,
    }
    config_hash = hashlib.sha256(_json(recipe, sort_keys=True).encode()).hexdigest()
    output_hashes, entries = {}, {}
    seen_train: Counter[str] = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _staging_directory(destination) as staging:
        for split in SPLITS:
            count = 0
            with (
                paths[split].open("rb") as stream,
                (staging / f"{split}.jsonl").open("xb") as output,
            ):
                for count, line in enumerate(stream, 1):
                    example = SFTExample.model_validate_json(line)
                    if (
                        hashlib.sha256(example.messages[0].content.encode()).hexdigest()
                        != parent.system_prompt_sha256
                    ):
                        raise ValueError("Source system prompt differs from its manifest.")
                    if split == "train" and count in changed_lines:
                        example = revised[count - 1]
                        line = (example.model_dump_json() + "\n").encode()
                    digest = _sft_input_hash(example)
                    if split == "train":
                        seen_train[digest] += 1
                    elif digest in changed_inputs:
                        raise ValueError(
                            "A revised training input overlaps a protected held-out input."
                        )
                    output.write(line)
            if count != parent.splits[split].count:
                raise ValueError("Source file count differs from its manifest.")
            output_hashes[split] = _file_sha256(staging / f"{split}.jsonl")
            entries[split] = parent.splits[split].model_copy(
                update={"path": str(output_dir / f"{split}.jsonl")}
            )
        if any(seen_train[digest] != 1 for digest in changed_inputs):
            raise ValueError("Revised training inputs contain duplicates.")
        if (
            any(_file_sha256(paths[split]) != parent_hashes[split] for split in SPLITS)
            or manifest_path.read_bytes() != manifest_bytes
        ):
            raise ValueError("Source changed during derivation.")
        if any(output_hashes[split] != parent_hashes[split] for split in ("validation", "test")):
            raise ValueError("Held-out bytes changed.")
        revision = hashlib.sha256(
            _json({"config_hash": config_hash, "files": output_hashes}, sort_keys=True).encode()
        ).hexdigest()
        derived = parent.model_copy(
            update={
                "config_hash": config_hash,
                "revision": revision,
                "output_dir": str(output_dir),
                "splits": entries,
            }
        )
        (staging / "split-manifest.json").write_text(
            derived.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (staging / "grounding-changes.json").write_text(
            json.dumps(changes, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "grounding-provenance.json").write_text(
            json.dumps(
                {
                    **recipe,
                    "parent_revision": parent.revision,
                    "output_files": output_hashes,
                    "changed_rows": len(applied),
                    "skipped_duplicate_rows": len(changes) - len(applied),
                    "rules": dict(Counter(rule for change in applied for rule in change["rules"])),
                    "class_counts": dict(Counter(map(_behavior, revised))),
                    "baymax_class_counts": dict(
                        Counter(
                            _behavior(row)
                            for row, ref in zip(
                                revised, parent.splits["train"].source_rows[:train_cap], strict=True
                            )
                            if str(ref.source_row_id).startswith("baymax:")
                        )
                    ),
                    "source_refs": (
                        "Parent identities preserved; revised contents are recorded "
                        "in grounding-changes.json."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    return derived
