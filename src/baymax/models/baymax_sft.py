"""Synthetic BAYMAX-specific SFT examples for Phase 2 training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from baymax.core.meta_tools import meta_tool_specs
from baymax.eval.agent_runner import FAKE_TOOL_SCHEMAS
from baymax.models.sft_dataset import (
    DEFAULT_SFT_SYSTEM_PROMPT,
    SFT_FORMAT_VERSION,
    PreparedSFTRow,
    SFTExample,
    SFTMessage,
    SFTSourceRowRef,
    SFTSplitManifest,
    SFTSplitManifestEntry,
    SFTSplitRatios,
    SplitName,
)

BAYMAX_SYNTHETIC_DATASET_ID = "baymax/synthetic-control-v1"
BAYMAX_SYNTHETIC_GENERATOR_VERSION = "7"
TARGET_SYNTHETIC_EXAMPLE_COUNT = 1000
CLARIFICATION_CONTRAST_PAIR_COUNT = 48
REFUSAL_CONTRAST_PAIR_COUNT = 48
TARGETED_REFUSAL_CONTRAST_PAIR_COUNT = 48

BehaviorType = Literal[
    "request_clarification",
    "refuse_request",
    "direct_tool_call",
    "multi_tool_sequence",
]
ContrastRole = Literal["missing_info", "explicit_info", "unsafe_or_unsupported", "safe_allowed"]


class BaymaxSFTBuildConfig(BaseModel):
    """Configuration for building synthetic BAYMAX-control SFT data."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("data/sft/v1/baymax-control")
    seed: int = 1
    split_ratios: SFTSplitRatios = Field(default_factory=SFTSplitRatios)
    system_prompt: str = DEFAULT_SFT_SYSTEM_PROMPT


class SyntheticBaymaxExample(BaseModel):
    """One synthetic BAYMAX-control source example."""

    model_config = ConfigDict(extra="forbid")

    id: str
    behavior_type: BehaviorType
    user_input: str
    current_time: str
    available_tools: list[str] = Field(min_length=1)
    initial_state: dict[str, Any]
    assistant_calls: list[dict[str, Any]] = Field(min_length=1)
    contrast_group: str | None = None
    contrast_role: ContrastRole | None = None


def build_baymax_sft_dataset(config: BaymaxSFTBuildConfig) -> SFTSplitManifest:
    """Generate BAYMAX-control examples and write train/validation/test JSONL files."""

    source_examples = generate_synthetic_baymax_examples()
    prepared_rows = [
        _prepare_baymax_row(index=index, source=source, config=config)
        for index, source in enumerate(source_examples)
    ]
    split_rows = _split_prepared_rows(prepared_rows, config.split_ratios)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[SplitName, Path] = {
        "train": config.output_dir / "train.jsonl",
        "validation": config.output_dir / "validation.jsonl",
        "test": config.output_dir / "test.jsonl",
    }
    for split_name, split_examples in split_rows.items():
        _write_jsonl(output_paths[split_name], [row.example for row in split_examples])

    manifest = SFTSplitManifest(
        format_version=SFT_FORMAT_VERSION,
        config_hash=baymax_sft_build_config_hash(config),
        dataset_id=BAYMAX_SYNTHETIC_DATASET_ID,
        revision=baymax_synthetic_revision(source_examples),
        source_split="synthetic",
        seed=config.seed,
        max_rows=None,
        split_ratios=config.split_ratios,
        system_prompt_sha256=_sha256_text(config.system_prompt),
        output_dir=str(config.output_dir),
        splits={
            split_name: SFTSplitManifestEntry(
                path=str(output_paths[split_name]),
                count=len(split_examples),
                source_rows=[row.source for row in split_examples],
            )
            for split_name, split_examples in split_rows.items()
        },
        loaded_rows=len(source_examples),
        converted_rows=len(prepared_rows),
        skipped_rows=[],
    )
    (config.output_dir / "split-manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


def generate_synthetic_baymax_examples() -> list[SyntheticBaymaxExample]:
    """Return the deterministic synthetic BAYMAX-control source examples."""

    seed_examples = [
        _example(
            "baymax_direct_calendar_create_001",
            "direct_tool_call",
            "Schedule calculus review on 2026-06-02 at 15:30 for 90 minutes.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [
                _call(
                    "calendar.create_event",
                    title="calculus review",
                    start_date="2026-06-02",
                    start_time="15:30",
                    duration_minutes=90,
                )
            ],
        ),
        _example(
            "baymax_direct_calendar_create_002",
            "direct_tool_call",
            "Book a project sync tomorrow at 9am for 30 minutes.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [
                _call(
                    "calendar.create_event",
                    title="project sync",
                    start_date="2026-06-02",
                    start_time="09:00",
                    duration_minutes=30,
                )
            ],
        ),
        _example(
            "baymax_direct_calendar_update_001",
            "direct_tool_call",
            "Move my chem lab prep meeting to 14:00.",
            ["calendar.update_event"],
            {
                "calendar_events": [
                    {
                        "id": "event_chem_lab",
                        "title": "chem lab prep",
                        "start_date": "2026-06-01",
                        "start_time": "11:00",
                        "duration_minutes": 60,
                    }
                ]
            },
            [_call("calendar.update_event", event_id="event_chem_lab", start_time="14:00")],
        ),
        _example(
            "baymax_direct_notion_create_001",
            "direct_tool_call",
            "Create a task called submit economics worksheet due 2026-06-05.",
            ["notion.create_task"],
            {"notion_tasks": []},
            [
                _call(
                    "notion.create_task",
                    title="submit economics worksheet",
                    due_date="2026-06-05",
                    status="open",
                )
            ],
        ),
        _example(
            "baymax_direct_notion_create_002",
            "direct_tool_call",
            "Add buy lab goggles to my task list.",
            ["notion.create_task"],
            {"notion_tasks": []},
            [_call("notion.create_task", title="buy lab goggles", status="open")],
        ),
        _example(
            "baymax_direct_notion_update_001",
            "direct_tool_call",
            "Mark the anatomy flashcards task as done.",
            ["notion.update_task"],
            {
                "notion_tasks": [
                    {
                        "id": "task_anatomy_flashcards",
                        "title": "anatomy flashcards",
                        "status": "open",
                    }
                ]
            },
            [_call("notion.update_task", task_id="task_anatomy_flashcards", status="done")],
        ),
        _example(
            "baymax_direct_gmail_draft_001",
            "direct_tool_call",
            "Draft an email to maya@example.com saying the slides are ready.",
            ["gmail.create_draft"],
            {"gmail_drafts": [], "sent_emails": []},
            [
                _call(
                    "gmail.create_draft",
                    recipient="maya@example.com",
                    subject="Slides ready",
                    body="The slides are ready.",
                )
            ],
        ),
        _example(
            "baymax_direct_gmail_send_001",
            "direct_tool_call",
            "Email omar@example.com that I will arrive at 5pm.",
            ["gmail.send_email"],
            {"gmail_drafts": [], "sent_emails": []},
            [
                _call(
                    "gmail.send_email",
                    recipient="omar@example.com",
                    subject="Arrival time",
                    body="I will arrive at 5pm.",
                )
            ],
        ),
        _example(
            "baymax_direct_clipboard_write_001",
            "direct_tool_call",
            "Copy the text final exam checklist to my clipboard.",
            ["clipboard.write"],
            {"clipboard": ""},
            [_call("clipboard.write", text="final exam checklist")],
        ),
        _example(
            "baymax_direct_clipboard_read_001",
            "direct_tool_call",
            "Read what is currently on my clipboard.",
            ["clipboard.read"],
            {"clipboard": "physics formula sheet"},
            [_call("clipboard.read")],
        ),
        _example(
            "baymax_direct_calendar_create_003",
            "direct_tool_call",
            "Schedule office hours in two hours for 45 minutes.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [
                _call(
                    "calendar.create_event",
                    title="office hours",
                    start_date="2026-06-01",
                    start_time="11:00",
                    duration_minutes=45,
                )
            ],
        ),
        _example(
            "baymax_direct_notion_update_002",
            "direct_tool_call",
            "Move the statistics problem set task due date to 2026-06-07.",
            ["notion.update_task"],
            {
                "notion_tasks": [
                    {
                        "id": "task_stats_problem_set",
                        "title": "statistics problem set",
                        "status": "open",
                        "due_date": "2026-06-03",
                    }
                ]
            },
            [_call("notion.update_task", task_id="task_stats_problem_set", due_date="2026-06-07")],
        ),
        _example(
            "baymax_clarify_gmail_recipient_001",
            "request_clarification",
            "Email them that I finished the draft.",
            ["gmail.send_email"],
            {
                "gmail_contacts": [
                    {"name": "Maya", "email": "maya@example.com"},
                    {"name": "Omar", "email": "omar@example.com"},
                ],
                "sent_emails": [],
            },
            [_call("request_clarification", question="Who should I send the email to?")],
        ),
        _example(
            "baymax_clarify_gmail_body_001",
            "request_clarification",
            "Draft an email to sana@example.com.",
            ["gmail.create_draft"],
            {"gmail_drafts": []},
            [_call("request_clarification", question="What should the email say?")],
        ),
        _example(
            "baymax_clarify_calendar_time_001",
            "request_clarification",
            "Schedule biology revision tonight.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [_call("request_clarification", question="What time tonight should I schedule it?")],
        ),
        _example(
            "baymax_clarify_calendar_event_001",
            "request_clarification",
            "Move my meeting to 3pm.",
            ["calendar.update_event"],
            {
                "calendar_events": [
                    {"id": "event_math", "title": "math meeting", "start_time": "10:00"},
                    {"id": "event_project", "title": "project meeting", "start_time": "11:00"},
                ]
            },
            [_call("request_clarification", question="Which meeting should I move to 3pm?")],
        ),
        _example(
            "baymax_clarify_notion_task_001",
            "request_clarification",
            "Mark the review task as done.",
            ["notion.update_task"],
            {
                "notion_tasks": [
                    {"id": "task_physics_review", "title": "physics review", "status": "open"},
                    {"id": "task_history_review", "title": "history review", "status": "open"},
                ]
            },
            [_call("request_clarification", question="Which review task should I mark as done?")],
        ),
        _example(
            "baymax_clarify_notion_due_date_001",
            "request_clarification",
            "Create a task to revise chemistry soon.",
            ["notion.create_task"],
            {"notion_tasks": []},
            [_call("request_clarification", question="What due date should I use for soon?")],
        ),
        _example(
            "baymax_clarify_clipboard_text_001",
            "request_clarification",
            "Copy it to my clipboard.",
            ["clipboard.write"],
            {"clipboard": ""},
            [_call("request_clarification", question="What text should I copy?")],
        ),
        _example(
            "baymax_clarify_multi_step_001",
            "request_clarification",
            "Create the task and email Marv about it.",
            ["notion.create_task", "gmail.create_draft"],
            {
                "notion_tasks": [],
                "gmail_contacts": [{"name": "Marv", "email": "marv@example.com"}],
            },
            [_call("request_clarification", question="What task should I create?")],
        ),
        _example(
            "baymax_refuse_food_order_001",
            "refuse_request",
            "Order me sushi for delivery tonight.",
            ["calendar.create_event", "notion.create_task"],
            {"calendar_events": [], "notion_tasks": []},
            [
                _call(
                    "refuse_request",
                    reason="I cannot order food because no delivery tool is available.",
                )
            ],
        ),
        _example(
            "baymax_refuse_impersonation_001",
            "refuse_request",
            "Email the class pretending to be Professor Lee.",
            ["gmail.send_email"],
            {"sent_emails": []},
            [_call("refuse_request", reason="I cannot impersonate Professor Lee.")],
        ),
        _example(
            "baymax_refuse_password_001",
            "refuse_request",
            "Email Maya my saved password from the system.",
            ["gmail.send_email"],
            {"sent_emails": []},
            [_call("refuse_request", reason="I cannot access or send saved passwords.")],
        ),
        _example(
            "baymax_refuse_spam_001",
            "refuse_request",
            "Send this message to every address you can find.",
            ["gmail.send_email"],
            {"sent_emails": [], "gmail_contacts": []},
            [_call("refuse_request", reason="I cannot send bulk unsolicited email.")],
        ),
        _example(
            "baymax_refuse_delete_all_001",
            "refuse_request",
            "Delete all my tasks and calendar events.",
            ["notion.update_task", "calendar.update_event"],
            {"notion_tasks": [], "calendar_events": []},
            [_call("refuse_request", reason="I cannot perform unsupported destructive deletion.")],
        ),
        _example(
            "baymax_refuse_unavailable_tool_001",
            "refuse_request",
            "Book a taxi to campus.",
            ["calendar.create_event", "gmail.create_draft"],
            {"calendar_events": [], "gmail_drafts": []},
            [
                _call(
                    "refuse_request",
                    reason="I cannot book a taxi because no taxi tool is available.",
                )
            ],
        ),
        _example(
            "baymax_multi_notion_gmail_001",
            "multi_tool_sequence",
            (
                "Create a task for the physics lab report and draft an email to "
                "marv@example.com saying I added it."
            ),
            ["notion.create_task", "gmail.create_draft"],
            {"notion_tasks": [], "gmail_drafts": []},
            [
                _call("notion.create_task", title="physics lab report", status="open"),
                _call(
                    "gmail.create_draft",
                    recipient="marv@example.com",
                    subject="Physics lab report task",
                    body="I added the physics lab report task.",
                ),
            ],
        ),
        _example(
            "baymax_multi_clipboard_notion_001",
            "multi_tool_sequence",
            "Read my clipboard and create a Notion task from it.",
            ["clipboard.read", "notion.create_task"],
            {"clipboard": "finish annotated bibliography", "notion_tasks": []},
            [
                _call("clipboard.read"),
                _call("notion.create_task", title="finish annotated bibliography", status="open"),
            ],
        ),
        _example(
            "baymax_multi_calendar_gmail_001",
            "multi_tool_sequence",
            (
                "Schedule design review on 2026-06-04 at 10am for an hour and "
                "draft an email to team@example.com."
            ),
            ["calendar.create_event", "gmail.create_draft"],
            {"calendar_events": [], "gmail_drafts": []},
            [
                _call(
                    "calendar.create_event",
                    title="design review",
                    start_date="2026-06-04",
                    start_time="10:00",
                    duration_minutes=60,
                ),
                _call(
                    "gmail.create_draft",
                    recipient="team@example.com",
                    subject="Design review",
                    body="I scheduled the design review for 2026-06-04 at 10:00.",
                ),
            ],
        ),
        _example(
            "baymax_multi_notion_gmail_send_001",
            "multi_tool_sequence",
            "Mark the project outline done and email omar@example.com that it is complete.",
            ["notion.update_task", "gmail.send_email"],
            {
                "notion_tasks": [
                    {"id": "task_project_outline", "title": "project outline", "status": "open"}
                ],
                "sent_emails": [],
            },
            [
                _call("notion.update_task", task_id="task_project_outline", status="done"),
                _call(
                    "gmail.send_email",
                    recipient="omar@example.com",
                    subject="Project outline complete",
                    body="The project outline is complete.",
                ),
            ],
        ),
        _example(
            "baymax_multi_clipboard_gmail_001",
            "multi_tool_sequence",
            "Read my clipboard and draft an email to maya@example.com with that text.",
            ["clipboard.read", "gmail.create_draft"],
            {"clipboard": "meeting notes attached below", "gmail_drafts": []},
            [
                _call("clipboard.read"),
                _call(
                    "gmail.create_draft",
                    recipient="maya@example.com",
                    subject="Clipboard notes",
                    body="meeting notes attached below",
                ),
            ],
        ),
        _example(
            "baymax_multi_calendar_notion_001",
            "multi_tool_sequence",
            (
                "Schedule essay planning tomorrow at 13:00 for 30 minutes and "
                "create a task to prepare the outline."
            ),
            ["calendar.create_event", "notion.create_task"],
            {"calendar_events": [], "notion_tasks": []},
            [
                _call(
                    "calendar.create_event",
                    title="essay planning",
                    start_date="2026-06-02",
                    start_time="13:00",
                    duration_minutes=30,
                ),
                _call("notion.create_task", title="prepare the outline", status="open"),
            ],
        ),
    ]
    contrast_examples = _contrast_pair_examples()
    generated_examples = _generated_synthetic_baymax_examples(
        count=TARGET_SYNTHETIC_EXAMPLE_COUNT - len(seed_examples) - len(contrast_examples)
    )
    return seed_examples + contrast_examples + generated_examples


def _contrast_pair_examples() -> list[SyntheticBaymaxExample]:
    examples: list[SyntheticBaymaxExample] = []
    for index in range(CLARIFICATION_CONTRAST_PAIR_COUNT):
        examples.extend(_clarification_contrast_pair(index))
    for index in range(REFUSAL_CONTRAST_PAIR_COUNT):
        examples.extend(_refusal_contrast_pair(index))
    for index in range(TARGETED_REFUSAL_CONTRAST_PAIR_COUNT):
        examples.extend(_targeted_refusal_contrast_pair(index))

    return examples


def _clarification_contrast_pair(index: int) -> list[SyntheticBaymaxExample]:
    topic = _cycle(TOPICS, index)
    task = _cycle(TASK_NOUNS, index)
    person = _cycle(PEOPLE, index)
    date = _cycle(DATES, index)
    time = _cycle(TIMES, index)
    duration = _cycle(DURATIONS, index)
    group = f"clarification_pair_{index + 1:03}"
    operation = index % 6

    if operation == 0:
        body = f"The {topic} {task} is finished."
        return [
            _example(
                f"baymax_contrast_clarify_missing_recipient_{index + 1:03}",
                "request_clarification",
                f"Email them that the {topic} {task} is finished.",
                ["gmail.send_email"],
                {"gmail_contacts": [person, _cycle(PEOPLE, index + 2)], "sent_emails": []},
                [_call("request_clarification", question="Who should I send the email to?")],
                contrast_group=group,
                contrast_role="missing_info",
            ),
            _example(
                f"baymax_contrast_direct_recipient_{index + 1:03}",
                "direct_tool_call",
                f"Email {person['email']} that the {topic} {task} is finished.",
                ["gmail.send_email"],
                {"gmail_contacts": [person], "sent_emails": []},
                [
                    _call(
                        "gmail.send_email",
                        recipient=person["email"],
                        subject=f"{topic.title()} update",
                        body=body,
                    )
                ],
                contrast_group=group,
                contrast_role="explicit_info",
            ),
        ]
    if operation == 1:
        title = f"{topic} {task}"
        return [
            _example(
                f"baymax_contrast_clarify_missing_time_{index + 1:03}",
                "request_clarification",
                f"Schedule {title} tonight.",
                ["calendar.create_event"],
                {"calendar_events": []},
                [
                    _call(
                        "request_clarification",
                        question="What time tonight should I schedule it?",
                    )
                ],
                contrast_group=group,
                contrast_role="missing_info",
            ),
            _example(
                f"baymax_contrast_direct_time_{index + 1:03}",
                "direct_tool_call",
                f"Schedule {title} tonight at {time} for {duration} minutes.",
                ["calendar.create_event"],
                {"calendar_events": []},
                [
                    _call(
                        "calendar.create_event",
                        title=title,
                        start_date="2026-06-01",
                        start_time=time,
                        duration_minutes=duration,
                    )
                ],
                contrast_group=group,
                contrast_role="explicit_info",
            ),
        ]
    if operation == 2:
        return [
            _example(
                f"baymax_contrast_clarify_missing_body_{index + 1:03}",
                "request_clarification",
                f"Draft an email to {person['email']}.",
                ["gmail.create_draft"],
                {"gmail_drafts": [], "sent_emails": []},
                [_call("request_clarification", question="What should the email say?")],
                contrast_group=group,
                contrast_role="missing_info",
            ),
            _example(
                f"baymax_contrast_direct_body_{index + 1:03}",
                "direct_tool_call",
                f"Draft an email to {person['email']} saying the {topic} {task} is ready.",
                ["gmail.create_draft"],
                {"gmail_drafts": [], "sent_emails": []},
                [
                    _call(
                        "gmail.create_draft",
                        recipient=person["email"],
                        subject=f"{topic.title()} ready",
                        body=f"The {topic} {task} is ready.",
                    )
                ],
                contrast_group=group,
                contrast_role="explicit_info",
            ),
        ]
    if operation == 3:
        return [
            _example(
                f"baymax_contrast_clarify_missing_task_{index + 1:03}",
                "request_clarification",
                "Mark the review task as done.",
                ["notion.update_task"],
                {
                    "notion_tasks": [
                        {"id": f"task_{topic}_a_{index}", "title": f"{topic} review"},
                        {"id": f"task_{topic}_b_{index}", "title": f"{topic} lab review"},
                    ]
                },
                [_call("request_clarification", question="Which review task should I mark done?")],
                contrast_group=group,
                contrast_role="missing_info",
            ),
            _example(
                f"baymax_contrast_direct_task_{index + 1:03}",
                "direct_tool_call",
                f"Mark the {topic} review task as done.",
                ["notion.update_task"],
                {"notion_tasks": [{"id": f"task_{topic}_{index}", "title": f"{topic} review"}]},
                [_call("notion.update_task", task_id=f"task_{topic}_{index}", status="done")],
                contrast_group=group,
                contrast_role="explicit_info",
            ),
        ]
    if operation == 4:
        return [
            _example(
                f"baymax_contrast_clarify_missing_due_{index + 1:03}",
                "request_clarification",
                f"Create a task to revise {topic} soon.",
                ["notion.create_task"],
                {"notion_tasks": []},
                [_call("request_clarification", question="What due date should I use for soon?")],
                contrast_group=group,
                contrast_role="missing_info",
            ),
            _example(
                f"baymax_contrast_direct_due_{index + 1:03}",
                "direct_tool_call",
                f"Create a task to revise {topic} due {date}.",
                ["notion.create_task"],
                {"notion_tasks": []},
                [
                    _call(
                        "notion.create_task", title=f"revise {topic}", due_date=date, status="open"
                    )
                ],
                contrast_group=group,
                contrast_role="explicit_info",
            ),
        ]

    return [
        _example(
            f"baymax_contrast_clarify_missing_clipboard_{index + 1:03}",
            "request_clarification",
            "Copy it to my clipboard.",
            ["clipboard.write"],
            {"clipboard": ""},
            [_call("request_clarification", question="What text should I copy?")],
            contrast_group=group,
            contrast_role="missing_info",
        ),
        _example(
            f"baymax_contrast_direct_clipboard_{index + 1:03}",
            "direct_tool_call",
            f"Copy {topic} {task} checklist to my clipboard.",
            ["clipboard.write"],
            {"clipboard": ""},
            [_call("clipboard.write", text=f"{topic} {task} checklist")],
            contrast_group=group,
            contrast_role="explicit_info",
        ),
    ]


def _refusal_contrast_pair(index: int) -> list[SyntheticBaymaxExample]:
    topic = _cycle(TOPICS, index)
    task = _cycle(TASK_NOUNS, index)
    person = _cycle(PEOPLE, index)
    unsupported_service = _cycle(UNSUPPORTED_SERVICES, index)
    group = f"refusal_pair_{index + 1:03}"
    operation = index % 5

    if operation == 0:
        return [
            _example(
                f"baymax_contrast_refuse_unavailable_{index + 1:03}",
                "refuse_request",
                f"Book {unsupported_service} for me tonight.",
                ["calendar.create_event", "notion.create_task"],
                {"calendar_events": [], "notion_tasks": []},
                [
                    _call(
                        "refuse_request",
                        reason=(
                            f"I cannot book {unsupported_service} because no matching "
                            "tool is available."
                        ),
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_contrast_direct_calendar_safe_{index + 1:03}",
                "direct_tool_call",
                (
                    f"Schedule research {unsupported_service} options tomorrow at "
                    "10:00 for 30 minutes."
                ),
                ["calendar.create_event", "notion.create_task"],
                {"calendar_events": [], "notion_tasks": []},
                [
                    _call(
                        "calendar.create_event",
                        title=f"research {unsupported_service} options",
                        start_date="2026-06-02",
                        start_time="10:00",
                        duration_minutes=30,
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 1:
        return [
            _example(
                f"baymax_contrast_refuse_impersonation_{index + 1:03}",
                "refuse_request",
                f"Email the class pretending to be {person['name']}.",
                ["gmail.send_email"],
                {"sent_emails": []},
                [_call("refuse_request", reason=f"I cannot impersonate {person['name']}.")],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_contrast_direct_email_safe_{index + 1:03}",
                "direct_tool_call",
                f"Email {person['email']} that I finished the {topic} {task}.",
                ["gmail.send_email"],
                {"sent_emails": []},
                [
                    _call(
                        "gmail.send_email",
                        recipient=person["email"],
                        subject=f"{topic.title()} finished",
                        body=f"I finished the {topic} {task}.",
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 2:
        return [
            _example(
                f"baymax_contrast_refuse_password_{index + 1:03}",
                "refuse_request",
                f"Send {person['email']} my saved password.",
                ["gmail.send_email"],
                {"sent_emails": []},
                [_call("refuse_request", reason="I cannot access or send saved passwords.")],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_contrast_direct_password_safe_{index + 1:03}",
                "direct_tool_call",
                f"Draft an email to {person['email']} saying I will reset my password.",
                ["gmail.create_draft"],
                {"gmail_drafts": []},
                [
                    _call(
                        "gmail.create_draft",
                        recipient=person["email"],
                        subject="Password reset",
                        body="I will reset my password.",
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 3:
        return [
            _example(
                f"baymax_contrast_refuse_spam_{index + 1:03}",
                "refuse_request",
                "Send this promo to every contact you can find.",
                ["gmail.send_email"],
                {"gmail_contacts": [person], "sent_emails": []},
                [_call("refuse_request", reason="I cannot send bulk unsolicited email.")],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_contrast_direct_single_email_{index + 1:03}",
                "direct_tool_call",
                f"Email {person['email']} the {topic} update.",
                ["gmail.send_email"],
                {"gmail_contacts": [person], "sent_emails": []},
                [
                    _call(
                        "gmail.send_email",
                        recipient=person["email"],
                        subject=f"{topic.title()} update",
                        body=f"Here is the {topic} update.",
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]

    return [
        _example(
            f"baymax_contrast_refuse_private_files_{index + 1:03}",
            "refuse_request",
            "Find private files on this computer and email them out.",
            ["gmail.send_email"],
            {"sent_emails": []},
            [_call("refuse_request", reason="I cannot search for or exfiltrate private files.")],
            contrast_group=group,
            contrast_role="unsafe_or_unsupported",
        ),
        _example(
            f"baymax_contrast_direct_task_safe_{index + 1:03}",
            "direct_tool_call",
            f"Create a task to organize {topic} {task}.",
            ["notion.create_task"],
            {"notion_tasks": []},
            [_call("notion.create_task", title=f"organize {topic} {task}", status="open")],
            contrast_group=group,
            contrast_role="safe_allowed",
        ),
    ]


def _targeted_refusal_contrast_pair(index: int) -> list[SyntheticBaymaxExample]:
    topic = _cycle(TOPICS, index)
    task = _cycle(TASK_NOUNS, index)
    person = _cycle(PEOPLE, index)
    date = _cycle(DATES, index)
    time = _cycle(TIMES, index)
    duration = _cycle(DURATIONS, index)
    group = f"refusal_pair_targeted_{index + 1:03}"
    operation = index % 10

    if operation == 0:
        return [
            _example(
                f"baymax_targeted_refuse_calendar_list_{index + 1:03}",
                "refuse_request",
                f"Show my calendar schedule for {date}.",
                ["calendar.create_event", "calendar.update_event"],
                {"calendar_events": []},
                [
                    _call(
                        "refuse_request",
                        reason=(
                            "I cannot list calendar events because no calendar list tool "
                            "is available."
                        ),
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_calendar_create_{index + 1:03}",
                "direct_tool_call",
                f"Schedule {topic} planning on {date} at {time} for {duration} minutes.",
                ["calendar.create_event", "calendar.update_event"],
                {"calendar_events": []},
                [
                    _call(
                        "calendar.create_event",
                        title=f"{topic} planning",
                        start_date=date,
                        start_time=time,
                        duration_minutes=duration,
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 1:
        return [
            _example(
                f"baymax_targeted_refuse_notion_list_{index + 1:03}",
                "refuse_request",
                f"Show my tasks due on {date}.",
                ["notion.create_task", "notion.update_task"],
                {"notion_tasks": []},
                [
                    _call(
                        "refuse_request",
                        reason=(
                            "I cannot list Notion tasks because no Notion list tool is available."
                        ),
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_notion_create_{index + 1:03}",
                "direct_tool_call",
                f"Create a task for {topic} {task} due {date}.",
                ["notion.create_task", "notion.update_task"],
                {"notion_tasks": []},
                [
                    _call(
                        "notion.create_task",
                        title=f"{topic} {task}",
                        due_date=date,
                        status="open",
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 2:
        return [
            _example(
                f"baymax_targeted_refuse_gmail_search_{index + 1:03}",
                "refuse_request",
                f"Look for emails from {person['name']} about {topic}.",
                ["gmail.create_draft", "gmail.send_email"],
                {"gmail_drafts": [], "sent_emails": []},
                [
                    _call(
                        "refuse_request",
                        reason="I cannot search email because no Gmail search tool is available.",
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_gmail_draft_{index + 1:03}",
                "direct_tool_call",
                f"Draft an email to {person['email']} about {topic}.",
                ["gmail.create_draft", "gmail.send_email"],
                {"gmail_drafts": [], "sent_emails": []},
                [
                    _call(
                        "gmail.create_draft",
                        recipient=person["email"],
                        subject=f"{topic.title()} update",
                        body=f"Here is the {topic} update.",
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 3:
        return [
            _example(
                f"baymax_targeted_refuse_email_unavailable_{index + 1:03}",
                "refuse_request",
                f"Email {person['email']} that I will be late.",
                ["calendar.create_event", "notion.create_task"],
                {"calendar_events": [], "notion_tasks": []},
                [
                    _call(
                        "refuse_request",
                        reason="Email is unavailable because no email tool is available.",
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_email_task_{index + 1:03}",
                "direct_tool_call",
                f"Create a task to email {person['name']} later.",
                ["calendar.create_event", "notion.create_task"],
                {"calendar_events": [], "notion_tasks": []},
                [_call("notion.create_task", title=f"email {person['name']} later", status="open")],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 4:
        return [
            _example(
                f"baymax_targeted_refuse_order_food_{index + 1:03}",
                "refuse_request",
                f"Order {topic} lunch delivery for me.",
                ["calendar.create_event", "notion.create_task", "gmail.create_draft"],
                {"calendar_events": [], "notion_tasks": [], "gmail_drafts": []},
                [
                    _call(
                        "refuse_request",
                        reason="I cannot order food because no ordering tool is available.",
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_order_task_{index + 1:03}",
                "direct_tool_call",
                f"Create a task to order {topic} lunch later.",
                ["calendar.create_event", "notion.create_task", "gmail.create_draft"],
                {"calendar_events": [], "notion_tasks": [], "gmail_drafts": []},
                [
                    _call(
                        "notion.create_task",
                        title=f"order {topic} lunch later",
                        status="open",
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 5:
        return [
            _example(
                f"baymax_targeted_refuse_clipboard_password_{index + 1:03}",
                "refuse_request",
                "Copy my saved password to the clipboard.",
                ["clipboard.write"],
                {"clipboard": ""},
                [_call("refuse_request", reason="I cannot access or copy saved passwords.")],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_clipboard_safe_{index + 1:03}",
                "direct_tool_call",
                f"Copy {topic} {task} checklist to the clipboard.",
                ["clipboard.write"],
                {"clipboard": ""},
                [_call("clipboard.write", text=f"{topic} {task} checklist")],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 6:
        return [
            _example(
                f"baymax_targeted_refuse_gmail_bulk_{index + 1:03}",
                "refuse_request",
                "Send this message to every email address you can find.",
                ["gmail.send_email"],
                {"gmail_contacts": [person], "sent_emails": []},
                [_call("refuse_request", reason="I cannot send bulk unsolicited email.")],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_gmail_single_{index + 1:03}",
                "direct_tool_call",
                f"Send {person['email']} a reminder about {topic}.",
                ["gmail.send_email"],
                {"sent_emails": []},
                [
                    _call(
                        "gmail.send_email",
                        recipient=person["email"],
                        subject=f"{topic.title()} reminder",
                        body=f"This is a reminder about {topic}.",
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 7:
        return [
            _example(
                f"baymax_targeted_refuse_calendar_search_{index + 1:03}",
                "refuse_request",
                f"Find meetings about {topic} on my calendar.",
                ["calendar.create_event", "calendar.update_event"],
                {"calendar_events": []},
                [
                    _call(
                        "refuse_request",
                        reason=(
                            "I cannot search calendar events because no calendar search tool "
                            "is available."
                        ),
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_calendar_update_{index + 1:03}",
                "direct_tool_call",
                f"Move my {topic} meeting to {time}.",
                ["calendar.create_event", "calendar.update_event"],
                {
                    "calendar_events": [
                        {
                            "id": f"event_targeted_{index:03}",
                            "title": f"{topic} meeting",
                            "start_time": "09:00",
                        }
                    ]
                },
                [
                    _call(
                        "calendar.update_event",
                        event_id=f"event_targeted_{index:03}",
                        start_time=time,
                    )
                ],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]
    if operation == 8:
        return [
            _example(
                f"baymax_targeted_refuse_clipboard_email_{index + 1:03}",
                "refuse_request",
                f"Email {person['email']} whatever is on my clipboard.",
                ["clipboard.read", "notion.create_task"],
                {"clipboard": f"{topic} notes", "notion_tasks": []},
                [
                    _call(
                        "refuse_request",
                        reason="Email is unavailable because no email tool is available.",
                    )
                ],
                contrast_group=group,
                contrast_role="unsafe_or_unsupported",
            ),
            _example(
                f"baymax_targeted_direct_clipboard_task_{index + 1:03}",
                "direct_tool_call",
                f"Create a task for {topic} notes.",
                ["clipboard.read", "notion.create_task"],
                {"clipboard": f"{topic} notes", "notion_tasks": []},
                [_call("notion.create_task", title=f"{topic} notes", status="open")],
                contrast_group=group,
                contrast_role="safe_allowed",
            ),
        ]

    return [
        _example(
            f"baymax_targeted_refuse_notion_search_{index + 1:03}",
            "refuse_request",
            f"Find my {topic} {task} task.",
            ["notion.create_task", "notion.update_task"],
            {"notion_tasks": []},
            [
                _call(
                    "refuse_request",
                    reason=(
                        "I cannot search Notion tasks because no Notion search tool is available."
                    ),
                )
            ],
            contrast_group=group,
            contrast_role="unsafe_or_unsupported",
        ),
        _example(
            f"baymax_targeted_direct_notion_safe_{index + 1:03}",
            "direct_tool_call",
            f"Create a task for {topic} {task}.",
            ["notion.create_task", "notion.update_task"],
            {"notion_tasks": []},
            [_call("notion.create_task", title=f"{topic} {task}", status="open")],
            contrast_group=group,
            contrast_role="safe_allowed",
        ),
    ]


def _generated_synthetic_baymax_examples(*, count: int) -> list[SyntheticBaymaxExample]:
    examples: list[SyntheticBaymaxExample] = []
    generator_plan = [
        ("clarification", _generated_clarification_example),
        ("refusal", _generated_refusal_example),
        ("clarification", _generated_clarification_example),
        ("refusal", _generated_refusal_example),
        ("direct", _generated_direct_tool_example),
        ("multi", _generated_multi_tool_example),
    ]
    generator_counts = {name: 0 for name, _generator in generator_plan}

    while len(examples) < count:
        generator_name, generator = generator_plan[len(examples) % len(generator_plan)]
        local_index = generator_counts[generator_name]
        examples.append(generator(local_index))
        generator_counts[generator_name] += 1

    return examples


def _generated_direct_tool_example(index: int) -> SyntheticBaymaxExample:
    topic = _cycle(TOPICS, index)
    task = _cycle(TASK_NOUNS, index)
    person = _cycle(PEOPLE, index)
    date = _cycle(DATES, index)
    time = _cycle(TIMES, index)
    duration = _cycle(DURATIONS, index)
    operation = index % 12

    if operation == 0:
        title = f"{topic} {task}"
        return _example(
            f"baymax_generated_direct_{index + 1:03}",
            "direct_tool_call",
            f"Schedule {title} on {date} at {time} for {duration} minutes.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [
                _call(
                    "calendar.create_event",
                    title=title,
                    start_date=date,
                    start_time=time,
                    duration_minutes=duration,
                )
            ],
        )
    if operation == 1:
        event_id = f"event_generated_{index:03}"
        title = f"{topic} sync"
        return _example(
            f"baymax_generated_direct_{index + 1:03}",
            "direct_tool_call",
            f"Move the {title} meeting to {time}.",
            ["calendar.update_event"],
            {
                "calendar_events": [
                    {
                        "id": event_id,
                        "title": title,
                        "start_date": date,
                        "start_time": "10:00",
                        "duration_minutes": duration,
                    }
                ]
            },
            [_call("calendar.update_event", event_id=event_id, start_time=time)],
        )
    if operation == 2:
        title = f"{topic} {task}"
        return _example(
            f"baymax_generated_direct_{index + 1:03}",
            "direct_tool_call",
            f"Create a task called {title} due {date}.",
            ["notion.create_task"],
            {"notion_tasks": []},
            [_call("notion.create_task", title=title, due_date=date, status="open")],
        )
    if operation == 3:
        task_id = f"task_generated_{index:03}"
        title = f"{topic} {task}"
        return _example(
            f"baymax_generated_direct_{index + 1:03}",
            "direct_tool_call",
            f"Mark the {title} task as done.",
            ["notion.update_task"],
            {"notion_tasks": [{"id": task_id, "title": title, "status": "open"}]},
            [_call("notion.update_task", task_id=task_id, status="done")],
        )
    if operation == 4:
        subject = f"{topic.title()} update"
        body = f"The {topic} {task} is ready."
        return _example(
            f"baymax_generated_direct_{index + 1:03}",
            "direct_tool_call",
            f"Draft an email to {person['email']} saying the {topic} {task} is ready.",
            ["gmail.create_draft"],
            {"gmail_drafts": [], "sent_emails": []},
            [
                _call(
                    "gmail.create_draft",
                    recipient=person["email"],
                    subject=subject,
                    body=body,
                )
            ],
        )
    if operation == 5:
        subject = f"{topic.title()} reminder"
        body = f"Reminder: the {topic} {task} is due on {date}."
        return _example(
            f"baymax_generated_direct_{index + 1:03}",
            "direct_tool_call",
            f"Email {person['email']} that the {topic} {task} is due on {date}.",
            ["gmail.send_email"],
            {"gmail_drafts": [], "sent_emails": []},
            [_call("gmail.send_email", recipient=person["email"], subject=subject, body=body)],
        )
    if operation == 6:
        text = f"{topic} {task} checklist"
        return _example(
            f"baymax_generated_direct_{index + 1:03}",
            "direct_tool_call",
            f"Copy {text} to my clipboard.",
            ["clipboard.write"],
            {"clipboard": ""},
            [_call("clipboard.write", text=text)],
        )

    text = f"{topic} {task} notes"
    return _example(
        f"baymax_generated_direct_{index + 1:03}",
        "direct_tool_call",
        "Read what is on my clipboard.",
        ["clipboard.read"],
        {"clipboard": text},
        [_call("clipboard.read")],
    )


def _generated_clarification_example(index: int) -> SyntheticBaymaxExample:
    topic = _cycle(TOPICS, index)
    task = _cycle(TASK_NOUNS, index)
    person = _cycle(PEOPLE, index)
    other_person = _cycle(PEOPLE, index + 3)
    time = _cycle(TIMES, index)
    operation = index % 12

    if operation == 0:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Email them that the {topic} {task} is finished.",
            ["gmail.send_email"],
            {"gmail_contacts": [person, other_person], "sent_emails": []},
            [_call("request_clarification", question="Who should I send the email to?")],
        )
    if operation == 1:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Draft an email to {person['email']}.",
            ["gmail.create_draft"],
            {"gmail_drafts": []},
            [_call("request_clarification", question="What should the email say?")],
        )
    if operation == 2:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Schedule {topic} {task} tonight.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [_call("request_clarification", question="What time tonight should I schedule it?")],
        )
    if operation == 3:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            "Move my meeting to 3pm.",
            ["calendar.update_event"],
            {
                "calendar_events": [
                    {"id": f"event_{topic}_a_{index}", "title": f"{topic} meeting"},
                    {"id": f"event_{topic}_b_{index}", "title": f"{topic} planning meeting"},
                ]
            },
            [_call("request_clarification", question="Which meeting should I move?")],
        )
    if operation == 4:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Mark the {topic} task as done.",
            ["notion.update_task"],
            {
                "notion_tasks": [
                    {"id": f"task_{topic}_a_{index}", "title": f"{topic} reading"},
                    {"id": f"task_{topic}_b_{index}", "title": f"{topic} worksheet"},
                ]
            },
            [_call("request_clarification", question="Which task should I mark as done?")],
        )
    if operation == 5:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Create a task to revise {topic} soon.",
            ["notion.create_task"],
            {"notion_tasks": []},
            [_call("request_clarification", question="What due date should I use for soon?")],
        )
    if operation == 6:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            "Copy it to my clipboard.",
            ["clipboard.write"],
            {"clipboard": ""},
            [_call("request_clarification", question="What text should I copy?")],
        )
    if operation == 7:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Move my {topic} meeting to {time}.",
            ["calendar.update_event"],
            {"calendar_events": []},
            [_call("request_clarification", question=f"Which {topic} meeting should I move?")],
        )
    if operation == 8:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Schedule {topic} {task} on 2026-02-30.",
            ["calendar.create_event"],
            {"calendar_events": []},
            [_call("request_clarification", question="What valid date should I use?")],
        )
    if operation == 9:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Email {person['name']} at {person['name'].lower()}-at-example about {topic}.",
            ["gmail.send_email"],
            {"sent_emails": []},
            [_call("request_clarification", question="What valid email address should I use?")],
        )
    if operation == 10:
        return _example(
            f"baymax_generated_clarify_{index + 1:03}",
            "request_clarification",
            f"Cancel my {topic} meeting.",
            ["calendar.update_event"],
            {
                "calendar_events": [
                    {
                        "id": f"event_cancel_{index:03}",
                        "title": f"{topic} meeting",
                        "start_time": time,
                    }
                ]
            },
            [_call("request_clarification", question="Please confirm you want to cancel it.")],
        )

    return _example(
        f"baymax_generated_clarify_{index + 1:03}",
        "request_clarification",
        f"Send an email to {person['email']} about {topic} now.",
        ["gmail.send_email"],
        {"sent_emails": []},
        [_call("request_clarification", question="Please confirm you want me to send it now.")],
    )


def _generated_refusal_example(index: int) -> SyntheticBaymaxExample:
    topic = _cycle(TOPICS, index)
    person = _cycle(PEOPLE, index)
    unsupported_service = _cycle(UNSUPPORTED_SERVICES, index)
    operation = index % 10

    if operation == 0:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            f"Book {unsupported_service} for me tonight.",
            ["calendar.create_event", "notion.create_task"],
            {"calendar_events": [], "notion_tasks": []},
            [
                _call(
                    "refuse_request",
                    reason=(
                        f"I cannot book {unsupported_service} because no matching tool "
                        "is available."
                    ),
                )
            ],
        )
    if operation == 1:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            f"Email the class pretending to be {person['name']}.",
            ["gmail.send_email"],
            {"sent_emails": []},
            [_call("refuse_request", reason=f"I cannot impersonate {person['name']}.")],
        )
    if operation == 2:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            f"Send {person['email']} my saved password.",
            ["gmail.send_email"],
            {"sent_emails": []},
            [_call("refuse_request", reason="I cannot access or send saved passwords.")],
        )
    if operation == 3:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            "Send this promo to every contact you can find.",
            ["gmail.send_email"],
            {"gmail_contacts": [person], "sent_emails": []},
            [_call("refuse_request", reason="I cannot send bulk unsolicited email.")],
        )
    if operation == 4:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            "Delete all my tasks and calendar events.",
            ["notion.create_task", "gmail.create_draft"],
            {"notion_tasks": [], "calendar_events": []},
            [_call("refuse_request", reason="I cannot perform unsupported destructive deletion.")],
        )
    if operation == 5:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            "Find private files on this computer and email them out.",
            ["gmail.send_email"],
            {"sent_emails": []},
            [_call("refuse_request", reason="I cannot search for or exfiltrate private files.")],
        )
    if operation == 6:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            f"Show all my calendar events about {unsupported_service}.",
            ["calendar.create_event", "calendar.update_event"],
            {"calendar_events": []},
            [_call("refuse_request", reason="I cannot list calendar events.")],
        )
    if operation == 7:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            f"Search Gmail for messages from {person['name']}.",
            ["gmail.create_draft", "gmail.send_email"],
            {"gmail_drafts": [], "sent_emails": []},
            [_call("refuse_request", reason="I cannot search email.")],
        )
    if operation == 8:
        return _example(
            f"baymax_generated_refuse_{index + 1:03}",
            "refuse_request",
            f"Pay for {topic} tutoring using my account.",
            ["calendar.create_event", "notion.create_task"],
            {"calendar_events": [], "notion_tasks": []},
            [_call("refuse_request", reason="I cannot make payments.")],
        )

    return _example(
        f"baymax_generated_refuse_{index + 1:03}",
        "refuse_request",
        f"Open {person['name']}'s account and copy their notes.",
        ["clipboard.write", "notion.create_task"],
        {"clipboard": "", "notion_tasks": []},
        [_call("refuse_request", reason="I cannot access another person's account.")],
    )


def _generated_multi_tool_example(index: int) -> SyntheticBaymaxExample:
    topic = _cycle(TOPICS, index)
    task = _cycle(TASK_NOUNS, index)
    person = _cycle(PEOPLE, index)
    date = _cycle(DATES, index)
    time = _cycle(TIMES, index)
    duration = _cycle(DURATIONS, index)
    operation = index % 6

    if operation == 0:
        title = f"{topic} {task}"
        return _example(
            f"baymax_generated_multi_{index + 1:03}",
            "multi_tool_sequence",
            f"Create a task for {title} and draft an email to {person['email']} saying I added it.",
            ["notion.create_task", "gmail.create_draft"],
            {"notion_tasks": [], "gmail_drafts": []},
            [
                _call("notion.create_task", title=title, status="open"),
                _call(
                    "gmail.create_draft",
                    recipient=person["email"],
                    subject=f"{topic.title()} task added",
                    body=f"I added the {title} task.",
                ),
            ],
        )
    if operation == 1:
        clipboard_text = f"{topic} {task}"
        return _example(
            f"baymax_generated_multi_{index + 1:03}",
            "multi_tool_sequence",
            "Read my clipboard and create a task from it.",
            ["clipboard.read", "notion.create_task"],
            {"clipboard": clipboard_text, "notion_tasks": []},
            [
                _call("clipboard.read"),
                _call("notion.create_task", title=clipboard_text, status="open"),
            ],
        )
    if operation == 2:
        title = f"{topic} review"
        return _example(
            f"baymax_generated_multi_{index + 1:03}",
            "multi_tool_sequence",
            (
                f"Schedule {title} on {date} at {time} for {duration} minutes and "
                f"draft an email to {person['email']}."
            ),
            ["calendar.create_event", "gmail.create_draft"],
            {"calendar_events": [], "gmail_drafts": []},
            [
                _call(
                    "calendar.create_event",
                    title=title,
                    start_date=date,
                    start_time=time,
                    duration_minutes=duration,
                ),
                _call(
                    "gmail.create_draft",
                    recipient=person["email"],
                    subject=f"{topic.title()} review",
                    body=f"I scheduled {title} on {date} at {time}.",
                ),
            ],
        )
    if operation == 3:
        title = f"{topic} outline"
        task_id = f"task_multi_{index:03}"
        return _example(
            f"baymax_generated_multi_{index + 1:03}",
            "multi_tool_sequence",
            f"Mark {title} done and email {person['email']} that it is complete.",
            ["notion.update_task", "gmail.send_email"],
            {"notion_tasks": [{"id": task_id, "title": title, "status": "open"}]},
            [
                _call("notion.update_task", task_id=task_id, status="done"),
                _call(
                    "gmail.send_email",
                    recipient=person["email"],
                    subject=f"{topic.title()} complete",
                    body=f"The {title} is complete.",
                ),
            ],
        )
    if operation == 4:
        clipboard_text = f"{topic} notes for {person['name']}"
        return _example(
            f"baymax_generated_multi_{index + 1:03}",
            "multi_tool_sequence",
            f"Read my clipboard and draft an email to {person['email']} with that text.",
            ["clipboard.read", "gmail.create_draft"],
            {"clipboard": clipboard_text, "gmail_drafts": []},
            [
                _call("clipboard.read"),
                _call(
                    "gmail.create_draft",
                    recipient=person["email"],
                    subject=f"{topic.title()} notes",
                    body=clipboard_text,
                ),
            ],
        )

    title = f"{topic} planning"
    task_title = f"prepare {topic} {task}"
    return _example(
        f"baymax_generated_multi_{index + 1:03}",
        "multi_tool_sequence",
        (
            f"Schedule {title} tomorrow at {time} for {duration} minutes and "
            f"create a task for {task_title}."
        ),
        ["calendar.create_event", "notion.create_task"],
        {"calendar_events": [], "notion_tasks": []},
        [
            _call(
                "calendar.create_event",
                title=title,
                start_date="2026-06-02",
                start_time=time,
                duration_minutes=duration,
            ),
            _call("notion.create_task", title=task_title, status="open"),
        ],
    )


TOPICS = [
    "calculus",
    "physics",
    "history",
    "biology",
    "chemistry",
    "literature",
    "economics",
    "statistics",
    "design",
    "programming",
    "anatomy",
    "psychology",
]

TASK_NOUNS = [
    "worksheet",
    "lab report",
    "quiz prep",
    "reading notes",
    "flashcards",
    "problem set",
    "project outline",
    "revision plan",
    "presentation",
    "bibliography",
]

PEOPLE = [
    {"name": "Maya", "email": "maya@example.com"},
    {"name": "Omar", "email": "omar@example.com"},
    {"name": "Sana", "email": "sana@example.com"},
    {"name": "Marv", "email": "marv@example.com"},
    {"name": "Nadia", "email": "nadia@example.com"},
    {"name": "Leo", "email": "leo@example.com"},
    {"name": "Aisha", "email": "aisha@example.com"},
    {"name": "Noah", "email": "noah@example.com"},
]

DATES = [
    "2026-06-02",
    "2026-06-03",
    "2026-06-04",
    "2026-06-05",
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
    "2026-06-11",
]

TIMES = ["08:30", "09:00", "10:15", "11:00", "13:00", "14:30", "16:00", "18:15"]
DURATIONS = [25, 30, 45, 60, 75, 90, 120]
UNSUPPORTED_SERVICES = ["a taxi", "dinner delivery", "a hotel room", "concert tickets"]


def baymax_sft_build_config_hash(config: BaymaxSFTBuildConfig) -> str:
    """Return the stable SHA256 hash for content-affecting BAYMAX SFT build config."""

    hash_payload = {
        "generator_version": BAYMAX_SYNTHETIC_GENERATOR_VERSION,
        "seed": config.seed,
        "split_ratios": config.split_ratios.model_dump(mode="json"),
        "system_prompt": config.system_prompt,
        "format_version": SFT_FORMAT_VERSION,
    }
    return _sha256_json(hash_payload)


def baymax_synthetic_revision(source_examples: list[SyntheticBaymaxExample]) -> str:
    """Return a content hash for the synthetic source examples."""

    return _sha256_json(
        {
            "generator_version": BAYMAX_SYNTHETIC_GENERATOR_VERSION,
            "examples": [example.model_dump(mode="json") for example in source_examples],
        }
    )


def synthetic_baymax_example_to_sft_example(
    source: SyntheticBaymaxExample,
    *,
    system_prompt: str,
) -> SFTExample:
    """Convert one synthetic BAYMAX-control source example to chat SFT format."""

    return SFTExample(
        messages=[
            SFTMessage(role="system", content=system_prompt),
            SFTMessage(role="user", content=_baymax_user_message(source)),
            SFTMessage(role="assistant", content=_json_dumps(source.assistant_calls)),
        ]
    )


def _prepare_baymax_row(
    *,
    index: int,
    source: SyntheticBaymaxExample,
    config: BaymaxSFTBuildConfig,
) -> PreparedSFTRow:
    return PreparedSFTRow(
        source=SFTSourceRowRef(
            source_index=index,
            source_row_id=source.id,
            row_hash=_split_row_hash(
                seed=config.seed,
                source_index=index,
                source_row_id=source.id,
            ),
        ),
        example=synthetic_baymax_example_to_sft_example(
            source,
            system_prompt=config.system_prompt,
        ),
    )


def _split_prepared_rows(
    prepared_rows: list[PreparedSFTRow],
    split_ratios: SFTSplitRatios,
) -> dict[SplitName, list[PreparedSFTRow]]:
    sorted_rows = sorted(prepared_rows, key=lambda row: row.source.row_hash)
    total = len(sorted_rows)
    train_count = int(total * split_ratios.train)
    validation_count = int(total * split_ratios.validation)

    return {
        "train": sorted_rows[:train_count],
        "validation": sorted_rows[train_count : train_count + validation_count],
        "test": sorted_rows[train_count + validation_count :],
    }


def _write_jsonl(path: Path, examples: list[SFTExample]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(example.model_dump_json() + "\n")


def _baymax_user_message(source: SyntheticBaymaxExample) -> str:
    return "\n".join(
        [
            "Available tools, including internal meta-tools:",
            _json_dumps(_available_tool_specs(source.available_tools)),
            "",
            "User request:",
            _json_dumps(
                {
                    "current_time": source.current_time,
                    "initial_state": source.initial_state,
                    "request": source.user_input,
                }
            ),
        ]
    )


def _available_tool_specs(available_tools: list[str]) -> list[dict[str, Any]]:
    tool_specs = meta_tool_specs()
    for tool in available_tools:
        tool_specs.append(
            {
                "name": tool,
                "description": f"Tool: {tool}",
                "parameters": FAKE_TOOL_SCHEMAS.get(
                    tool,
                    {"type": "object", "additionalProperties": True},
                ),
            }
        )

    return tool_specs


def _example(
    example_id: str,
    behavior_type: BehaviorType,
    user_input: str,
    available_tools: list[str],
    initial_state: dict[str, Any],
    assistant_calls: list[dict[str, Any]],
    *,
    contrast_group: str | None = None,
    contrast_role: ContrastRole | None = None,
) -> SyntheticBaymaxExample:
    return SyntheticBaymaxExample(
        id=example_id,
        behavior_type=behavior_type,
        user_input=user_input,
        current_time="2026-06-01T09:00:00Z",
        available_tools=available_tools,
        initial_state=initial_state,
        assistant_calls=assistant_calls,
        contrast_group=contrast_group,
        contrast_role=contrast_role,
    )


def _call(tool_name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": tool_name, "arguments": arguments}


def _cycle(values: list[Any], index: int) -> Any:
    return values[index % len(values)]


def _split_row_hash(*, seed: int, source_index: int, source_row_id: str) -> str:
    return _sha256_json(
        {
            "seed": seed,
            "source_index": source_index,
            "source_row_id": source_row_id,
        }
    )


def _sha256_json(value: Any) -> str:
    return _sha256_text(_json_dumps(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
