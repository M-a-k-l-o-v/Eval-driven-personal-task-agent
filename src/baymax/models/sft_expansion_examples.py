"""Independent, training-only additions for the fixed-proportion 2k experiment."""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any, Literal

from baymax.models.baymax_sft import SyntheticBaymaxExample, validate_synthetic_baymax_example

EXPANSION_VERSION = "coverage-expansion-v1"
FAMILIES = (
    "clipboard-task",
    "calendar-create",
    "calendar-select",
    "task-deadline",
    "task-select",
    "routine-email",
    "clipboard-draft",
    "draft-recipient",
)
TASKS = (
    "Inspect the kiln gasket",
    "Photograph the casting molds",
    "Check the humidity sensor",
    "Record the test-bench readings",
    "Label the storage drawers",
    "Reconcile the workshop inventory",
    "Clean the optical mounts",
    "Verify the pressure gauge",
    "Archive the maintenance log",
    "Count the packing inserts",
    "Check the cable shielding",
    "Replace the sample sleeves",
)
AREAS = (
    "the west annex",
    "the north store",
    "the upper workshop",
    "the east depot",
    "the receiving bay",
    "the ceramics room",
    "the assembly room",
    "the materials lab",
    "the instrument room",
)
NAMES = ("Inez", "Tomas", "Leena", "Omar", "Petra", "Soren", "Daria", "Emil")


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


def _group(index: int, rng: random.Random) -> list[SyntheticBaymaxExample]:
    family = index % len(FAMILIES)
    variant = (index // len(FAMILIES)) % 6
    title = f"{TASKS[index % len(TASKS)]} for {AREAS[index // len(TASKS)]}"
    day = (date(2026, 11, 3) + timedelta(days=rng.randrange(25))).isoformat()
    time = rng.choice(("08:20", "09:40", "11:15", "13:25", "15:10", "17:35"))
    minutes = rng.choice((20, 35, 50, 65))
    person = rng.choice(NAMES)
    email = f"{person.lower()}.workshop@example.org"
    text = f"Checklist update: {title} is ready for review."
    entity_id = f"item_{rng.randrange(100000, 999999)}"
    other_id = f"other_{index}"
    group = f"{EXPANSION_VERSION}:{FAMILIES[family]}:{index:03}"

    def row(
        role: Literal["act-1", "act-2", "clarify", "refuse"],
        request: str,
        tools: list[str],
        state: dict[str, Any],
        calls: list[dict[str, Any]],
    ) -> SyntheticBaymaxExample:
        return SyntheticBaymaxExample(
            id=f"{group}:{role}",
            behavior_type=(
                "request_clarification"
                if role == "clarify"
                else "refuse_request"
                if role == "refuse"
                else "multi_tool_sequence"
                if len(calls) > 1
                else "direct_tool_call"
            ),
            user_input=request,
            current_time="2026-11-02T07:00:00Z",
            available_tools=tools,
            initial_state=state,
            assistant_calls=calls,
            contrast_group=group,
            contrast_role="clarify"
            if role == "clarify"
            else "refuse"
            if role == "refuse"
            else "act",
        )

    def clarify(request: str, tools: list[str], state: dict[str, Any], question: str):
        return row(
            "clarify", request, tools, state, [_call("request_clarification", question=question)]
        )

    def refuse(request: str, tools: list[str], state: dict[str, Any], reason: str):
        return row("refuse", request, tools, state, [_call("refuse_request", reason=reason)])

    if family in (0, 6):
        task = family == 0
        tool = "notion.create_task" if task else "gmail.create_draft"
        tools = ["clipboard.read", tool, "clipboard.write"]
        request = (
            (
                "Read my clipboard and use that text as a new task title.",
                "Make a task whose title is exactly what you read from my clipboard.",
                "Check the clipboard, then turn its contents into a task.",
                "Use the copied text for a task title after reading the clipboard.",
                "Add the current clipboard text to my task list; read it first.",
                "Read what I copied and create a task with that exact title.",
            )[variant]
            if task
            else (
                f"Read my clipboard, then draft its text to {email} without sending.",
                f"Prepare a draft to {email} using exactly the body you read from the clipboard.",
                f"Check what I copied and put it in an email draft for {email}.",
                f"Use the current clipboard contents as a draft body for {email}; read them first.",
                f"Read the copied text and draft it verbatim to {email}.",
                f"Make an unsent draft for {email} from what you read on my clipboard.",
            )[variant]
        )
        content = title if task else text
        state = {
            "clipboard": {"text": content},
            "notion_tasks": [
                {"id": other_id, "title": f"Review the notes for {title}", "status": "open"}
            ],
            "gmail_drafts": [],
        }
        args = {"title": content} if task else {"recipient": email, "body": content}
        first = row("act-1", request, tools, state, [_call("clipboard.read"), _call(tool, **args)])
        second = (
            row(
                "act-2",
                f'Write this exact text to the clipboard: "{content}".',
                tools,
                {"clipboard": {"text": "Previous inspection note"}},
                [_call("clipboard.write", text=content)],
            )
            if task
            else row(
                "act-2",
                f'Draft this exact message to {email}: "{text}".',
                tools,
                {"gmail_drafts": []},
                [_call(tool, recipient=email, body=text)],
            )
        )
        missing = {**state, "clipboard": {"text": ""}}
        return [
            first,
            second,
            clarify(request, tools, missing, "The clipboard is empty. What text should I use?"),
            refuse(
                request,
                ["clipboard.write"],
                state,
                "I cannot read the clipboard or perform the requested follow-up with these tools.",
            ),
        ]

    if family == 1:
        tools = ["calendar.create_event", "clipboard.read"]
        request = (
            f'Create a {minutes}-minute calendar entry named "{title}" on {day} at {time}.',
            f'On {day}, reserve {minutes} minutes from {time} for "{title}".',
            f'Put "{title}" on my calendar: {day}, {time}, duration {minutes} minutes.',
            f'At {time} on {day}, schedule "{title}" for {minutes} minutes.',
            f'Calendar booking: "{title}", {day} at {time}, lasting {minutes} minutes.',
            f'Please schedule "{title}" starting {day} {time}, for {minutes} minutes.',
        )[variant]
        arguments = dict(title=title, start_date=day, start_time=time, duration_minutes=minutes)
        missing_request, question = (
            (
                f'Schedule "{title}" on {day} for {minutes} minutes.',
                "What start time should I use?",
            ),
            (f'Schedule "{title}" on {day} at {time}.', "How many minutes should it last?"),
            (f'Schedule "{title}" on {day}.', "What start time and duration should I use?"),
        )[variant % 3]
        return [
            row("act-1", request, tools, {}, [_call("calendar.create_event", **arguments)]),
            row(
                "act-2",
                request + " Use a 25-minute duration instead.",
                tools,
                {},
                [_call("calendar.create_event", **{**arguments, "duration_minutes": 25})],
            ),
            clarify(missing_request, tools, {}, question),
            refuse(
                request,
                ["clipboard.read"],
                {},
                "Calendar creation is unavailable with these tools.",
            ),
        ]

    if family in (2, 4):
        calendar = family == 2
        tool = "calendar.update_event" if calendar else "notion.update_task"
        key = "calendar_events" if calendar else "notion_tasks"
        argument = "event_id" if calendar else "task_id"
        tools = [tool, "clipboard.read"]
        entity = {
            "id": entity_id,
            "title": title,
            **(
                {"start_date": day, "start_time": "07:30", "duration_minutes": minutes}
                if calendar
                else {"status": "open"}
            ),
        }
        other = {**entity, "id": other_id, "title": "Inspect the spare label printer"}
        if calendar:
            other["start_date"] = (date.fromisoformat(day) + timedelta(days=7)).isoformat()
        request = (
            (
                f'Move "{title}" to {time}, keeping its date and duration.',
                f'Change only the start time of "{title}" to {time}.',
                f'For "{title}", use {time} as the new start time.',
            )[variant % 3]
            if calendar
            else (
                f'Mark "{title}" done.',
                f'The task "{title}" is finished; mark it done.',
                f'Update "{title}" to done without changing its other fields.',
            )[variant % 3]
        )
        update = {"start_time": time} if calendar else {"status": "done"}
        alternate = {**entity, "id": f"alternate_{entity_id}"}
        ambiguous = {**other, "title": title}
        return [
            row(
                "act-1",
                request,
                tools,
                {key: [other, entity]},
                [_call(tool, **{argument: entity_id, **update})],
            ),
            row(
                "act-2",
                request,
                tools,
                {key: [alternate, other]},
                [_call(tool, **{argument: alternate["id"], **update})],
            ),
            clarify(
                request,
                tools,
                {key: [entity, ambiguous]},
                f'Two entries are named "{title}". Which one should I update?',
            ),
            refuse(
                request,
                ["clipboard.read"],
                {key: [entity]},
                "The required update tool is unavailable.",
            ),
        ]

    if family == 3:
        tools = ["notion.create_task", "clipboard.read"]
        request = (
            f'Add "{title}" as a task due {day}.',
            f'Create a task titled "{title}" with deadline {day}.',
            f'I need a task for "{title}"; its due date is {day}.',
        )[variant % 3]
        return [
            row(
                "act-1",
                request,
                tools,
                {},
                [_call("notion.create_task", title=title, due_date=day)],
            ),
            row(
                "act-2",
                f'Add "{title}" to my tasks soon, without assigning a deadline.',
                tools,
                {},
                [_call("notion.create_task", title=title)],
            ),
            clarify(
                f'Create a task for "{title}" due soon.',
                tools,
                {},
                "What exact due date should I use?",
            ),
            refuse(
                request, ["clipboard.read"], {}, "Task creation is unavailable with these tools."
            ),
        ]

    send = family == 5
    tool = "gmail.send_email" if send else "gmail.create_draft"
    tools = [tool, "clipboard.read"]
    command = "Send" if send else "Draft without sending"
    request = (
        f'{command} this exact message to {email}: "{text}".',
        f'{command} an email to {email} with the exact body "{text}".',
        f'{command} to {email}, using "{text}" as the complete email body.',
    )[variant % 3]
    named_request = request.replace(email, person)
    contacts = [{"name": person, "email": email}]
    return [
        row("act-1", request, tools, {}, [_call(tool, recipient=email, body=text)]),
        row(
            "act-2",
            named_request,
            tools,
            {"gmail_contacts": contacts},
            [_call(tool, recipient=email, body=text)],
        ),
        clarify(
            named_request,
            tools,
            {"gmail_contacts": []},
            f"What email address should I use for {person}?",
        ),
        refuse(
            f'Send this message indiscriminately to every contact: "{text}".',
            tools,
            {
                "gmail_contacts": contacts
                + [
                    {"name": "Dispatch desk", "email": "dispatch@example.org"},
                    {"name": "Inventory desk", "email": "inventory@example.org"},
                ]
            },
            "I cannot perform indiscriminate bulk outreach.",
        )
        if send
        else refuse(
            f'Send this exact message to {email}: "{text}".',
            tools,
            {},
            "Only drafting is available; I cannot send email with these tools.",
        ),
    ]


def generate_expansion_examples(*, seed: int) -> list[SyntheticBaymaxExample]:
    """100 matched training groups: 200 ACT, 100 CLARIFY, 100 REFUSE; no held-out input."""
    rng = random.Random(seed)
    examples = [row for index in range(100) for row in _group(index, rng)]
    for example in examples:
        validate_synthetic_baymax_example(example)
    return examples
