"""Generate the additive 100-scenario v2 evaluation suite."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from baymax.eval.scenario_loader import Scenario

OUTPUT_DIR = Path("scenarios/v2")
CURRENT_TIME = "2026-10-12T08:00:00Z"


def main() -> int:
    scenarios = [
        *_calendar_scenarios(),
        *_gmail_scenarios(),
        *_notion_scenarios(),
        *_clipboard_scenarios(),
        *_multi_tool_scenarios(),
    ]
    _assert_unique((str(raw["id"]) for raw in scenarios), "scenario ID")
    _assert_unique(
        (str(raw["user_input"]).casefold().strip() for raw in scenarios),
        "normalized user input",
    )
    validated = [Scenario.model_validate(raw) for raw in scenarios]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in validated:
        path = OUTPUT_DIR / f"{scenario.id}.json"
        path.write_text(
            json.dumps(scenario.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(validated)} v2 scenarios to {OUTPUT_DIR}")
    return 0


def _base(
    *,
    scenario_id: str,
    category: str,
    difficulty: str,
    description: str,
    user_input: str,
    available_tools: list[str],
    initial_state: dict[str, Any],
    expected_behavior: dict[str, Any],
    tags: list[str],
    current_time: str = CURRENT_TIME,
) -> dict[str, Any]:
    behavior_type = expected_behavior["type"]
    if behavior_type == "clarification":
        criteria = ["tool_not_called", "clarification_requested"]
    elif behavior_type == "refusal":
        criteria = ["tool_not_called", "refusal_given"]
    elif behavior_type == "tool_calls":
        criteria = ["correct_tool_call_order", "correct_tools_called", "correct_arguments"]
    else:
        criteria = ["correct_tool_called", "correct_arguments"]
    return {
        "id": scenario_id,
        "category": category,
        "difficulty": difficulty,
        "description": description,
        "user_input": user_input,
        "current_time": current_time,
        "available_tools": available_tools,
        "initial_state": initial_state,
        "expected_behavior": expected_behavior,
        "success_criteria": criteria,
        "tags": [category, difficulty, *tags],
    }


def _call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "tool": tool, "arguments": arguments}


def _clarify(*keywords: str) -> dict[str, Any]:
    return {"type": "clarification", "question_contains": list(keywords)}


def _refuse(*keywords: str) -> dict[str, Any]:
    return {"type": "refusal", "reason_contains": list(keywords)}


def _calls(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "tool_calls",
        "calls": [{"tool": tool, "arguments": arguments} for tool, arguments in calls],
    }


def _calendar_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    explicit = [
        ("portfolio review", "2026-10-15", "09:30", 45),
        ("database practice", "2026-10-16", "18:15", 75),
        ("project retrospective", "2026-10-20", "13:00", 30),
    ]
    for index, (title, date, time, duration) in enumerate(explicit, start=1):
        scenarios.append(
            _base(
                scenario_id=f"calendar_v2_explicit_create_{index:03}",
                category="calendar",
                difficulty="explicit",
                description="Create an event from fully specified calendar details.",
                user_input=(
                    f"Add {title} on {date} at {time} for {duration} minutes to my calendar."
                ),
                available_tools=["calendar.create_event"],
                initial_state={"calendar_events": []},
                expected_behavior=_call(
                    "calendar.create_event",
                    {
                        "title_contains": title,
                        "start_date": date,
                        "start_time": time,
                        "duration_minutes": duration,
                    },
                ),
                tags=["create_event", "single_tool"],
            )
        )
    scenarios.append(
        _base(
            scenario_id="calendar_v2_explicit_attendees_001",
            category="calendar",
            difficulty="explicit",
            description=(
                "Refuse an attendee invitation operation unavailable in the calendar tools."
            ),
            user_input="Invite lea@example.com to my calendar event on 2026-10-15.",
            available_tools=["calendar.create_event", "calendar.update_event"],
            initial_state={"calendar_events": []},
            expected_behavior=_refuse("attendee", "available"),
            tags=["unsupported_operation", "attendees"],
        )
    )
    implicit = [
        ("statistics review", "tomorrow", "2026-10-13", "17:00", 60),
        ("year-end journal", "in one hour", "2027-01-01", "00:30", 30),
        ("design check-in", "next Monday", "2026-10-19", "11:00", 45),
    ]
    for index, (title, phrase, date, time, duration) in enumerate(implicit, start=1):
        scenarios.append(
            _base(
                scenario_id=f"calendar_v2_implicit_create_{index:03}",
                category="calendar",
                difficulty="implicit",
                description="Resolve a relative date before creating a calendar event.",
                user_input=f"Schedule {title} {phrase} at {time} for {duration} minutes.",
                available_tools=["calendar.create_event"],
                initial_state={"calendar_events": []},
                expected_behavior=_call(
                    "calendar.create_event",
                    {
                        "title_contains": title,
                        "start_date": date,
                        "start_time": time,
                        "duration_minutes": duration,
                    },
                ),
                tags=["relative_time", "create_event"],
                current_time=(
                    "2026-12-31T23:30:00+04:00" if title == "year-end journal" else CURRENT_TIME
                ),
            )
        )
    for scenario_id, request, keywords in [
        (
            "calendar_v2_implicit_morning_001",
            "Book reading time tomorrow morning.",
            ("what time", "morning"),
        ),
        (
            "calendar_v2_implicit_lunch_001",
            "Put a mentor catch-up around lunch on Friday.",
            ("what time", "lunch"),
        ),
    ]:
        scenarios.append(
            _base(
                scenario_id=scenario_id,
                category="calendar",
                difficulty="implicit",
                description="Clarify an imprecise relative time period.",
                user_input=request,
                available_tools=["calendar.create_event"],
                initial_state={"calendar_events": []},
                expected_behavior=_clarify(*keywords),
                tags=["ambiguous_time", "clarification"],
            )
        )
    scenarios.append(
        _base(
            scenario_id="calendar_v2_implicit_list_week_001",
            category="calendar",
            difficulty="implicit",
            description="Refuse calendar listing when no read tool is available.",
            user_input="What appointments do I have next week?",
            available_tools=["calendar.create_event", "calendar.update_event"],
            initial_state={"calendar_events": []},
            expected_behavior=_refuse("list", "available"),
            tags=["read_unavailable", "refusal"],
        )
    )
    updates = [
        (
            "evt_v2_budget",
            "Budget review",
            "start_time",
            "15:30",
            "Move the budget review to 3:30pm.",
        ),
        (
            "evt_v2_coach",
            "Coaching call",
            "duration_minutes",
            60,
            "Make my coaching call one hour long.",
        ),
        (
            "evt_v2_demo",
            "Product demo",
            "start_date",
            "2026-10-16",
            "Move the product demo to Friday.",
        ),
    ]
    for index, (event_id, title, key, value, request) in enumerate(updates, start=1):
        scenarios.append(
            _base(
                scenario_id=f"calendar_v2_contextual_update_{index:03}",
                category="calendar",
                difficulty="contextual",
                description="Update one uniquely matching event from calendar state.",
                user_input=request,
                available_tools=["calendar.update_event"],
                initial_state={
                    "calendar_events": [
                        {
                            "id": event_id,
                            "title": title,
                            "start_date": "2026-10-14",
                            "start_time": "10:00",
                            "duration_minutes": 30,
                        }
                    ]
                },
                expected_behavior=_call(
                    "calendar.update_event", {"event_id": event_id, key: value}
                ),
                tags=["update_event", "state_lookup"],
            )
        )
    scenarios.extend(_calendar_boundary_scenarios())
    return scenarios


def _calendar_boundary_scenarios() -> list[dict[str, Any]]:
    common = {"category": "calendar"}
    multiple = {
        "calendar_events": [
            {
                "id": "evt_v2_sync_a",
                "title": "Team sync",
                "start_date": "2026-10-13",
                "start_time": "09:00",
                "duration_minutes": 30,
            },
            {
                "id": "evt_v2_sync_b",
                "title": "Team sync",
                "start_date": "2026-10-14",
                "start_time": "16:00",
                "duration_minutes": 30,
            },
        ]
    }
    rows = [
        (
            "calendar_v2_contextual_multiple_sync_001",
            "contextual",
            "Move the team sync to noon.",
            ["calendar.update_event"],
            multiple,
            _clarify("which", "team sync"),
        ),
        (
            "calendar_v2_contextual_delete_001",
            "contextual",
            "Delete the product demo from my calendar.",
            ["calendar.update_event"],
            {"calendar_events": []},
            _refuse("delete", "available"),
        ),
        (
            "calendar_v2_ambiguous_missing_date_001",
            "ambiguous",
            "Schedule a supplier call at 2pm.",
            ["calendar.create_event"],
            {"calendar_events": []},
            _clarify("date"),
        ),
        (
            "calendar_v2_ambiguous_missing_time_001",
            "ambiguous",
            "Add quarterly planning on October 22.",
            ["calendar.create_event"],
            {"calendar_events": []},
            _clarify("time"),
        ),
        (
            "calendar_v2_ambiguous_invalid_duration_001",
            "ambiguous",
            "Book focus time tomorrow for zero minutes.",
            ["calendar.create_event"],
            {"calendar_events": []},
            _clarify("duration"),
        ),
        (
            "calendar_v2_ambiguous_recurring_001",
            "ambiguous",
            "Create a recurring stand-up every weekday forever.",
            ["calendar.update_event"],
            {"calendar_events": []},
            _refuse("recurring", "available"),
        ),
        (
            "calendar_v2_ambiguous_rooms_001",
            "ambiguous",
            "Reserve conference room Cedar for my meeting.",
            ["calendar.update_event"],
            {"calendar_events": []},
            _refuse("room", "available"),
        ),
    ]
    return [
        _base(
            scenario_id=scenario_id,
            difficulty=difficulty,
            description=(
                "Handle a calendar boundary without taking unsupported or premature action."
            ),
            user_input=request,
            available_tools=available_tools,
            initial_state=state,
            expected_behavior=behavior,
            tags=["boundary_decision"],
            **common,
        )
        for scenario_id, difficulty, request, available_tools, state, behavior in rows
    ]


def _gmail_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    explicit = [
        (
            "draft",
            "nina@example.com",
            "Workshop slides",
            "The workshop slides are attached for review.",
        ),
        (
            "draft",
            "haris@example.com",
            "Travel receipt",
            "Here is the receipt from the conference trip.",
        ),
        ("send", "teamlead@example.com", "Status update", "The migration checklist is complete."),
    ]
    for index, (mode, recipient, subject, body) in enumerate(explicit, start=1):
        tool = "gmail.create_draft" if mode == "draft" else "gmail.send_email"
        scenarios.append(
            _base(
                scenario_id=f"gmail_v2_explicit_{mode}_{index:03}",
                category="gmail",
                difficulty="explicit",
                description=f"{mode.title()} an email with explicit recipient and content.",
                user_input=(
                    f"{mode.title()} an email to {recipient} titled '{subject}' saying '{body}'"
                ),
                available_tools=[tool],
                initial_state={"gmail_drafts": [], "sent_emails": []},
                expected_behavior=_call(
                    tool,
                    {"recipient": recipient, "subject": subject, "body_contains": body},
                ),
                tags=[mode, "single_tool"],
            )
        )
    scenarios.append(
        _base(
            scenario_id="gmail_v2_explicit_attachment_001",
            category="gmail",
            difficulty="explicit",
            description="Refuse an unsupported attachment operation.",
            user_input="Send finance@example.com an email with C:\\reports\\budget.xlsx attached.",
            available_tools=["gmail.send_email"],
            initial_state={"sent_emails": []},
            expected_behavior=_refuse("attachment", "available"),
            tags=["attachment", "unsupported_operation"],
        )
    )
    contacts = [
        (
            "Aisha Khan",
            "aisha.khan@example.com",
            "Draft Aisha a note that the prototype is ready.",
            "prototype is ready",
        ),
        (
            "Rami Noor",
            "rami.noor@example.com",
            "Write Rami an email saying I will join the call tomorrow.",
            "join the call tomorrow",
        ),
        (
            "Sara Chen",
            "sara.chen@example.com",
            "Prepare a message for Sara that the venue has changed.",
            "venue has changed",
        ),
    ]
    for index, (name, email, request, body) in enumerate(contacts, start=1):
        scenarios.append(
            _base(
                scenario_id=f"gmail_v2_implicit_contact_draft_{index:03}",
                category="gmail",
                difficulty="implicit",
                description="Resolve a named contact before creating a draft.",
                user_input=request,
                available_tools=["gmail.create_draft"],
                initial_state={
                    "gmail_contacts": [{"name": name, "email": email}],
                    "gmail_drafts": [],
                },
                expected_behavior=_call(
                    "gmail.create_draft",
                    {"recipient": email, "body_contains": body},
                ),
                tags=["contact_lookup", "draft"],
            )
        )
    boundary_rows = [
        (
            "gmail_v2_implicit_unknown_contact_001",
            "implicit",
            "Email Dana that the booking is confirmed.",
            {},
            _clarify("email", "Dana"),
        ),
        (
            "gmail_v2_implicit_pronoun_001",
            "implicit",
            "Draft him a message saying the report is ready.",
            {},
            _clarify("who", "recipient"),
        ),
        (
            "gmail_v2_implicit_search_unavailable_001",
            "implicit",
            "Find the email I received yesterday about parking.",
            {},
            _refuse("search", "available"),
        ),
    ]
    for scenario_id, difficulty, request, state, behavior in boundary_rows:
        scenarios.append(_gmail_boundary(scenario_id, difficulty, request, state, behavior))
    messages = [
        (
            "msg_v2_lease",
            "Mona",
            "mona@example.com",
            "Lease renewal",
            "Draft a reply to Mona saying I accept the renewal terms.",
        ),
        (
            "msg_v2_lab",
            "Dr Patel",
            "patel@example.edu",
            "Lab access",
            "Reply to Dr Patel that Thursday morning works.",
        ),
        (
            "msg_v2_vendor",
            "Vendor Support",
            "support@vendor.example",
            "Ticket 482",
            "Draft a reply to the vendor asking them to keep the ticket open.",
        ),
    ]
    for index, (message_id, name, email, subject, request) in enumerate(messages, start=1):
        scenarios.append(
            _base(
                scenario_id=f"gmail_v2_contextual_reply_{index:03}",
                category="gmail",
                difficulty="contextual",
                description="Draft a reply using one matching message in Gmail state.",
                user_input=request,
                available_tools=["gmail.create_draft"],
                initial_state={
                    "gmail_messages": [
                        {
                            "id": message_id,
                            "sender_name": name,
                            "sender_email": email,
                            "subject": subject,
                            "received_at": "2026-10-12T07:00:00Z",
                        }
                    ],
                    "gmail_drafts": [],
                },
                expected_behavior=_call(
                    "gmail.create_draft", {"recipient": email, "thread_id": message_id}
                ),
                tags=["reply", "message_lookup"],
            )
        )
    contextual = [
        (
            "gmail_v2_contextual_multiple_mona_001",
            "Reply to Mona's email saying yes.",
            {
                "gmail_messages": [
                    {
                        "id": "msg_mona_a",
                        "sender_name": "Mona",
                        "sender_email": "mona@example.com",
                        "subject": "Budget",
                        "received_at": "2026-10-12T06:00:00Z",
                    },
                    {
                        "id": "msg_mona_b",
                        "sender_name": "Mona",
                        "sender_email": "mona@example.com",
                        "subject": "Venue",
                        "received_at": "2026-10-12T07:00:00Z",
                    },
                ]
            },
            _clarify("which", "Mona"),
        ),
        (
            "gmail_v2_contextual_delete_001",
            "Delete the old draft to legal@example.com.",
            {"gmail_drafts": []},
            _refuse("delete", "available"),
        ),
    ]
    for scenario_id, request, state, behavior in contextual:
        scenarios.append(_gmail_boundary(scenario_id, "contextual", request, state, behavior))
    ambiguous = [
        (
            "gmail_v2_ambiguous_recipient_001",
            "Send the quarterly figures now.",
            {},
            _clarify("recipient"),
        ),
        (
            "gmail_v2_ambiguous_two_contacts_001",
            "Draft Alex a note that lunch moved.",
            {
                "gmail_contacts": [
                    {"name": "Alex Kim", "email": "alex.kim@example.com"},
                    {"name": "Alex Reed", "email": "alex.reed@example.com"},
                ]
            },
            _clarify("which", "Alex"),
        ),
        (
            "gmail_v2_ambiguous_send_or_draft_001",
            "Email me the onboarding wording for approval.",
            {},
            _clarify("draft", "send"),
        ),
        (
            "gmail_v2_ambiguous_schedule_001",
            "Schedule this email to be sent at midnight.",
            {},
            _refuse("schedule", "available"),
        ),
        (
            "gmail_v2_ambiguous_recall_001",
            "Recall the message I sent this morning.",
            {},
            _refuse("recall", "available"),
        ),
    ]
    for scenario_id, request, state, behavior in ambiguous:
        scenarios.append(_gmail_boundary(scenario_id, "ambiguous", request, state, behavior))
    return scenarios


def _gmail_boundary(
    scenario_id: str, difficulty: str, request: str, state: dict[str, Any], behavior: dict[str, Any]
) -> dict[str, Any]:
    return _base(
        scenario_id=scenario_id,
        category="gmail",
        difficulty=difficulty,
        description="Handle an email boundary without guessing or using an unsupported operation.",
        user_input=request,
        available_tools=["gmail.create_draft", "gmail.send_email"],
        initial_state={
            "gmail_contacts": [],
            "gmail_messages": [],
            "gmail_drafts": [],
            "sent_emails": [],
            **state,
        },
        expected_behavior=behavior,
        tags=["boundary_decision"],
    )


def _notion_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    explicit = [
        ("Renew library membership", "2026-10-28"),
        ("Submit internship reflection", "2026-11-02"),
        ("Review accessibility checklist", "2026-10-21"),
    ]
    for index, (title, due_date) in enumerate(explicit, start=1):
        scenarios.append(
            _base(
                scenario_id=f"notion_v2_explicit_create_{index:03}",
                category="notion",
                difficulty="explicit",
                description="Create a fully specified Notion task.",
                user_input=f"Create a Notion task called '{title}' due {due_date}.",
                available_tools=["notion.create_task"],
                initial_state={"notion_tasks": []},
                expected_behavior=_call(
                    "notion.create_task", {"title": title, "due_date": due_date}
                ),
                tags=["create_task", "single_tool"],
            )
        )
    scenarios.append(
        _base(
            scenario_id="notion_v2_explicit_database_001",
            category="notion",
            difficulty="explicit",
            description="Refuse an unsupported database creation request.",
            user_input=(
                "Create a Notion database named Vendor Directory with company and phone columns."
            ),
            available_tools=["notion.create_task", "notion.update_task"],
            initial_state={"notion_tasks": []},
            expected_behavior=_refuse("database", "available"),
            tags=["unsupported_operation", "database"],
        )
    )
    implicit = [
        ("Order replacement charger", "tomorrow", "2026-10-13"),
        ("Prepare seminar questions", "next Friday", "2026-10-16"),
        ("Send insurance documents", "in two days", "2026-10-14"),
    ]
    for index, (title, phrase, due_date) in enumerate(implicit, start=1):
        scenarios.append(
            _base(
                scenario_id=f"notion_v2_implicit_create_{index:03}",
                category="notion",
                difficulty="implicit",
                description="Resolve a relative due date before creating a task.",
                user_input=f"Add a task to {title.lower()} due {phrase}.",
                available_tools=["notion.create_task"],
                initial_state={"notion_tasks": []},
                expected_behavior=_call(
                    "notion.create_task", {"title_contains": title, "due_date": due_date}
                ),
                tags=["relative_date", "create_task"],
            )
        )
    notion_boundaries = [
        (
            "notion_v2_implicit_later_001",
            "implicit",
            "Remind me to update my portfolio later.",
            _clarify("when", "due"),
        ),
        (
            "notion_v2_implicit_weekend_001",
            "implicit",
            "Create a task to clean the storage room this weekend.",
            _clarify("date", "weekend"),
        ),
        (
            "notion_v2_implicit_list_unavailable_001",
            "implicit",
            "Show me everything due next month.",
            _refuse("list", "available"),
        ),
    ]
    for scenario_id, difficulty, request, behavior in notion_boundaries:
        scenarios.append(_notion_boundary(scenario_id, difficulty, request, {}, behavior))
    tasks = [
        (
            "task_v2_tax",
            "Collect tax statements",
            {"status": "done"},
            "Mark the tax statements task complete.",
        ),
        (
            "task_v2_bike",
            "Book bicycle service",
            {"due_date": "2026-10-18"},
            "Move the bicycle service task deadline to October 18.",
        ),
        (
            "task_v2_outline",
            "Draft conference outline",
            {"title": "Draft keynote outline"},
            "Rename the conference outline task to Draft keynote outline.",
        ),
    ]
    for index, (task_id, title, updates, request) in enumerate(tasks, start=1):
        scenarios.append(
            _base(
                scenario_id=f"notion_v2_contextual_update_{index:03}",
                category="notion",
                difficulty="contextual",
                description="Update one uniquely matching task from Notion state.",
                user_input=request,
                available_tools=["notion.update_task"],
                initial_state={
                    "notion_tasks": [
                        {"id": task_id, "title": title, "status": "open", "due_date": "2026-10-20"}
                    ]
                },
                expected_behavior=_call("notion.update_task", {"task_id": task_id, **updates}),
                tags=["update_task", "state_lookup"],
            )
        )
    contextual = [
        (
            "notion_v2_contextual_duplicate_invoice_001",
            "Mark the invoice task done.",
            {
                "notion_tasks": [
                    {"id": "task_invoice_a", "title": "Process invoice", "status": "open"},
                    {"id": "task_invoice_b", "title": "Process invoice", "status": "open"},
                ]
            },
            _clarify("which", "invoice"),
        ),
        (
            "notion_v2_contextual_delete_001",
            "Delete the expired warranty task.",
            {},
            _refuse("delete", "available"),
        ),
    ]
    for scenario_id, request, state, behavior in contextual:
        scenarios.append(_notion_boundary(scenario_id, "contextual", request, state, behavior))
    ambiguous = [
        (
            "notion_v2_ambiguous_missing_title_001",
            "Create a task due October 30.",
            _clarify("task", "title"),
        ),
        (
            "notion_v2_ambiguous_invalid_date_001",
            "Add renew permit due 2026-11-31.",
            _clarify("valid", "date"),
        ),
        ("notion_v2_ambiguous_target_001", "Mark that task as done.", _clarify("which", "task")),
        (
            "notion_v2_ambiguous_page_001",
            "Publish a public Notion page with my address.",
            _refuse("page", "available"),
        ),
        (
            "notion_v2_ambiguous_comments_001",
            "Add a comment to my planning task saying approved.",
            _refuse("comment", "available"),
        ),
    ]
    for scenario_id, request, behavior in ambiguous:
        scenarios.append(_notion_boundary(scenario_id, "ambiguous", request, {}, behavior))
    return scenarios


def _notion_boundary(
    scenario_id: str, difficulty: str, request: str, state: dict[str, Any], behavior: dict[str, Any]
) -> dict[str, Any]:
    return _base(
        scenario_id=scenario_id,
        category="notion",
        difficulty=difficulty,
        description="Handle a task boundary without guessing or using an unsupported operation.",
        user_input=request,
        available_tools=["notion.create_task", "notion.update_task"],
        initial_state={"notion_tasks": [], **state},
        expected_behavior=behavior,
        tags=["boundary_decision"],
    )


def _clipboard_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    explicit_texts = ["Gate B12", "Reference code ZX-481", "Call the electrician on Tuesday"]
    for index, value in enumerate(explicit_texts, start=1):
        scenarios.append(
            _base(
                scenario_id=f"clipboard_v2_explicit_write_{index:03}",
                category="clipboard",
                difficulty="explicit",
                description="Write explicitly provided text to the clipboard.",
                user_input=f"Copy this exact text to my clipboard: {value}",
                available_tools=["clipboard.write"],
                initial_state={"clipboard": {"text": ""}},
                expected_behavior=_call("clipboard.write", {"text": value}),
                tags=["write", "single_tool"],
            )
        )
    scenarios.append(
        _base(
            scenario_id="clipboard_v2_explicit_history_001",
            category="clipboard",
            difficulty="explicit",
            description="Refuse unsupported clipboard history access.",
            user_input="Show me the last ten things copied to my clipboard.",
            available_tools=["clipboard.read", "clipboard.write"],
            initial_state={"clipboard": {"text": "Current item"}},
            expected_behavior=_refuse("history", "available"),
            tags=["history", "unsupported_operation"],
        )
    )
    implicit_rows = [
        (
            "clipboard_v2_implicit_copy_number_001",
            "Put my booking number BK-9071 on the clipboard.",
            "BK-9071",
        ),
        (
            "clipboard_v2_implicit_copy_address_001",
            "Copy 14 Palm Street, Unit 8 for me.",
            "14 Palm Street, Unit 8",
        ),
        (
            "clipboard_v2_implicit_copy_phrase_001",
            "Keep 'review after lunch' ready to paste.",
            "review after lunch",
        ),
    ]
    for scenario_id, request, value in implicit_rows:
        scenarios.append(
            _base(
                scenario_id=scenario_id,
                category="clipboard",
                difficulty="implicit",
                description="Infer that supplied text should be written to the clipboard.",
                user_input=request,
                available_tools=["clipboard.write"],
                initial_state={"clipboard": {"text": ""}},
                expected_behavior=_call("clipboard.write", {"text": value}),
                tags=["write", "natural_phrasing"],
            )
        )
    boundaries = [
        (
            "clipboard_v2_implicit_missing_text_001",
            "implicit",
            "Copy that for me.",
            {},
            _clarify("what", "copy"),
        ),
        (
            "clipboard_v2_implicit_keep_it_001",
            "implicit",
            "Keep it on my clipboard.",
            {},
            _clarify("what", "clipboard"),
        ),
        (
            "clipboard_v2_implicit_passwords_001",
            "implicit",
            "Find any passwords I copied earlier.",
            {"clipboard": {"text": ""}},
            _refuse("password", "history"),
        ),
    ]
    for scenario_id, difficulty, request, state, behavior in boundaries:
        scenarios.append(_clipboard_boundary(scenario_id, difficulty, request, state, behavior))
    contextual = [
        (
            "clipboard_v2_contextual_read_reference_001",
            "Read the reference currently on my clipboard.",
            {"clipboard": {"text": "REF-2026-881"}},
            _call("clipboard.read", {}),
        ),
        (
            "clipboard_v2_contextual_replace_address_001",
            "Replace the copied address with 22 Marina Walk.",
            {"clipboard": {"text": "Old address"}},
            _call("clipboard.write", {"text": "22 Marina Walk"}),
        ),
        (
            "clipboard_v2_contextual_clear_001",
            "Clear the copied text.",
            {"clipboard": {"text": "Temporary note"}},
            _call("clipboard.write", {"text": ""}),
        ),
        (
            "clipboard_v2_contextual_whitespace_001",
            "Create a task from the clipboard text.",
            {"clipboard": {"text": "   "}},
            _clarify("empty", "clipboard"),
        ),
        (
            "clipboard_v2_contextual_file_001",
            "Paste my clipboard into C:\\notes\\today.txt.",
            {"clipboard": {"text": "Meeting summary"}},
            _refuse("file", "available"),
        ),
    ]
    for scenario_id, request, state, behavior in contextual:
        available_tools = None
        if scenario_id == "clipboard_v2_contextual_whitespace_001":
            available_tools = ["clipboard.read", "notion.create_task"]
        scenarios.append(
            _clipboard_boundary(
                scenario_id,
                "contextual",
                request,
                state,
                behavior,
                available_tools=available_tools,
            )
        )
    ambiguous = [
        (
            "clipboard_v2_ambiguous_read_or_write_001",
            "Use the clipboard for the access code.",
            _clarify("read", "write"),
        ),
        (
            "clipboard_v2_ambiguous_two_values_001",
            "Copy either the booking code or the address.",
            _clarify("which"),
        ),
        (
            "clipboard_v2_ambiguous_empty_write_001",
            "Overwrite my clipboard.",
            _clarify("text"),
        ),
        (
            "clipboard_v2_ambiguous_monitor_001",
            "Watch my clipboard and tell me whenever it changes.",
            _refuse("monitor", "available"),
        ),
        (
            "clipboard_v2_ambiguous_other_device_001",
            "Read the clipboard from my phone.",
            _refuse("device", "available"),
        ),
    ]
    for scenario_id, request, behavior in ambiguous:
        scenarios.append(_clipboard_boundary(scenario_id, "ambiguous", request, {}, behavior))
    return scenarios


def _clipboard_boundary(
    scenario_id: str,
    difficulty: str,
    request: str,
    state: dict[str, Any],
    behavior: dict[str, Any],
    available_tools: list[str] | None = None,
) -> dict[str, Any]:
    return _base(
        scenario_id=scenario_id,
        category="clipboard",
        difficulty=difficulty,
        description=(
            "Handle a clipboard boundary without guessing or using an unsupported operation."
        ),
        user_input=request,
        available_tools=available_tools or ["clipboard.read", "clipboard.write"],
        initial_state={"clipboard": {"text": ""}, **state},
        expected_behavior=behavior,
        tags=["boundary_decision"],
    )


def _multi_tool_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    topics = [
        ("visa appointment", "2026-10-19", "14:00", "visa documents", "consulate@example.com"),
        ("portfolio review", "2026-10-20", "10:30", "portfolio checklist", "mentor@example.com"),
        ("volunteer briefing", "2026-10-21", "18:00", "volunteer notes", "coordinator@example.com"),
        ("budget workshop", "2026-10-22", "09:00", "budget questions", "finance@example.com"),
        (
            "research interview",
            "2026-10-23",
            "15:30",
            "interview outline",
            "researcher@example.com",
        ),
    ]
    for index, (event, date, time, task, recipient) in enumerate(topics, start=1):
        scenarios.extend(
            [
                _base(
                    scenario_id=f"multi_tool_v2_calendar_notion_{index:03}",
                    category="multi_tool",
                    difficulty="multi_step",
                    description="Create a calendar event and preparation task.",
                    user_input=(
                        f"Schedule {event} on {date} at {time} for 45 minutes, "
                        f"then create a task to prepare the {task}."
                    ),
                    available_tools=["calendar.create_event", "notion.create_task"],
                    initial_state={"calendar_events": [], "notion_tasks": []},
                    expected_behavior=_calls(
                        (
                            "calendar.create_event",
                            {
                                "title_contains": event,
                                "start_date": date,
                                "start_time": time,
                                "duration_minutes": 45,
                            },
                        ),
                        ("notion.create_task", {"title_contains": task}),
                    ),
                    tags=["calendar", "notion"],
                ),
                _base(
                    scenario_id=f"multi_tool_v2_notion_draft_{index:03}",
                    category="multi_tool",
                    difficulty="multi_step",
                    description="Create a task and draft an email about it.",
                    user_input=(
                        f"Create a task called '{task}' due {date}, then draft "
                        f"{recipient} a message that it is on my list."
                    ),
                    available_tools=["notion.create_task", "gmail.create_draft"],
                    initial_state={"notion_tasks": [], "gmail_drafts": []},
                    expected_behavior=_calls(
                        ("notion.create_task", {"title": task, "due_date": date}),
                        (
                            "gmail.create_draft",
                            {"recipient": recipient, "body_contains": "on my list"},
                        ),
                    ),
                    tags=["notion", "gmail"],
                ),
                _base(
                    scenario_id=f"multi_tool_v2_clipboard_email_{index:03}",
                    category="multi_tool",
                    difficulty="multi_step",
                    description="Read clipboard text and place it in an email draft.",
                    user_input=(
                        f"Read the clipboard and draft its contents to {recipient} "
                        f"with subject '{event.title()}'."
                    ),
                    available_tools=["clipboard.read", "gmail.create_draft"],
                    initial_state={"clipboard": {"text": f"Notes for {event}"}, "gmail_drafts": []},
                    expected_behavior=_calls(
                        ("clipboard.read", {}),
                        (
                            "gmail.create_draft",
                            {
                                "recipient": recipient,
                                "subject": event.title(),
                                "body_contains": f"Notes for {event}",
                            },
                        ),
                    ),
                    tags=["clipboard", "gmail"],
                ),
                _base(
                    scenario_id=f"multi_tool_v2_two_tasks_{index:03}",
                    category="multi_tool",
                    difficulty="multi_step",
                    description="Create two distinct Notion tasks in order.",
                    user_input=(
                        f"Create a task for {task}, then another task to confirm the {event} venue."
                    ),
                    available_tools=["notion.create_task"],
                    initial_state={"notion_tasks": []},
                    expected_behavior=_calls(
                        ("notion.create_task", {"title_contains": task}),
                        ("notion.create_task", {"title_contains": f"{event} venue"}),
                    ),
                    tags=["notion", "repeated_tool"],
                ),
            ]
        )
    return scenarios


def _assert_unique(values: Iterable[str], label: str) -> None:
    values_list = list(values)
    if len(values_list) != len(set(values_list)):
        raise ValueError(f"duplicate {label} in generated v2 scenarios")


if __name__ == "__main__":
    raise SystemExit(main())
