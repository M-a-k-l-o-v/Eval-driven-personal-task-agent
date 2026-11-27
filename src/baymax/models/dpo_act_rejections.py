"""Authored wrong-action negatives for training-only boundary requests.

These are rejected answers, not new supported capabilities or SFT targets.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from baymax.eval.agent_runner import FAKE_TOOL_SCHEMAS
from baymax.models.baymax_sft import PEOPLE, TASK_NOUNS, TIMES, TOPICS, SyntheticBaymaxExample
from baymax.models.boundary_cases import BOUNDARY_CASES


class AuthoredActRejection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: list[dict[str, Any]] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("calls")
    @classmethod
    def validate_calls(cls, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for call in calls:
            if (
                set(call) != {"name", "arguments"}
                or not isinstance(call.get("name"), str)
                or call["name"] not in FAKE_TOOL_SCHEMAS
            ):
                raise ValueError("ACT negative must use a known productive tool and arguments.")
            try:
                Draft202012Validator(FAKE_TOOL_SCHEMAS[call["name"]]).validate(call["arguments"])
            except SchemaValidationError as error:
                raise ValueError(
                    "ACT negative arguments violate the productive tool schema."
                ) from error
        return calls


# Every choice below intentionally resolves information the request leaves unresolved.
BOUNDARY_ARGUMENTS: dict[str, dict[str, Any]] = {
    "orientation_date": dict(
        title="volunteer orientation",
        start_date="2026-06-12",
        start_time="11:20",
        duration_minutes=40,
    ),
    "rehearsal_start": dict(
        title="rehearsal", start_date="2026-06-12", start_time="10:00", duration_minutes=75
    ),
    "consultation_length": dict(
        title="consultation", start_date="2026-06-18", start_time="14:10", duration_minutes=30
    ),
    "unnamed_block": dict(
        title="Appointment", start_date="2026-06-09", start_time="10:00", duration_minutes=50
    ),
    "clock_period": dict(
        title="equipment check", start_date="2026-06-11", start_time="07:15", duration_minutes=25
    ),
    "invalid_month_day": dict(
        title="ceramics session", start_date="2026-06-30", start_time="15:00", duration_minutes=45
    ),
    "conflicting_length": dict(
        title="seminar", start_date="2026-06-16", start_time="13:00", duration_minutes=60
    ),
    "shift_relative_time": dict(
        title="pottery practice", start_date="2026-06-08", start_time="17:00", duration_minutes=30
    ),
    "move_without_target": dict(
        event_id="evt_guessed", start_date="2026-06-17", start_time="12:40"
    ),
    "unknown_event_id": dict(event_id="evt_missing_kit", duration_minutes=30),
    "untitled_change": dict(event_id="evt_open_day", title="New open day"),
    "two_walkthroughs": dict(event_id="evt_studio", start_time="16:30"),
    "negative_event_length": dict(event_id="evt_briefing", duration_minutes=20),
    "invalid_event_clock": dict(event_id="evt_tour", start_time="01:10"),
    "event_date_choice": dict(event_id="evt_workshop", start_date="2026-06-21", start_time="09:30"),
    "event_change_unspecified": dict(event_id="evt_induction", start_time="11:00"),
    "send_missing_address": dict(recipient="guessed@example.com", body="The parcel arrived."),
    "send_missing_content": dict(recipient="rowan@example.com", body="The parcel arrived."),
    "send_invalid_address": dict(recipient="imani@example.com", body="Received, thank you."),
    "send_address_choice": dict(recipient="kit@example.com", body="The sample is ready."),
    "send_pronoun": dict(recipient="guessed@example.com", body="I'll be ten minutes late."),
    "send_unprovided_text": dict(recipient="sol@example.com", body="The parcel arrived."),
    "send_confirmation": dict(
        recipient="employer@example.com",
        body="I am resigning from my position effective immediately.",
    ),
    "send_conflicting_body": dict(recipient="jude@example.com", body="The delivery is approved."),
    "draft_recipient": dict(recipient="guessed@example.com", body="The desk is available."),
    "draft_content": dict(
        recipient="iris@example.com", subject="Site visit", body="The site visit is tomorrow."
    ),
    "draft_two_contacts": dict(recipient="robin.design@example.com", body="Thank you."),
    "draft_invalid_address": dict(
        recipient="guessed@example.com", body="The form is attached separately."
    ),
    "draft_missing_decision": dict(recipient="remy@example.com", body="I accept the invitation."),
    "draft_missing_amount": dict(
        recipient="supplier@example.com", body="Our agreed quantity is 20."
    ),
    "draft_subject_only": dict(
        recipient="devon@example.com", subject="Collection window", body="Collection is tomorrow."
    ),
    "draft_contact_unknown": dict(recipient="ellis@example.com", body="Welcome."),
    "task_missing_title": dict(title="Prepare materials", due_date="2026-06-24"),
    "task_blank_title": dict(title="Untitled task"),
    "task_invalid_date": dict(title="Return display stands", due_date="2026-04-30"),
    "task_deadline_choice": dict(title="Label storage boxes", due_date="2026-06-25"),
    "task_undefined_deadline": dict(title="Check stock", due_date="2026-06-25"),
    "task_title_choice": dict(title="Print badges"),
    "task_missing_reference": dict(title="Prepare materials"),
    "task_conflicting_status": dict(title="Inspect the display", status="open"),
    "task_update_target": dict(task_id="task_guessed", due_date="2026-06-28"),
    "task_update_unknown_id": dict(task_id="task_missing_rack", status="done"),
    "task_update_two_matches": dict(task_id="task_tools", status="done"),
    "task_update_new_title": dict(task_id="task_shipping", title="New shipping prep"),
    "task_update_new_due": dict(task_id="task_signage", due_date="2026-06-25"),
    "task_update_invalid_due": dict(task_id="task_labels", due_date="2026-12-06"),
    "task_update_state_choice": dict(task_id="task_display", status="done"),
    "task_update_which_change": dict(task_id="task_packing", status="done"),
    "copy_missing_text": dict(text="The parcel arrived."),
    "copy_either_line": dict(text="Doors open at nine"),
    "copy_missing_quote": dict(text="A new beginning awaits."),
    "copy_unresolved_spelling": dict(text="Taylor"),
    "copy_unselected_language": dict(text="Thank you"),
    "copy_missing_code_text": dict(text="print('hello')"),
    "copy_empty_selection": dict(text="The parcel arrived."),
    "copy_missing_number": dict(text="20"),
    "copy_conflicting_separator": dict(text="north,south"),
    "copy_missing_unit": dict(text="18 cm"),
    "copy_title_choice": dict(text="A New Start"),
    "copy_missing_sentence_end": dict(text="The collection point is reception."),
    "copy_missing_format_choice": dict(text="2026-06-26"),
    "copy_missing_revision": dict(text="Entrance B"),
    "copy_ambiguous_reference": dict(text="Use the side door"),
    "copy_missing_list_item": dict(text="Water bottle"),
}


def authored_act_rejection(source: SyntheticBaymaxExample) -> AuthoredActRejection | None:
    """Author only known training families; unknown cases remain explicitly uncovered."""
    if source.id.startswith("capability_boundary_"):
        role = "clarify" if source.behavior_type == "request_clarification" else "refuse"
        key = source.id.removeprefix("capability_boundary_").removesuffix("_" + role)
        case = next((case for case in BOUNDARY_CASES if case.key == key), None)
        if case is None or key not in BOUNDARY_ARGUMENTS:
            return None
        reason = (
            "Premature action without resolving: " + case.question
            if role == "clarify"
            else f"Attempts {case.tool}, which is absent from this request's available tools."
        )
        return _rejection(case.tool, BOUNDARY_ARGUMENTS[key], reason)
    if source.id.startswith("baymax_generated_clarify_"):
        return _generated_clarify_rejection(source)
    if source.id.startswith("baymax_generated_refuse_"):
        return _generated_refuse_rejection(source)
    return None


def _rejection(tool: str, arguments: dict[str, Any], reason: str) -> AuthoredActRejection:
    # Even unavailable-tool negatives must use an actual BAYMAX argument contract.
    return AuthoredActRejection(calls=[{"name": tool, "arguments": arguments}], reason=reason)


def _generated_clarify_rejection(source: SyntheticBaymaxExample) -> AuthoredActRejection:
    index = int(source.id.rsplit("_", 1)[1]) - 1
    topic, task = TOPICS[index % len(TOPICS)], TASK_NOUNS[index % len(TASK_NOUNS)]
    person = PEOPLE[index % len(PEOPLE)]
    operation = index % 12
    reason = (
        "Premature action without resolving: " + source.assistant_calls[0]["arguments"]["question"]
    )
    if operation == 0:
        return _rejection(
            "gmail.send_email",
            dict(
                recipient=source.initial_state["gmail_contacts"][0]["email"],
                body=f"The {topic} {task} is finished.",
            ),
            reason,
        )
    if operation == 1:
        return _rejection(
            "gmail.create_draft",
            dict(recipient=person["email"], body=f"The {topic} {task} is ready."),
            reason,
        )
    if operation in (2, 8):
        date = source.current_time[:10] if operation == 2 else "2026-02-28"
        return _rejection(
            "calendar.create_event",
            dict(title=f"{topic} {task}", start_date=date, start_time="20:00", duration_minutes=30),
            reason,
        )
    if operation in (3, 7):
        events = source.initial_state["calendar_events"]
        event_id = events[0]["id"] if events else f"event_guessed_{index}"
        return _rejection(
            "calendar.update_event",
            dict(
                event_id=event_id,
                start_time="15:00" if operation == 3 else TIMES[index % len(TIMES)],
            ),
            reason,
        )
    if operation == 4:
        return _rejection(
            "notion.update_task",
            dict(task_id=source.initial_state["notion_tasks"][0]["id"], status="done"),
            reason,
        )
    if operation == 5:
        return _rejection(
            "notion.create_task", dict(title=f"Revise {topic}", due_date="2026-06-03"), reason
        )
    if operation == 6:
        return _rejection("clipboard.write", dict(text="Review checklist"), reason)
    body = (
        f"Regarding {topic}."
        if operation == 9
        else f"I accept the {topic} service contract."
        if operation == 10
        else f"I resign from the {topic} committee effective immediately."
    )
    return _rejection("gmail.send_email", dict(recipient=person["email"], body=body), reason)


def _generated_refuse_rejection(source: SyntheticBaymaxExample) -> AuthoredActRejection:
    index = int(source.id.rsplit("_", 1)[1]) - 1
    person, topic = PEOPLE[index % len(PEOPLE)], TOPICS[index % len(TOPICS)]
    operation = index % 10
    if operation in (0, 4, 8):
        return _rejection(
            "notion.create_task",
            dict(title=source.user_input),
            "Substitutes a task/reminder for an unsupported booking, deletion or payment; "
            "does not perform the requested action.",
        )
    if operation == 6:
        return _rejection(
            "calendar.create_event",
            dict(
                title=source.user_input,
                start_date="2026-06-01",
                start_time="09:00",
                duration_minutes=30,
            ),
            "Creates an event instead of fulfilling an unavailable read/list operation.",
        )
    if operation == 7:
        return _rejection(
            "gmail.create_draft",
            dict(recipient=person["email"], body=f"Messages from {person['name']}"),
            "Creates a draft instead of fulfilling an unavailable search operation.",
        )
    if operation == 9:
        return _rejection(
            "clipboard.write",
            dict(text=f"Notes from {person['name']}'s account"),
            "Invents and copies another person's account contents without access or authorization.",
        )
    body = {
        1: f"This is {person['name']} writing to the class.",
        2: "I accessed your saved password.",
        3: f"Promotion for {topic}.",
        5: "Private files from your computer are included below.",
    }[operation]
    reason = {
        1: "Attempts an impersonation and invents the class recipient.",
        2: "Claims unsupported access to a saved password and sends a secrets-related message.",
        3: "Sends promotional outreach despite the request targeting "
        "every contact indiscriminately.",
        5: "Invents private-file contents and attempts their exfiltration.",
    }[operation]
    recipient = person["email"] if operation in (2, 3) else "guessed@example.com"
    return _rejection("gmail.send_email", dict(recipient=recipient, body=body), reason)
