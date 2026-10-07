"""Authored training-only requests for supported-versus-unavailable contrast pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundaryCase:
    key: str
    tool: str
    request: str
    question: str
    initial_state: dict[str, Any] = field(default_factory=dict)


BOUNDARY_CASES: tuple[BoundaryCase, ...] = (
    BoundaryCase(
        "orientation_date",
        "calendar.create_event",
        "Reserve 40 minutes at 11:20 for volunteer orientation. I haven't picked the day.",
        "Which date should I use for volunteer orientation?",
    ),
    BoundaryCase(
        "rehearsal_start",
        "calendar.create_event",
        "June 12 is set for the rehearsal. Add a 75-minute event; the starting time is undecided.",
        "What time should the rehearsal start on June 12?",
    ),
    BoundaryCase(
        "consultation_length",
        "calendar.create_event",
        "Put a consultation on 2026-06-18 at 14:10. I still need to specify its length.",
        "How many minutes should the consultation last?",
    ),
    BoundaryCase(
        "unnamed_block",
        "calendar.create_event",
        "Add an event on 2026-06-09 from 10:00 to 10:50. Leave the title undecided for now.",
        "What title should I give the event? A title is required to create it.",
    ),
    BoundaryCase(
        "clock_period",
        "calendar.create_event",
        "Book a 25-minute equipment check on June 11 at 7:15. I might mean morning or evening.",
        "Should the equipment check start at 07:15 or 19:15?",
    ),
    BoundaryCase(
        "invalid_month_day",
        "calendar.create_event",
        "Schedule a 45-minute ceramics session on June 31, 2026 at 15:00.",
        "June has only 30 days. Which date did you intend?",
    ),
    BoundaryCase(
        "conflicting_length",
        "calendar.create_event",
        "Add a seminar on June 16 from 13:00 to 14:00, and make it exactly 90 minutes.",
        "Should the seminar end at 14:00 or last 90 minutes and end at 14:30?",
    ),
    BoundaryCase(
        "shift_relative_time",
        "calendar.create_event",
        "Schedule a half-hour pottery practice on June 8 immediately after my "
        "shift. My shift hours aren't provided.",
        "What time does your shift end on June 8?",
    ),
    BoundaryCase(
        "move_without_target",
        "calendar.update_event",
        "Move an event to June 17 at 12:40. I haven't told you which event yet.",
        "Which event should I move? Please provide its ID or identifying details.",
    ),
    BoundaryCase(
        "unknown_event_id",
        "calendar.update_event",
        "Change event evt_missing_kit to 30 minutes. That ID might be wrong.",
        "I cannot find evt_missing_kit in the supplied events. What is the correct event ID?",
        {"calendar_events": []},
    ),
    BoundaryCase(
        "untitled_change",
        "calendar.update_event",
        "Rename evt_open_day. I'll tell you the new name when you ask.",
        "What new title should evt_open_day have?",
        {
            "calendar_events": [
                {
                    "id": "evt_open_day",
                    "title": "Open day",
                    "start_date": "2026-06-13",
                    "start_time": "10:00",
                    "duration_minutes": 60,
                }
            ]
        },
    ),
    BoundaryCase(
        "two_walkthroughs",
        "calendar.update_event",
        "Shift the walkthrough to 16:30. There are two in the schedule; only one should move.",
        "Do you mean the studio walkthrough or the warehouse walkthrough?",
        {
            "calendar_events": [
                {
                    "id": "evt_studio",
                    "title": "Studio walkthrough",
                    "start_date": "2026-06-10",
                    "start_time": "12:00",
                    "duration_minutes": 40,
                },
                {
                    "id": "evt_warehouse",
                    "title": "Warehouse walkthrough",
                    "start_date": "2026-06-10",
                    "start_time": "14:00",
                    "duration_minutes": 40,
                },
            ]
        },
    ),
    BoundaryCase(
        "negative_event_length",
        "calendar.update_event",
        "Set the duration of evt_briefing to minus 20 minutes.",
        "The duration must be positive. How many minutes should the briefing last?",
        {
            "calendar_events": [
                {
                    "id": "evt_briefing",
                    "title": "Briefing",
                    "start_date": "2026-06-15",
                    "start_time": "09:00",
                    "duration_minutes": 20,
                }
            ]
        },
    ),
    BoundaryCase(
        "invalid_event_clock",
        "calendar.update_event",
        "Reschedule evt_tour to 25:10 on June 19.",
        "25:10 is not a valid clock time. What start time did you mean?",
        {
            "calendar_events": [
                {
                    "id": "evt_tour",
                    "title": "Campus tour",
                    "start_date": "2026-06-19",
                    "start_time": "11:00",
                    "duration_minutes": 50,
                }
            ]
        },
    ),
    BoundaryCase(
        "event_date_choice",
        "calendar.update_event",
        "Move evt_workshop to either June 21 or June 22 at 09:30; I haven't chosen between them.",
        "Which date should I use: June 21 or June 22?",
        {
            "calendar_events": [
                {
                    "id": "evt_workshop",
                    "title": "Repair workshop",
                    "start_date": "2026-06-20",
                    "start_time": "09:30",
                    "duration_minutes": 80,
                }
            ]
        },
    ),
    BoundaryCase(
        "event_change_unspecified",
        "calendar.update_event",
        "Update evt_induction, but don't change anything until I say what needs updating.",
        "What would you like to change about evt_induction?",
        {
            "calendar_events": [
                {
                    "id": "evt_induction",
                    "title": "Induction",
                    "start_date": "2026-06-23",
                    "start_time": "10:00",
                    "duration_minutes": 35,
                }
            ]
        },
    ),
    BoundaryCase(
        "send_missing_address",
        "gmail.send_email",
        "Send a note saying the parcel arrived. I haven't given you the recipient.",
        "Which email address should receive the parcel update?",
    ),
    BoundaryCase(
        "send_missing_content",
        "gmail.send_email",
        "Send an email to rowan@example.com. Ask me what it should say first.",
        "What should the email to rowan@example.com say?",
    ),
    BoundaryCase(
        "send_invalid_address",
        "gmail.send_email",
        "Email the receipt acknowledgement to imani-at-example.com: 'Received, thank you.'",
        "What is Imani's valid email address? imani-at-example.com is not a valid address.",
    ),
    BoundaryCase(
        "send_address_choice",
        "gmail.send_email",
        "Send 'The sample is ready' to one of these addresses: kit@example.com or "
        "ren@example.com. I need to choose one.",
        "Should I send the sample update to kit@example.com or ren@example.com?",
    ),
    BoundaryCase(
        "send_pronoun",
        "gmail.send_email",
        "Tell them by email that I'll be ten minutes late. There is no earlier "
        "conversation identifying them.",
        "Who should receive the delay notice, and what is their email address?",
    ),
    BoundaryCase(
        "send_unprovided_text",
        "gmail.send_email",
        "Email sol@example.com the wording I mentioned, but that wording isn't in "
        "this conversation.",
        "Please provide the wording you want sent to sol@example.com.",
    ),
    BoundaryCase(
        "send_confirmation",
        "gmail.send_email",
        "Email employer@example.com that I am resigning from my position effective immediately.",
        "Please confirm that you want me to send your immediate resignation to "
        "employer@example.com.",
    ),
    BoundaryCase(
        "send_conflicting_body",
        "gmail.send_email",
        "Email jude@example.com that the delivery is approved, but also say I "
        "have not approved it. Only one statement should go out.",
        "Should the email say the delivery is approved or not approved?",
    ),
    BoundaryCase(
        "draft_recipient",
        "gmail.create_draft",
        "Prepare an email draft saying the desk is available. I haven't selected a recipient yet.",
        "Who should the desk-availability draft be addressed to?",
    ),
    BoundaryCase(
        "draft_content",
        "gmail.create_draft",
        "Make an email draft for iris@example.com with subject 'Site visit'. I'll "
        "supply the body next.",
        "What should the body of the site-visit draft say?",
    ),
    BoundaryCase(
        "draft_two_contacts",
        "gmail.create_draft",
        "Draft a thank-you email to Robin. Either of the two Robins could be the one I mean.",
        "Should the draft go to robin.design@example.com or robin.events@example.com?",
        {
            "gmail_contacts": [
                {"name": "Robin", "email": "robin.design@example.com"},
                {"name": "Robin", "email": "robin.events@example.com"},
            ]
        },
    ),
    BoundaryCase(
        "draft_invalid_address",
        "gmail.create_draft",
        "Draft 'The form is attached separately' to the address @example.com.",
        "What is the full recipient address? @example.com is missing the part before @.",
    ),
    BoundaryCase(
        "draft_missing_decision",
        "gmail.create_draft",
        "Draft a reply to remy@example.com stating whether I accept or decline "
        "the invitation. I haven't decided yet.",
        "Do you want the draft to accept or decline the invitation?",
    ),
    BoundaryCase(
        "draft_missing_amount",
        "gmail.create_draft",
        "Draft an email to supplier@example.com with our agreed quantity. The "
        "quantity hasn't been provided; don't use a placeholder.",
        "What agreed quantity should I put in the draft?",
    ),
    BoundaryCase(
        "draft_subject_only",
        "gmail.create_draft",
        "Recipient: devon@example.com. Subject: Collection window. Create the "
        "draft once you know the message, which I haven't supplied.",
        "What message should the collection-window draft contain?",
    ),
    BoundaryCase(
        "draft_contact_unknown",
        "gmail.create_draft",
        "Draft a welcome email to Ellis. Ellis is not in the supplied contacts.",
        "What email address should I use for Ellis?",
        {"gmail_contacts": []},
    ),
    BoundaryCase(
        "task_missing_title",
        "notion.create_task",
        "Create a task due June 24. I still need to tell you what the task is.",
        "What title should I use for the new task?",
    ),
    BoundaryCase(
        "task_blank_title",
        "notion.create_task",
        "Add a task with an empty title: ''.",
        "A task needs a nonempty title. What should it be called?",
    ),
    BoundaryCase(
        "task_invalid_date",
        "notion.create_task",
        "Add 'Return display stands' with a due date of 2026-04-31.",
        "April has 30 days. Which due date should I use?",
    ),
    BoundaryCase(
        "task_deadline_choice",
        "notion.create_task",
        "Create 'Label storage boxes', due June 25 or June 27. Don't pick a date for me.",
        "Should Label storage boxes be due June 25 or June 27?",
    ),
    BoundaryCase(
        "task_undefined_deadline",
        "notion.create_task",
        "Make a task called 'Check stock' due on the deadline we agreed. That "
        "deadline is not included here.",
        "What exact due date did you agree for Check stock?",
    ),
    BoundaryCase(
        "task_title_choice",
        "notion.create_task",
        "Add one task, either 'Print badges' or 'Pack supplies'. Wait for me to choose which.",
        "Which task should I create: Print badges or Pack supplies?",
    ),
    BoundaryCase(
        "task_missing_reference",
        "notion.create_task",
        "Turn that into a task. There is no earlier message or copied text "
        "explaining what 'that' is.",
        "What action should the task describe?",
    ),
    BoundaryCase(
        "task_conflicting_status",
        "notion.create_task",
        "Create 'Inspect the display' and mark it both open and done. I only want one status.",
        "Should Inspect the display be open or done?",
    ),
    BoundaryCase(
        "task_update_target",
        "notion.update_task",
        "Change a task's deadline to June 28. I haven't identified the task yet.",
        "Which task should receive the June 28 deadline?",
    ),
    BoundaryCase(
        "task_update_unknown_id",
        "notion.update_task",
        "Mark task_missing_rack as done; I may have copied the wrong ID.",
        "I cannot find task_missing_rack in the supplied state. What is the correct task ID?",
        {"notion_tasks": []},
    ),
    BoundaryCase(
        "task_update_two_matches",
        "notion.update_task",
        "Complete the inventory task. There are two inventory tasks and I only finished one.",
        "Did you finish the tools inventory or the furniture inventory?",
        {
            "notion_tasks": [
                {"id": "task_tools", "title": "Tools inventory", "status": "open"},
                {"id": "task_furniture", "title": "Furniture inventory", "status": "open"},
            ]
        },
    ),
    BoundaryCase(
        "task_update_new_title",
        "notion.update_task",
        "Rename task_shipping. Ask me for the replacement title.",
        "What should task_shipping be renamed to?",
        {"notion_tasks": [{"id": "task_shipping", "title": "Shipping prep", "status": "open"}]},
    ),
    BoundaryCase(
        "task_update_new_due",
        "notion.update_task",
        "Change task_signage to the new deadline. No new date has been supplied yet.",
        "What is the new due date for task_signage?",
        {
            "notion_tasks": [
                {
                    "id": "task_signage",
                    "title": "Print signage",
                    "due_date": "2026-06-14",
                    "status": "open",
                }
            ]
        },
    ),
    BoundaryCase(
        "task_update_invalid_due",
        "notion.update_task",
        "Set task_labels due on 2026-13-06.",
        "There is no month 13. What valid due date should I use?",
        {"notion_tasks": [{"id": "task_labels", "title": "Replace labels", "status": "open"}]},
    ),
    BoundaryCase(
        "task_update_state_choice",
        "notion.update_task",
        "For task_display, use either open or done. I need to decide which status is accurate.",
        "Should task_display be open or done?",
        {"notion_tasks": [{"id": "task_display", "title": "Assemble display", "status": "open"}]},
    ),
    BoundaryCase(
        "task_update_which_change",
        "notion.update_task",
        "Update task_packing. I haven't said whether its title, date, or status needs changing.",
        "Which field should I change on task_packing, and what should the new value be?",
        {"notion_tasks": [{"id": "task_packing", "title": "Pack stands", "status": "open"}]},
    ),
    BoundaryCase(
        "copy_missing_text",
        "clipboard.write",
        "Replace my clipboard with the text I'm about to provide. I haven't provided it yet.",
        "What text should replace your clipboard contents?",
    ),
    BoundaryCase(
        "copy_either_line",
        "clipboard.write",
        "Copy one line: 'Doors open at nine' or 'Doors open at ten'. I still need "
        "to pick the correct line.",
        "Which line should I copy: Doors open at nine or Doors open at ten?",
    ),
    BoundaryCase(
        "copy_missing_quote",
        "clipboard.write",
        "Copy the quotation I referred to. Its words haven't appeared in this conversation.",
        "Please provide the exact quotation to copy.",
    ),
    BoundaryCase(
        "copy_unresolved_spelling",
        "clipboard.write",
        "Put the corrected spelling of my surname on the clipboard. I haven't told you my surname.",
        "What is the correctly spelled surname you want copied?",
    ),
    BoundaryCase(
        "copy_unselected_language",
        "clipboard.write",
        "Copy either 'Thank you' or 'Merci' for me. I haven't chosen the language.",
        "Should I copy Thank you or Merci?",
    ),
    BoundaryCase(
        "copy_missing_code_text",
        "clipboard.write",
        "Copy the sample code snippet I'll type next. No snippet has been supplied yet.",
        "Please provide the code snippet to copy.",
    ),
    BoundaryCase(
        "copy_empty_selection",
        "clipboard.write",
        "Copy the selected passage. The selection supplied here is empty; I meant "
        "to select some text.",
        "Which passage should I copy? The supplied selection is empty.",
    ),
    BoundaryCase(
        "copy_missing_number",
        "clipboard.write",
        "Put our agreed attendance count on the clipboard. The count is not given here.",
        "What attendance count should I copy?",
    ),
    BoundaryCase(
        "copy_conflicting_separator",
        "clipboard.write",
        "Copy 'north south' with spaces replaced by both commas and slashes. Only "
        "one separator should be used.",
        "Should the copied text use commas or slashes between north and south?",
    ),
    BoundaryCase(
        "copy_missing_unit",
        "clipboard.write",
        "Copy the measurement '18' with its correct unit. I haven't specified the "
        "unit; do not guess.",
        "Which unit should follow 18 in the copied measurement?",
    ),
    BoundaryCase(
        "copy_title_choice",
        "clipboard.write",
        "Copy one title for the brochure: 'A New Start' or 'Fresh Beginnings'. Wait for my choice.",
        "Which brochure title should I copy?",
    ),
    BoundaryCase(
        "copy_missing_sentence_end",
        "clipboard.write",
        "Copy a complete sentence starting 'The collection point is'. I haven't "
        "supplied its ending.",
        "Where is the collection point? Please provide the rest of the sentence.",
    ),
    BoundaryCase(
        "copy_missing_format_choice",
        "clipboard.write",
        "Copy the date June 26, 2026, either as 2026-06-26 or 26/06/2026. I need "
        "to choose the format.",
        "Which date format should I copy: 2026-06-26 or 26/06/2026?",
    ),
    BoundaryCase(
        "copy_missing_revision",
        "clipboard.write",
        "Copy the revised text. The earlier version was 'Entrance A', but I have "
        "not supplied the revision.",
        "What revised text should replace Entrance A?",
    ),
    BoundaryCase(
        "copy_ambiguous_reference",
        "clipboard.write",
        "I have two phrases, 'Use the side door' and 'Wait at reception'. Copy "
        "the one I mean; I haven't identified it.",
        "Should I copy Use the side door or Wait at reception?",
    ),
    BoundaryCase(
        "copy_missing_list_item",
        "clipboard.write",
        "Copy the first item from my packing list. I haven't provided any of the items.",
        "What is the first item on your packing list?",
    ),
)
