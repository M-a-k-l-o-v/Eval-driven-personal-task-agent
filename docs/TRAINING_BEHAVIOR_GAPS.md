# Training Behavior Gaps

This document tracks boundary examples BAYMAX should learn during Phase 2 SFT.
The v1 eval scenarios remain held out and must not be copied into training data.

## Clarification Examples

Use these when the request is supported, but the model needs more information
before it can safely call a tool.

- Missing recipient: "Email them the notes."
- Unclear pronoun: "Send him the update."
- Missing email body: "Draft an email to maya@example.com."
- Missing calendar time: "Schedule biology revision tomorrow."
- Vague time: "Schedule this tonight."
- Multiple matching calendar events: "Move my meeting to 3pm."
- Multiple matching Notion tasks: "Mark the review task done."
- No matching entity: "Move my physics meeting" when no such meeting exists.
- Invalid or impossible date/time: "Schedule this on 2026-02-30."
- Invalid email address: "Email Maya at maya-at-example."
- Empty clipboard write: "Copy this" with no text to copy.
- Multi-step missing dependency: "Create the task and email Marv about it."

## Refusal Examples

Use these when the request should not be done, even with more information.

- Unavailable service: taxi, food delivery, hotel booking, ticket purchase.
- Unsupported read/list/search operation when only create/update tools exist.
- Secret or password access.
- Private file exfiltration.
- Impersonation.
- Bulk unsolicited email.
- External payment or purchase.
- Unsupported deletion when no delete tool exists.
- Real-world action outside available tools.
- Account or data access not present in `initial_state`.

## Confirmation-Before-Action Examples

Use these when the action may be possible but should pause for user confirmation.
In the current SFT output shape, these are `request_clarification` examples.

- Send an email immediately.
- Cancel a calendar event.
- Delete a task.
- Overwrite important clipboard contents.
- Mark many tasks done.
- Destructive bulk update.
- Irreversible or high-impact action.
- Action that affects another person and cannot easily be undone.

## Template Rule

Each boundary template should ideally have a contrast example:

```text
unsupported, unsafe, or under-specified -> request_clarification/refuse_request
similar but complete and supported       -> direct tool call
```

This helps the model learn the boundary instead of memorizing one keyword.
