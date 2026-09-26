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

Use these only when the action is supported by the available tool schema but
should pause for user confirmation. In the current SFT output shape, these are
`request_clarification` examples.

- A supported destructive or irreversible operation.
- A supported destructive bulk update.
- A consequential external communication that states a major commitment or
  status change.

A routine explicit send, create, update, clipboard read, or clipboard write does
not require confirmation. Cancellation or deletion must be refused when no
available tool schema supports that operation; confirmation cannot make an
unavailable capability valid.

## Template Rule

Each boundary template should ideally have a contrast example:

```text
unsupported, unavailable, or unsafe -> refuse_request
supported but under-specified        -> request_clarification
supported, safe, and complete        -> direct tool call
```

This helps the model learn the boundary instead of memorizing one keyword.
