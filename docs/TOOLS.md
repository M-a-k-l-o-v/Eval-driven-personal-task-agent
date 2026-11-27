# BAYMAX tool list (v1, fake adapters)

Source of truth for argument schemas: `FAKE_TOOL_SCHEMAS` in `src/baymax/eval/agent_runner.py`.
The same dict feeds the OpenAI backend's function definitions, the SFT data builder
(`models/baymax_sft.py`) and the LoRA runner (`eval/lora_runner.py`). If this file and that
dict disagree, the dict wins. Update both in the same PR.

All tools are in-memory fakes (`src/baymax/tools/`). No real API calls.
Every call returns `FakeToolResult(success, tool, error, data)`.

## Summary

| Adapter | Create | Update | Read / list / search | Send |
|---|---|---|---|---|
| calendar | `calendar.create_event` | `calendar.update_event` | none | n/a |
| notion | `notion.create_task` | `notion.update_task` | none | n/a |
| gmail | `gmail.create_draft` | none | none | `gmail.send_email` |
| clipboard | `clipboard.write` | n/a | `clipboard.read` | n/a |

8 tools total, plus 2 meta-tools. Only `clipboard.read` reads state.
Calendar, Notion and Gmail have no read, list, search, delete or cancel tools.
The meta-tool `refuse_request` is specified to cover "unsupported read/list/search",
so the fine-tune is trained to REFUSE those requests. Adding a read tool is a contract
and methodology change (docs/PROJECT.md §4), not a doc edit.

## calendar (state key: `calendar_events`)

### `calendar.create_event`
| Arg | Type | Required |
|---|---|---|
| title | string | yes |
| start_date | string | yes |
| start_time | string | yes |
| duration_minutes | integer >= 1 | yes |

Returns `data.event_id`, format `evt_fake_calendar_###`.

### `calendar.update_event`
| Arg | Type | Required |
|---|---|---|
| event_id | string | yes |
| title | string | no |
| start_date | string | no |
| start_time | string | no |
| duration_minutes | integer >= 1 | no |

Fails if `event_id` is not in state. Re-validates the whole event after the change.

## notion (state key: `notion_tasks`)

### `notion.create_task`
| Arg | Type | Required |
|---|---|---|
| title | string (min length 1) | yes |
| due_date | string | no |
| status | `open` or `done` (default `open`) | no |

Returns `data.task_id`, format `task_fake_notion_###`.

### `notion.update_task`
| Arg | Type | Required |
|---|---|---|
| task_id | string | yes |
| title | string | no |
| status | `open` or `done` | no |
| due_date | string | no |

Fails if `task_id` is not in state.

## gmail (state keys: `gmail_contacts`, `gmail_messages`, `gmail_drafts`, `sent_emails`)

### `gmail.create_draft`
| Arg | Type | Required |
|---|---|---|
| recipient | string, must be an email address (adapter-enforced) | yes |
| body | string (min length 1) | yes |
| subject | string | no |
| thread_id | string | no |

Returns a draft id, format `draft_fake_gmail_###`. Does not add to `sent_emails`.

### `gmail.send_email`
| Arg | Type | Required |
|---|---|---|
| recipient | string, must be an email address (adapter-enforced) | yes |
| body | string (min length 1) | yes |
| subject | string | no |

Returns a sent id, format `email_fake_gmail_###`. Does not add to `gmail_drafts`.
In v1 every fake is reversible, so the ADR 0009 verification gate never triggers.

## clipboard (state key: `clipboard`)

### `clipboard.read`
No arguments. Returns the current text in `data`.

### `clipboard.write`
| Arg | Type | Required |
|---|---|---|
| text | string | yes |

Replaces the clipboard text.

## Meta-tools (`src/baymax/core/meta_tools.py`)

| Tool | Args | Use |
|---|---|---|
| `request_clarification` | question: string (required) | action is available and safe but missing details or ambiguous |
| `refuse_request` | reason: string (required) | unsupported, unavailable, out of scope or unsafe |

## Schema enforcement status (as of 27 Nov 2026)

- All 8 tools have a JSON schema with `additionalProperties: false`.
- `Validator.validate_tool_call` does NOT enforce them yet. The `tool_schemas` branch is a `pass`, deferred to v2.
- In practice, bad arguments are caught by the fake adapters' typed signatures and Pydantic models.
  The executor returns `invalid arguments for <tool>` or `validation failed for <tool>`.
- Not specified anywhere: date and time formats for `start_date`, `start_time` and `due_date`.
- `gmail` recipient format is enforced only in the adapter, not in the schema.
- ADR 0002's example uses a single `start` field. The real contract is `start_date` + `start_time`.
- ADR 0002 and ARCHITECTURE section 3.1 name `src/baymax/core/schemas/tool_call.schema.json`. That file does not exist.
