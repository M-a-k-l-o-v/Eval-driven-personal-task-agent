# v2 Additive Scenario Coverage

This blueprint defines 100 new evaluation scenarios that extend, rather than
replace or paraphrase, the 50 scenarios in `scenarios/v1/`.

The two directories form a 150-scenario combined benchmark. Historical v1
results remain tied to the original 50 files. The v2 scenarios are evaluation
only and must never be used to construct training data.

## Suite Size

| Category | New v2 | Combined v1 + v2 |
| --- | ---: | ---: |
| Calendar | 20 | 33 |
| Gmail | 20 | 32 |
| Notion | 20 | 31 |
| Clipboard | 20 | 25 |
| Multi-tool | 20 | 29 |
| **Total** | **100** | **150** |

## Expected Behavior

| Expected behavior | New v2 | Combined v1 + v2 |
| --- | ---: | ---: |
| Single tool call | 36 | 54 |
| Ordered tool calls | 20 | 27 |
| Clarification | 24 | 40 |
| Refusal | 20 | 29 |
| **Total** | **100** | **150** |

Each single-tool category contributes nine direct actions, six clarifications,
and five refusals. Multi-tool contributes twenty ordered multi-call scenarios.

## Difficulty

| Difficulty | New v2 | Combined v1 + v2 |
| --- | ---: | ---: |
| Explicit | 16 | 31 |
| Implicit | 24 | 28 |
| Contextual | 20 | 32 |
| Ambiguous | 20 | 32 |
| Multi-step | 20 | 27 |
| **Total** | **100** | **150** |

Each single-tool category contributes four explicit, six implicit, five
contextual, and five ambiguous scenarios. All twenty multi-tool scenarios use
the `multi_step` difficulty.

## Functional Coverage

### Calendar

- Create events with exact, relative, cross-day, and timezone-aware times.
- Update title, date, time, and duration using contextual state.
- Clarify missing time, conflicting matches, and destructive changes.
- Refuse unsupported listing, deletion, recurrence, and attendee operations.

### Gmail

- Create drafts and send explicitly authorized messages.
- Resolve contacts and message threads from state.
- Distinguish drafting, replying, and sending.
- Clarify missing or conflicting recipients and confirmation requirements.
- Refuse unsupported search, deletion, scheduling, and attachment operations.

### Notion

- Create and update tasks with varied due dates and statuses.
- Resolve unique, duplicate, and missing tasks from state.
- Clarify invalid dates, unclear targets, and destructive requests.
- Refuse unsupported search, deletion, page, and database operations.

### Clipboard

- Read and write text without inventing clipboard state.
- Use existing clipboard content contextually.
- Clarify missing content and unclear read/write intent.
- Refuse password extraction, history access, monitoring, and filesystem access.

### Multi-tool

- Cover every supported tool family in ordered two- and three-call workflows.
- Include repeated calls to the same tool where the user requests multiple items.
- Verify order when one tool's state conceptually supplies a later action.
- Vary Calendar, Gmail, Notion, and clipboard pairings without copying v1 flows.

## State Coverage

- Empty state.
- One matching entity.
- Multiple matching entities.
- No matching entity.
- Duplicate entity.
- Conflicting calendar event.
- Multiple contacts with similar names.
- Multiple messages in one thread.
- Existing drafts and sent messages.
- Empty, whitespace-only, and populated clipboard state.
- Mixed state containing entities from multiple tools.

## Time Coverage

- Tomorrow and the day after tomorrow.
- Weekday names before and after the current weekday.
- Relative hour and minute offsets.
- Cross-midnight requests.
- Explicit UTC offsets.
- Month and year boundaries.
- Ambiguous periods such as morning, lunch, evening, and later.

## Safety And Failure Coverage

- Confirmation before supported destructive actions or consequential external commitments.
- Unsupported tool requests.
- Sensitive credential and password handling.
- Invalid dates, times, durations, recipients, and empty content.
- Duplicate prevention.
- No-match and multiple-match handling.
- No mutation for clarification and refusal scenarios.
- Calls only to tools listed in `available_tools`.

Adapter timeout, permission-denied, and rate-limit scenarios remain deferred
until scenario state can configure deterministic adapter failures and the scorer
can verify failure-aware responses. They must not be represented by pretending
an unsupported tool is an adapter failure.

## Acceptance Checks

- Exactly 100 JSON scenario files exist in `scenarios/v2/`.
- Every file validates through the Pydantic loader and JSON Schema.
- IDs and normalized user inputs are unique across v1 and v2.
- Category, expected-behavior, and difficulty counts match this blueprint.
- Every expected tool is present in that scenario's `available_tools`.
- No v2 scenario is a simple entity-renamed paraphrase of a v1 scenario.
- No evaluation scenario is imported into an SFT or training-data path.
