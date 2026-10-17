# Template-Shift Diagnostic v1

This is a locked diagnostic suite for measuring whether ACT, CLARIFY, and
REFUSE behavior transfers beyond the wording used in BAYMAX training data.

The suite contains 30 hand-authored scenarios:

- 10 ACT scenarios
- 10 CLARIFY scenarios
- 10 REFUSE scenarios

The scenarios use indirect phrasing, different entities and domains, and hard
counterexamples where familiar words do not imply the familiar decision. For
example, a harmless clipboard phrase containing "password" must still ACT.

## Integrity Rules

- Never include these scenarios or their expected outputs in training data.
- Never generate training examples by reading or transforming these files.
- Do not change an expected behavior because a model failed it.
- Preserve raw result artifacts from every run.
- Use the same prompt and generation settings when comparing checkpoints.
- After results influence training decisions, describe this as a development
  diagnostic rather than an untouched final test.

The established 150-scenario benchmark remains `scenarios/v1/` plus
`scenarios/v2/`. This suite is reported separately.
