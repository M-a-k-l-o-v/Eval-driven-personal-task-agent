# OWNERSHIP — module ownership and PR rules

**Locked:** 2026-05-16
**Amended:** 2026-11-27 (tool adapters split into four rows, tool-contract rules added)
**Revisit at:** v1 retro (Jul 26), v2 retro (Sep 6)

---

## Why this exists

Clear ownership means a reviewer can ask "what did *you* build?" and get a specific, defensible answer. This file is the ground truth for that.

## The split

| Module | Owner | Co-owner / reviewer |
|---|---|---|
| **Agent core** (planner, router, state machine, tool-call contract, retry/recovery) | Marv | Ronin |
| **Inference service** (FastAPI/async, batching, MLX/llama.cpp serving, latency optimization) | Marv | Ronin |
| **Telemetry** (structured logging, OpenTelemetry traces, cost tracker) | Marv | Ronin |
| **Eval harness** (scenario format, runner, metrics, judge protocol, baseline vs candidate runner) | Ronin | Marv |
| **Model training pipeline** (data prep, SFT/LoRA, eval-during-training, model registry, ablations) | Ronin | Marv |
| **Tool contract** (tool schemas, argument semantics, the schema registry, `core/meta_tools.py` text) | Marv | Ronin |
| **Real tool adapters** (Notion, Google Calendar, Gmail, clipboard; auth, retries, rate limits, idempotency) | Marv | Ronin |
| **Fake tool adapters** (`src/baymax/tools/`, in-memory) | Ronin | Marv |
| **Tool contract tests** (one suite, run against both the fake and the real adapters) | Marv | Ronin |
| **Shared:** repo scaffolding · `ARCHITECTURE.md` · final benchmark writeup · demo video | both | — |

Person 1 = Marv
Person 2 = Ronin

## Tool-contract rules (added 2026-11-27)

"Tool adapters" used to be one row. It is now four, because the contract, the real code, the fake code and the tests have different authors and different information they may see.

1. **A schema change needs both approvals.** The contract owner proposes it, the other person must approve it, and the fake adapters and the contract tests change in the same release. Neither side merges a schema change alone.
2. **The fake adapters must pass the contract tests.** Ronin keeps the fakes. A fake that fails the suite is a bug in the fake, not in the test.
3. **Contract tests are written from the tool schemas only.** Never from `scenarios/**` or `results/**`. The PR body for any contract-test change states this. This keeps the runtime work blind to eval content (see `docs/PROJECT.md` §4).
4. **The schema registry lives in a neutral module** (target: `src/baymax/core/tools/`), not inside `eval/`. `eval/` and `models/` import from it. Until that move lands in its own PR, the registry stays where it is (`FAKE_TOOL_SCHEMAS` in `eval/agent_runner.py`), and this file does not authorise editing `src/baymax/eval/` or `src/baymax/models/` before then.
5. **Meta-tool text is versioned.** The `request_clarification` and `refuse_request` wording lives in Marv's `core/meta_tools.py` but is injected into Ronin's training data. Any change bumps its version, and `training_config_hash` must cover it, so a model id states which contract it was trained on.
6. **Model ids are immutable.** A change to the contract, prompt or schemas that affects training produces a new model id. Old results stay as they were and are never compared directly with the new ones.

## What "owner" means

For your owned modules:
- You make the architectural decisions, document them in `docs/decisions/NNNN-title.md` (ADR format), and defend them in review
- You design the module and write or direct the code, and you must be able to explain it (PR rule 6)
- You write the tests
- You write the section in `ARCHITECTURE.md`
- You are the named designer of the module; the co-owner is named as reviewer

For your co-owned modules:
- You review every PR; the owner cannot merge without your approval
- You can propose changes via PR but cannot merge without the owner's approval
- You ask questions; the owner answers and documents

## PR rules

1. `main` is protected. No direct pushes. No force-push to `main`, ever.
2. Every PR closes one tracked GitHub Issue, scoped < 2 days of work.
3. Every PR requires the relevant co-owner's review and approval before merge.
4. CI (ruff + pyright + pytest) must be green to merge.
5. PR description includes: what changed, why, how to test, and explicit "I wrote this" or "AI-assisted with: [scope]" disclosure.
6. If AI wrote code, the human must be able to explain it line-by-line on request. Don't merge code you can't defend.
7. No `--no-verify`. No skipping hooks.
8. Squash merge to keep `main` history clean.

## Conflict resolution

When the two of you disagree on an architecture choice that crosses owned modules:
1. Each writes a short doc (≤300 words) stating their proposal + tradeoff
2. Pick the one with the clearer measurable consequence (e.g. "this design lets us swap inference backends in <100 LOC vs ~1000 LOC")
3. If still tied, the owner of the module most affected gets the call
4. Document the decision in an ADR

## Self-honesty audit

Once a month, each person writes a 200-word retro on what they actually built vs what they shipped via AI. If you could not rewrite the critical part yourself when needed, AI is weakening you rather than helping. Catch this early.

## When this file changes

Any scope change to an owned module requires a PR that updates this file. No silent ownership drift.
