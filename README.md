# BAYMAX

A locally-runnable personal-task agent with an eval harness. Stepping-stone learning project — shipped as portfolio evidence of engineering + methodology discipline, not as a daily-driver assistant or research contribution.

**Status**: Closed out by the project owner on 2026-10-17 as a stepping-stone learning project. The final local fine-tune records 73/150 task successes on the development suite. Fake adapters only; not a production assistant.

See [docs/PROJECT.md](docs/PROJECT.md) for the full "what BAYMAX is and isn't."

---

## What it does

1. Accepts text input describing a productivity task
2. Selects and executes tool calls against **fake** in-memory adapters (calendar / notion / gmail / clipboard)
3. Confirms, refuses, or requests clarification
4. Logs actions with trace / task identifiers
5. Runs against a scripted eval suite; produces metrics JSON

## What it doesn't do (honest scope)

- **No real API integration** — fake adapters only. Not connected to your real Google Calendar / Gmail / Notion.
- **Not a daily-driver assistant** — response quality too low to trust for real actions
- **No menu bar app / voice / chat UI** — REST endpoint only (`POST /invoke`)
- **No context memory across requests** — each request stateless
- **No streaming, no multi-turn dialog**

Full list in [docs/DEFERRED.md](docs/DEFERRED.md).

---

## Team

| Name | Owned modules |
|---|---|
| **Marv** | Agent core, inference service, backends (OpenAI + MLX), output adapter, telemetry |
| **Ronin** | Eval harness, training pipeline, tool adapters (fake), scenario authorship |

Full ownership rules: [docs/OWNERSHIP.md](docs/OWNERSHIP.md).

---

## Documentation

| Doc | Purpose |
|---|---|
| [docs/PROJECT.md](docs/PROJECT.md) | What BAYMAX is (and isn't) — the framing doc |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design |
| [docs/OWNERSHIP.md](docs/OWNERSHIP.md) | Module ownership + PR rules |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Timeline to ship (~5 weeks from 2026-08-24) |
| [docs/BACKEND_FLOW.md](docs/BACKEND_FLOW.md) | Backend + agent design diagrams |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Eval methodology + results |
| [docs/SCORING.md](docs/SCORING.md) | Deterministic scoring rules |
| [docs/DEFERRED.md](docs/DEFERRED.md) | Known limitations + not-implemented items |
| [docs/decisions/](docs/decisions/) | ADR archive |
| [docs/experiments/README.md](docs/experiments/README.md) | Experiment history, settings, results and limitations |
| [Experiment notebook PDF](docs/experiments/pdf/BAYMAX-experiment-notebook.pdf) | Combined experiment reports with internal navigation |
| [Complete documentation PDF](docs/pdf/BAYMAX-complete-documentation.pdf) | Public project documents and experiment reports |

---

## Quick start

Install dependencies:

```bash
uv sync --frozen --all-groups
```

Run checks:

```bash
uv run ruff format --check
uv run ruff check .
uv run pyright
uv run pytest
```

Run the scripted baseline:

```bash
python -m baymax.eval.cli run-scripted \
  --scenarios scenarios/v1 \
  --responses scenarios/v1/scripted_responses/v1-scripted.json \
  --output results/phase1/scripted/v1-baseline.json
```

Run the OpenAI agent baseline:

```bash
export OPENAI_API_KEY="your_api_key"
python -m baymax.eval.cli run-agent-openai \
  --scenarios scenarios/v1 \
  --output results/phase1/openai/gpt-4o-mini-full.json \
  --model gpt-4o-mini
```

Phase 1 result files are committed under `results/`.

---

## Final local evaluation (150 development scenarios)

| Local checkpoint | Task successes / 150 | Routing macro-F1 | Invalid outputs / 150 |
| --- | ---: | ---: | ---: |
| Unweighted name-first reference, 1k / 2 epochs | 57 | 0.612 | 1 |
| Corrected grounding, 1k / 2 epochs | 53 | 0.591 | 0 |
| Expanded coverage, 2k / 2 epochs | 64 | 0.615 | 2 |
| Expanded coverage, 2k / 4 epochs | 73 | 0.681 | 0 |

Source: [saved four-epoch evaluation](results/phase2/json-spacing/name-first-coverage-2k-v2-epochs4-compact-combined-150.json).
All four use compact JSON, greedy decoding and the same combined v1/v2 suite.
This is single-seed development evidence after repeated evaluation, not an untouched
test set or proof of statistical significance. The final run correctly routes
17/40 clarification cases and 27/29 refusal cases; clarification remains a weakness.
Its recorded mean latency is 5,652 ms, versus 2,758 ms for the two-epoch 2k run.
Hardware conditions were not controlled sufficiently to attribute that difference
to epochs alone. Reported API cost is zero, not zero hardware/electricity cost.

Recipes, decision matrices and qualifications are in the
[four-epoch experiment report](docs/experiments/08-output-contract-learning/20-unique-2k-four-epochs.md).
Generated training data, adapter weights and private local files are not included
in Git. The committed config pins the dataset revision; reproducing or serving
the local model requires generating the data and obtaining/training its adapter.

## Historical Phase 1 baselines (50 scenarios)

| Backend | task_success_rate | scenarios |
|---|---|---|
| Scripted baseline | see `results/phase1/scripted/v1-baseline.json` | 50 |
| gpt-4o-mini | **0.52** | 50 |
| Qwen 1.5B + LoRA (smoke test) | 0.24 | 50 |

These historical 50-case results are not directly comparable to the final 150-case
local evaluations above. No claim that the fine-tune beats a frontier model is made.

---

## License

[MIT](LICENSE) — Marv and Ronin, 2026.
