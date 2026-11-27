# Held-Out SFT Clarification: Ten-Example Generation Sanity Check

**Status:** Completed by the owner; ten-example routing diagnostic, not a benchmark.

## Why This Check

The name-first model produced no valid clarification decisions on the established
150-case development suite. A read-only audit found **zero raw outputs mentioning
request_clarification**, compared with 26 raw mentions and 23 accepted clarification
decisions for E. This is not simply a parser hiding malformed clarification calls.

Before changing training data, test whether this checkpoint can generate clarification
on held-out inputs from the same SFT distribution. This does not retrain a model or
alter the benchmark, prompt policy, parser, class counts or existing result files.

## Selection and Inference

- Select the first 10 unique clarification prompts in the unchanged first 200
  validation rows. The selection is deterministic, not a representative random sample.
- Validate the supplied target/tool contracts and system-prompt hash.
- Reject exact input overlap with the first 1,000 raw training rows actually used
  for this experiment. This is not a semantic/paraphrase leakage audit.
- Pass only the saved system/user messages, unchanged, to inference. The expected
  assistant answer is recorded for inspection but is never supplied to the model.
- Reuse the same LocalLoraAgent loading, chat-template, generation and strict parser
  path as benchmark evaluation: greedy decoding, 256 new tokens, no output repair.
- Check adapter metadata against the selected config and record config, dataset,
  validation-file and adapter hashes. No tool calls are executed.
- Record raw output, parsed decision and expected target separately for every row.
  Count clarification decisions, not wording keywords or full task success.

## Commands

Run from the repository root in PowerShell:

```powershell
.venv\Scripts\python scripts\probe_sft_clarification.py `
  --output tmp\name-first-heldout-clarification-10.json
```

The default config is the trained two-epoch QKVO name-first candidate. Its expected
adapter path comes from that config. No extra adapter path or base-model override
is needed. CUDA is required unless explicitly allowing slow CPU inference.

For selection/preflight checks without loading a model or tokenizer:

```powershell
.venv\Scripts\python scripts\probe_sft_clarification.py `
  --dry-run `
  --output tmp\name-first-heldout-clarification-setup.json
```

Use a new output filename for subsequent runs; existing diagnostics are not
overwritten. Reports stay under ignored `tmp/` because they contain local SFT
inputs and expected targets, and are not published benchmark artifacts.

## Observed Result

The saved owner-run report is `tmp/name-first-heldout-clarification-10.json`:

| Recorded Prediction | Examples |
| --- | ---: |
| CLARIFY | 7 |
| ACT | 3 |
| REFUSE | 0 |
| Invalid/unknown | 0 |

All ten expected CLARIFY. The same seven raw outputs mentioned
`request_clarification`. This establishes that the checkpoint can generate valid
clarification calls on these selected held-out inputs. It does not establish that
the seven questions are adequate or that clarification transfers to the benchmark.

Unblinded inspection found clear question-quality gaps: row 25 asks only for the
start time when duration is also missing; row 42 asks for the task name/status
instead of the missing due date, despite the title being supplied. These are
qualitative observations, not an independent semantic review or revised scores.

## Interpretation

If clarification is generated here but absent from benchmark outputs, investigate
transfer and differences in the model-visible input distribution. If clarification
is also absent here, investigate the checkpoint and learning/generation behavior
before assuming the problem is only transfer to the benchmark.

Neither outcome establishes the cause by itself. Ten selected examples cannot
prove general competence or inability. Valid clarification syntax is not evidence
that the question is sufficient or appropriate; inspect the saved raw outputs and
expected targets. Adapter metadata/hash checks identify the requested checkpoint,
but do not constitute an independent numerical comparison of all loaded weights.
