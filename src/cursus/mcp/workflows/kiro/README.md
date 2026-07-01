# Running cursus workflows on Kiro (`kiro-cli`)

This directory is the **Kiro alternative** to the Claude Code `Workflow` runtime. It lets the
*unmodified* dynamic-workflow scripts in the parent directory —
[`cursus-author-step.js`](../cursus-author-step.js) and
[`cursus-configure-pipeline.js`](../cursus-configure-pipeline.js) — run under `kiro-cli`, which has
no native workflow engine.

## Why this is needed

The parent scripts are written for the Claude Code `Workflow` runtime: an `export const meta = {...}`
header plus the host primitives `agent()` / `parallel()` / `pipeline()` / `phase()` / `log()` and the
`args` / `budget` globals. Those primitives are **injected by the Claude Code host** — they are not
defined in the file, and there is no `node <workflow>.js`. Under `kiro-cli` the scripts are therefore
inert: Kiro's own agent cannot execute that runtime.

The research behind this (abuse_slipbox `faq_kiro_run_claude_code_workflow.md`) found the fix: since
Kiro can't run the CC runtime, **re-implement the primitives on top of `kiro-cli`**. This directory
does exactly that. Because the primitive API is kept identical, the same workflow file runs in both
harnesses — Claude Code (via its `Workflow` tool) and Kiro (via this runner).

## What's here

| File | Role |
|------|------|
| `kiro-workflow-runtime.js` | The engine. Re-implements `agent/parallel/pipeline/phase/log` + `args`/`budget` on `kiro-cli`. Each `agent()` call is one headless `kiro-cli chat --no-interactive` turn (a fresh sub-agent, matching CC semantics). Schema-forced output is emulated by a JSON-schema prompt suffix + parse/validate + bounded re-prompt. |
| `run-workflow.js` | The CLI. Loads an unmodified CC workflow `.js`, binds the primitives as globals, executes it, prints the return value to **STDOUT** as JSON while all progress goes to **STDERR**. This is the Kiro counterpart of the CC `Workflow` tool. |

## Requirements

- `kiro-cli` on `PATH` (tested with `kiro-cli 2.10.0`). The runner shells out to it; nothing else
  (no npm packages — the runtime is dependency-free, standard-library Node only).
- Node.js ≥ 18 (tested on v22).
- The cursus MCP tools must be reachable **from `kiro-cli`'s own agent config** (`~/.kiro/…` or the
  active `--agent`). The workflow phases tell kiro-cli to call `author.*` / `validate.*` / `steps.io`
  / `compile.*`; kiro-cli invokes them itself. The workflows also carry a `cursus` CLI fallback.

## Usage

```bash
# author a new step (the undefined node between a producer and a consumer)
node run-workflow.js ../cursus-author-step.js \
  --args '{"intent":"a post-training probability calibration step","name":"BetaCalibration",
           "producer_node":"XGBoostModelEval_calibration","consumer_node":"ModelMetricsComputation",
           "dag_path":"projects/atoz_xgboost/atoz_xgboost_na.py"}' \
  1>result.json 2>progress.log

# generate a pipeline config.json for a DAG of existing step types
node run-workflow.js ../cursus-configure-pipeline.js \
  --args '{"project":"munged_address_pytorch","region":"NA",
           "project_root":"projects/munged_address_pytorch",
           "dag_nodes":["CradleDataLoading_tagging","TabularPreprocessing_sampling","PyTorchTraining"]}'
```

Since the workflow's return value goes to STDOUT and progress to STDERR, `1>result.json 2>progress.log`
cleanly separates them. Pass a JSON **array** to `--args` to batch (the `pipeline()` runs items
concurrently, capped).

### Options

| Flag | Meaning |
|------|---------|
| `--args '<json>'` | Value bound to the workflow's `args` global (object or array). Also `@file.json`, or the `KIRO_WF_ARGS` env var. |
| `--agent <name>` | kiro-cli agent (context profile) for every turn — use one whose config has the cursus MCP tools. |
| `--model <id>` / `--effort <low\|medium\|high\|xhigh\|max>` | Per-turn defaults (an `agent()` call's own `opts.effort` still overrides). |
| `--cwd <dir>` | Directory kiro-cli runs in. Default: the workflow file's directory, so relative `dag_path` / `project_root` in `args` resolve as they do under Claude Code. Point it at your project root if paths are relative to that. |
| `--concurrency <n>` | Max concurrent turns. Default `min(16, cores-2)`. |
| `--timeout-ms <n>` | Per-turn timeout. Default `900000` (15 min). |
| `--trust-tools <csv>` | Pass `--trust-tools=<csv>` instead of the default `--trust-all-tools`. |
| `--kiro-bin <path>` | kiro-cli binary (default `kiro-cli`). |
| `--budget <n>` | Ceiling for `budget.total` (a char/4 token *estimate* — kiro reports credits, not tokens). Default none. |

## Transport: headless vs ACP

This runner uses **headless** (`kiro-cli chat --no-interactive`) because each `agent()` call is an
independent one-shot turn (no shared session needed), it needs no dependencies, and it matches the
verified internal headless pattern (SAGE 2054738). kiro-cli invokes its own configured MCP tools
during each turn.

The streaming alternative is **`kiro-cli acp`** (Agent Client Protocol = JSON-RPC 2.0 over stdio),
driven from Node via the `@agentclientprotocol/sdk` npm package. Real internal JS/TS clients do this
(`A2AServer4KiroCLI`, `KiroOverlayApp`, `AGIArsenalConsole`). ACP is the right choice when you need a
persistent session, live token streaming, or interactive per-tool permission prompts — none of which
the workflow primitives require, so headless is simpler and sufficient here.

## How faithful is this to the Claude Code runtime?

Same primitive API, same tools, same phase order and gates. Differences to know:

| Aspect | Claude Code host | This Kiro runtime |
|--------|------------------|-------------------|
| `agent()` | fresh sub-agent, its final message is the return value | one `kiro-cli chat --no-interactive` turn — same "fresh sub-agent" semantics |
| Schema-forced output | forces a `StructuredOutput` tool call; retries on mismatch | appends the JSON Schema to the prompt, extracts + validates the JSON, **re-prompts** up to 2× on mismatch/parse-fail, then returns `null` |
| `parallel()` / `pipeline()` | barrier fan-out / no-barrier staged, cap `min(16,cores-2)` | same, same cap; a dead/throwing task → `null` (so `.filter(Boolean)` still works) |
| dead agent → `null` | yes | yes (transport failure after retries, or schema unmet) |
| `budget` | real output-token accounting | **estimate only** (`chars/4`); `budget.total` is `null` unless you pass `--budget`, so budget loops never run away |
| Resume | `resumeFromRunId` journal | not journaled — re-run from the top (that is why `Date`/`Math.random` are allowed here) |
| Gate rigor | schema is *enforced* by the tool layer | schema is *validated* post-hoc + re-prompted; a determined model can still drift, so the workflows' own executable oracles (`py_compile`, `json.load`, `validate.deps_resolve`) remain the real gates |

Net: the destination is identical because the same MCP tools do the real work and the same
non-skippable pipeline stages run in the same order. What you rely on the *workflow* (not the harness)
to guarantee correctness is the executable validation each script already performs — which is why the
cursus workflows gate on real parse/compile/resolve oracles rather than trusting a self-reported flag.

## Registering as a Kiro command

`kiro-cli` has no `.claude/workflows/` directory, so you don't register the `.js` itself. Two options:

1. **Wrapper skill (recommended):** a `.kiro/skills/<name>/SKILL.md` whose body runs
   `node <path>/run-workflow.js ../cursus-author-step.js --args '…'`. That makes `/<name>` a Kiro
   command that launches the workflow. (See `.kiro/skills/` in this repo for the skill pattern.)
2. **Direct:** call `node run-workflow.js …` from any shell / hook / CI step.

See also: abuse_slipbox `resources/faqs/faq_kiro_run_claude_code_workflow.md` ("Real ways JavaScript
DOES work with Kiro") and `resources/skills/skill_cursus_add_step.md` (the Kiro re-enactment mapping —
this runner is the *automated* form of that manual re-enactment).
