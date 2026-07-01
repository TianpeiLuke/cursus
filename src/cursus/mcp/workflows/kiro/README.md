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
| `kiro-workflow-runtime.js` | The engine. Re-implements `agent/parallel/pipeline/phase/log` + `args`/`budget` on `kiro-cli`. Each `agent()` call is one turn (a fresh sub-agent, matching CC semantics), served by either transport (see below). Schema-forced output is emulated by a JSON-schema prompt suffix + parse/validate + bounded re-prompt. |
| `kiro-acp-client.js` | A dependency-free JS **ACP client** (JSON-RPC 2.0 over newline-delimited stdio) that spawns and drives one long-lived `kiro-cli acp` process. Used when `--transport acp`. Hand-rolled (no npm SDK), mirroring the internal `AGIArsenalConsole` pattern; wire contract captured live from `kiro-cli 2.10.0` (protocol version 1). |
| `run-workflow.js` | The CLI. Loads an unmodified CC workflow `.js`, binds the primitives as globals, executes it, prints the return value to **STDOUT** as JSON while all progress goes to **STDERR**. This is the Kiro counterpart of the CC `Workflow` tool. |

## Requirements

- `kiro-cli` on `PATH`. Developed/verified on **`kiro-cli 2.10.0`**; **SAIS runs a frozen `2.5.0`
  snapshot** — see [Version skew](#version-skew-kiro-cli-25x-vs-210x) below and use the `--legacy-kiro`
  / `--acp-entry` flags there. The runner shells out to `kiro-cli`; nothing else (no npm packages —
  the runtime is dependency-free, standard-library Node only).
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
| `--transport <t>` | `headless` (default) or `acp`. See below. |
| `--legacy-kiro` | Emit only the **kiro-cli 2.5.0-safe** flag set (drops `--effort` + granular `--trust-tools`; keeps `--no-interactive`/`--trust-all-tools`/`--agent`/`--model`). Use on the SAIS frozen snapshot. |
| `--acp-entry <e>` | `subcommand` (default, `kiro-cli acp`) or `chat-binary` (2.5.0 ships a separate `kiro-cli-chat` binary that IS the ACP server). |
| `--kiro-chat-bin <p>` | The ACP-server binary for `--acp-entry chat-binary` (default `kiro-cli-chat`). |
| `--acp-protocol <n>` | ACP `protocolVersion` to request (default `1`; re-negotiated to whatever the server echoes). |

## Transport: headless vs ACP

Both transports are implemented; pick with `--transport`.

**`headless` (default)** — each `agent()` call spawns one `kiro-cli chat --no-interactive` process.
No shared state, no dependencies, matches the verified internal headless pattern (SAGE 2054738).
Simplest and robust; every call pays full process-start + agent-init cost.

**`acp`** — the runtime holds ONE long-lived `kiro-cli acp` process (Agent Client Protocol =
JSON-RPC 2.0 over newline-delimited stdio) and runs one ACP session per `agent()` turn, via the
dependency-free client in `kiro-acp-client.js`. This is the mechanism real internal JS clients use
(`A2AServer4KiroCLI`, `KiroOverlayApp`, `AGIArsenalConsole`); here it is hand-rolled (no
`@agentclientprotocol/sdk`). Choose ACP to amortize process/agent-init cost across many turns, for a
persistent connection, live streaming, and structured per-tool permission handling (the client
auto-approves permission requests for unattended runs).

```bash
node run-workflow.js ../cursus-author-step.js --transport acp --args '{ … }'
```

ACP caveats: `--agent` / `--model` / `--effort` are **session-level** (set once when the ACP process
starts), so a per-call `opts.model` / `opts.effort` is honored only under `headless`. Turns are
**serialized** on the single ACP process (one prompt in flight at a time), so `parallel()` does not
give true wall-clock parallelism under ACP — it does under `headless` (independent processes). Rule of
thumb: `headless` for wide fan-out, `acp` for long sequential runs where init cost dominates.

## Version skew: kiro-cli 2.5.x vs 2.10.x

This runtime was captured/verified on **`kiro-cli 2.10.0`**, but **SAIS runs a frozen `kiro-cli 2.5.0`
snapshot** that is not updated. Older builds differ, so on 2.5.0:

- **Newer CLI flags are absent.** `--effort` and granular `--trust-tools=<csv>` are 2.10.0-era; passing
  them to 2.5.0 hard-fails with "unexpected argument". `--trust-all-tools`, `--agent`, `--model`, and
  `--no-interactive` are present on 2.5.0 (confirmed by the internal headless pattern, SAGE 2054738:
  `--no-interactive --trust-all-tools --agent`). **Fix:** pass `--legacy-kiro` to emit only the safe set.
  **CAVEAT — this is a capability gap, not just syntax.** Dropping `--effort` means a step that requested
  `effort: 'high'` (e.g. the `cursus-author-step` **Resolve** phase, which asks a weaker model for a
  15-field `PLAN_SCHEMA` JSON off a ~40K-token prompt) runs at the build's default/Auto model and often
  returns prose instead of JSON — the schema turn then fails all re-prompts and the pipeline item drops
  to `null` (empirically confirmed on the SAIS 2.5.0 snapshot). The runtime warns once when this happens.
  **On a Cloud Desktop you can `kiro-cli >= 2.10.0` (with `--effort`) for a fair run — but you CANNOT
  upgrade inside SAIS**, which is a Red-certified VPC with no internet egress (fixed VPC endpoints; the
  SAIS install procedure pins 2.5.0). Inside SAIS the fix must be kiro-version-independent: pin a capable
  model with `--model <id>` (from `kiro-cli chat --list-models`), and/or reduce what one turn demands
  (trim the prompt, split a big schema into smaller sequential turns). Simple/low-schema workflows (e.g.
  `cursus-configure-pipeline`, smaller per-node schemas) still run on 2.5.0 with `--legacy-kiro`.
- **The ACP entry point differs.** 2.10.0 exposes the `kiro-cli acp` subcommand; the 2.5.0 snapshot
  ships a separate **`kiro-cli-chat`** binary that is itself the ACP server (per the vault
  `repo_kiro_cli` note). **Fix:** `--transport acp --acp-entry chat-binary --kiro-chat-bin kiro-cli-chat`.
- **ACP protocol version may differ.** The client requests `--acp-protocol` (default 1) but adopts
  whatever the server echoes in the `initialize` result, so a mismatch is logged and negotiated rather
  than fatal.

Recommended on the SAIS 2.5.0 snapshot — start with **headless + `--legacy-kiro`** (the most robust
path; headless only needs the 2.5.0-safe flags), and only try ACP if you need streaming/session reuse:

```bash
# headless, 2.5.0-safe (recommended on SAIS)
node run-workflow.js ../cursus-author-step.js --legacy-kiro --args '{ … }'

# ACP on 2.5.0 (separate chat binary is the ACP server)
node run-workflow.js ../cursus-author-step.js --transport acp \
  --acp-entry chat-binary --kiro-chat-bin kiro-cli-chat --legacy-kiro --args '{ … }'
```

If a probe shows ACP is entirely absent on the frozen build, use headless — it is sufficient for every
workflow (each `agent()` is an independent turn). The offline test suite asserts the 2.5.0-safe arg
shape (`--legacy-kiro` drops the new flags; `--acp-entry chat-binary` spawns `kiro-cli-chat` with no
`acp` arg) so this stays correct without a live 2.5.0 binary.

## Schema-output shape (explicit directive + auto-coercion + re-prompt)

Because Kiro can't tool-force structured output (the Claude Code host does), the model must voluntarily
emit JSON matching the schema. Three layers defend the shape, applied to EVERY schema-gated turn in
EVERY workflow (no per-workflow code):

1. **An explicit shape directive on the initial prompt** (`shapeDirective(schema)`). The raw JSON Schema
   alone does not override some models' bias to wrap the answer in an array — on the frozen kiro-cli
   2.5.0, Opus 4.8 returned `[{...},{...}]` even for the *small* `locate` sub-schema. So for an object
   schema the runtime spells it out in words AND shows a `{ "field": ..., ... }` skeleton: "Return
   EXACTLY ONE JSON object … do NOT return a JSON array … do NOT return a list of candidates or
   alternatives — pick the single best and return ONLY that one object." This is on the FIRST attempt,
   not just re-prompts, because the bias shows on attempt 1.
2. **Auto-coercion before validating** (`coerceToSchema`). A single-element array `[{...}]` is unwrapped
   to the object; a *multi*-element array whose elements are all IDENTICAL (the model emitted the same
   answer N times) is unwrapped to the first (no data lost); a bare object is wrapped when the schema
   wants an array. It never changes field values and won't unwrap a multi-element array of DIFFERENT
   objects (ambiguous — left for the re-prompt).
3. **A sharp, specific re-prompt** on any remaining mismatch (`--schema-retries`, default 3). An object
   schema answered with an N-element array of different objects is named as exactly that ("you returned
   a JSON ARRAY of N items … this is not a list task … pick the single best answer"), not a generic
   "expected object, got array" the model ignored last time.

Failure modes seen in real SAIS runs, and how the three layers handle them:

- **Weak/unspecified model returns prose instead of JSON** (kiro-cli 2.5.0 without `--effort`, routed to
  a weak Auto model): no fix except a stronger model — pin one with `--model <id>`, or run on a newer CLI.
- **Array bias on ANY schema, even small ones** — Opus 4.8 on 2.5.0 wraps the answer in an array (often
  the same object repeated) regardless of schema size. Handled by layer 1 (the directive discourages it
  up front) + layer 2 (identical-element arrays are unwrapped) + layer 3 (a "pick one, not a list"
  re-prompt). Because these are runtime-level, they cover BOTH workflows at once — `cursus-author-step`'s
  locate/triage/identity turns AND `cursus-configure-pipeline`'s DagCheck + per-node Validate turns
  (which showed the same multi-element-array failure) — with no workflow edits.
- **Capable model returns a MULTI-element array of DIFFERENT objects for a large all-at-once schema** —
  on a later SAIS run Opus 4.8 answered `cursus-author-step`'s 15-field `PLAN_SCHEMA` (which discusses
  producer/NEW/consumer nodes) with `[{...},{...},...]` — one plan object PER NODE — plus a per-turn
  timeout on the oversized turn. Coercion cannot unwrap a genuinely multi-element array of different
  objects (which object is "the" plan is ambiguous), and the *generic* re-prompt in use at the time did
  not move it (the sharper "pick one, not a list" re-prompt of layer 3 came later). The robust fix for
  the big turn is additionally **on the workflow side and harness-conditional**: `run-workflow.js` sets
  `__workflowHost.toolForcesSchemaOutput = false` on the
  sandbox, and `cursus-author-step` reads that to **split Resolve into three small single-object turns**
  (locate → triage → identity, merged into the same `PLAN_SCHEMA`-shaped plan) ONLY on Kiro. Each small
  turn has one unambiguous object shape, so no per-node array and a much smaller/faster turn. Under the
  Claude Code `Workflow` host `__workflowHost` is undefined → treated as tool-forcing → Resolve stays ONE
  `StructuredOutput`-forced `PLAN_SCHEMA` turn, byte-for-byte unchanged. Takeaway: a large schema a
  non-tool-forcing host can't reliably shape should be **decomposed by the workflow for that host**, not
  forced through coercion. Proven by `test-author-step-e2e.js` — the real workflow reaches a green step on
  the Kiro runtime and the combined `PLAN_SCHEMA` turn is never emitted.

## How faithful is this to the Claude Code runtime?

Same primitive API, same tools, same phase order and gates. Differences to know:

| Aspect | Claude Code host | This Kiro runtime |
|--------|------------------|-------------------|
| `agent()` | fresh sub-agent, its final message is the return value | one kiro-cli turn (a `chat --no-interactive` process under `headless`, or one ACP session under `acp`) — same "fresh sub-agent" semantics |
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
