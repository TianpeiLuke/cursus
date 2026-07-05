# The Agent (MCP) Tool Surface

Cursus ships its ML-pipeline engine three ways. Humans drive it through the
[CLI](../cli.rst) and the [Python API](../api/index.rst); **agents** drive it through the
`cursus.mcp` tool surface — a framework-neutral set of small, typed, JSON-in / JSON-out
tools that any agent framework (an MCP client, OpenAI function-calling, Claude `tool_use`,
Bedrock) can call. All three front ends sit on top of the same engine (`core.compiler`,
`api.dag`, `step_catalog`, `validation`, …), so the tool surface is not a parallel
implementation — it is a *third adapter* over the same capabilities.

This page explains that surface: the registry that declares every tool, the result
envelope every tool returns, the 12 tool namespaces, the self-documenting `help` design,
the optional MCP server adapter and its exporters, and the shipped dynamic-workflow
orchestrators plus the Kiro runtime that runs them.

```{contents}
:local:
:depth: 2
```

## Why a tool surface

An agent cannot import Python modules, read tracebacks, or hold a live `PipelineDAG`
object in memory the way a script can. It needs capabilities that are:

- **discoverable** — it can ask "what can I do?" and "how do I call this?" in-band;
- **uniform** — every call returns the same success/error shape, so the agent's control
  loop never has to special-case a stack trace or a printed message;
- **safe** — a tool call never crashes the caller, and state-mutating tools are flagged;
- **portable** — the same tool definitions feed an MCP server, an OpenAI tool list, and a
  Claude tool list without rewriting.

`cursus.mcp` provides exactly this. The package (`src/cursus/mcp/__init__.py`) is designed
to sit *parallel to* `cursus.core` / `cursus.steps` and explicitly does **not** replace
`cursus.cli` — the human-facing CLI and the agent-facing tools are siblings over one engine.

## The envelope: `ToolResult` / `ToolError`

Every tool returns a `ToolResult` (`src/cursus/mcp/envelope.py`) — a small,
JSON-serializable success/error record. Tools **never raise across the tool boundary and
never print**; the envelope is the entire contract, so any agent runtime can consume the
outcome uniformly.

```python
from cursus.mcp import call_tool

result = call_tool("catalog.search", {"query": "xgboost"})
result.ok        # True / False
result.data      # JSON-serializable payload (on success)
result.error     # human-readable message (on failure)
result.code      # machine-readable code, e.g. "not_found", "invalid_input"
```

`ToolResult` fields:

| Field | Meaning |
| --- | --- |
| `ok` | Whether the tool succeeded. |
| `data` | JSON-serializable payload on success (`None` on error). |
| `error` | Human-readable error message on failure. |
| `code` | Machine-readable error code (`not_found`, `invalid_input`, `internal_error`, …). |
| `warnings` | Non-fatal messages the agent may surface or reason about. |
| `next_steps` | On success: in-band `{"tool", "when", "why", "args_hint"?}` hints naming the tool(s) an agent would typically call next — the golden path lives on the result, not in the prompt. |
| `remedy` | On failure: `{"suggested_tools": [...], "fix_action": str}` recovery hint, so the agent need not reverse-engineer a fix from the bare `code`. |
| `meta` | Side-band info (tool name, timings, counts) — never required for correctness. |

Two constructors keep the shape consistent: `ToolResult.success(data, warnings=…,
next_steps=…, **meta)` and `ToolResult.failure(message, code=…, details=…, remedy=…)`.
`to_dict()` renders a compact wire form that drops empty optional fields.

`ToolError` is the *handled-failure* signal a tool raises internally (bad input,
not-found, validation). The registry's invoker catches it and turns it into
`ToolResult.failure(...)`, preserving `code` and `details`. Genuinely unexpected
exceptions are allowed to propagate so the invoker can wrap them as an `internal_error`
envelope — see [`call_tool`](#invocation-call-tool) below.

## The registry: `ToolDef` and friends

`src/cursus/mcp/registry.py` is the **single source of truth**. Every tool is one
declarative `ToolDef` (a frozen dataclass), and the whole registry is a map keyed by the
tool's dotted name (`"catalog.list_steps"`).

```python
@dataclass(frozen=True)
class ToolDef:
    name: str                  # dotted, unique, namespaced: "compile.dag"
    description: str           # what + when, for the agent
    schema: Dict[str, Any]     # JSON Schema (draft-07 object) for the arguments
    handler: ToolHandler       # (validated arg dict) -> ToolResult
    destructive: bool = False  # True if it mutates external state (SageMaker upsert, disk)
    tags: tuple = ()           # lifecycle phase(s): "planner" / "validator" / "programmer"
    when: str = ""             # one-line "call this when …" trigger cue
    examples: tuple = ()       # copy-paste invocation strings

    @property
    def namespace(self) -> str:      # leading segment of name → "compile"
        return self.name.split(".", 1)[0]
```

Three fields are what make the surface self-teaching:

- **`when`** — the *trigger* condition ("call this when …"), complementing the *what* in
  `description`.
- **`examples`** — real, copy-paste invocation strings, e.g.
  `'catalog.list_steps {"job_type": "training"}  # only training-variant steps'`. Stored as a tuple
  so the frozen dataclass stays truly immutable.
- **`tags`** — the lifecycle phase(s) a tool belongs to, so an agent can route by phase
  instead of scanning every description.

### Registry assembly

The registry is built once and cached. `_build_registry()` imports each namespace module
listed in `_TOOL_MODULES`, collects its module-level `TOOLS: List[ToolDef]`, and derives
the namespace key from the tools themselves (the leading segment of `td.name`), **not** the
filename — for example `info.py` defines the `tools.*` namespace. A namespace that fails to
import (say an optional engine dependency is missing) is logged and skipped rather than
breaking the whole registry. Each module also exposes a one-line `NAMESPACE` string that
becomes the namespace's description.

Public accessors:

| Function | Returns |
| --- | --- |
| `get_registry(force_reload=False)` | The canonical `name -> ToolDef` map (built once, cached). |
| `get_namespaces()` | `namespace -> one-line description` map (built with the registry). |
| `list_tools(namespace=None)` | All `ToolDef`s, sorted by name, optionally filtered to one namespace. |
| `get_tool(name)` | One `ToolDef` or `None`. |

### Invocation: `call_tool`

`call_tool(name, args)` is the in-process invoker and enforces the tool contract. It
**always returns a `ToolResult`** — an unknown tool, invalid arguments, a handled
`ToolError`, or an unexpected exception are all converted to error envelopes, so a tool
call can never crash the caller:

1. Unknown name → `ToolResult.failure(code="unknown_tool")` with the available names in
   `details`.
2. Light JSON-schema validation via `_validate_args` — checks required keys, unknown keys
   (when `additionalProperties` is `False`), and top-level `enum` membership. This is
   deliberately *not* a full validator; it gives the agent fast, clear feedback on obvious
   mistakes, while the handler remains the final authority. Failure →
   `ToolResult.failure(code="invalid_input")`.
3. Run the handler. A raised `ToolError` becomes `failure(code=te.code, …)`; any other
   exception is logged and wrapped as `code="internal_error"`. On success, `meta["tool"]`
   is stamped with the tool name.

## The 12 namespaces

Tools are grouped into 12 dotted namespaces, each declared by one module under
`src/cursus/mcp/tools/`. The namespace descriptions below are the exact `NAMESPACE`
strings the registry collects.

| Namespace | Purpose | Representative tools |
| --- | --- | --- |
| `catalog` | Discover/search steps, configs, and builders (`step_catalog` + registry). | `catalog.list_steps`, `catalog.search`, `catalog.step_info`, `catalog.step_spec`, `catalog.resolve_step`, `catalog.config_fields`, `catalog.list_builders` |
| `dag` | Construct, validate, and serialize pipeline DAGs (`api.dag`). | `dag.construct`, `dag.validate_integrity`, `dag.serialize`, `dag.deserialize`, `dag.dependencies`, `dag.resolve_plan` |
| `config` | Schema-driven config generation and loading (`api.factory` + `core.config_fields`). | `config.requirements`, `config.field_info`, `config.load`, `config.merge_save` |
| `compile` | Compile/validate/preview a DAG into a SageMaker pipeline (`core.compiler`). | `compile.dag`, `compile.validate`, `compile.preview`, `compile.with_report`, `compile.name`, `compile.single_node` |
| `validate` | Alignment, dependency, and script-execution checks (`validation`, `core.deps`). | `validate.alignment`, `validate.builder`, `validate.deps_resolve`, `validate.deps_explain`, `validate.step_interface`, `validate.step`, `validate.run_scripts`, `validate.info` |
| `execdoc` | MODS execution-document generation and validation (`mods.exe_doc`). | `execdoc.template`, `execdoc.generate`, `execdoc.merge`, `execdoc.validate` |
| `pipeline_catalog` | Recommend/select/load pre-built shared DAGs (`pipeline_catalog`). | `pipeline_catalog.list`, `pipeline_catalog.recommend`, `pipeline_catalog.auto_select`, `pipeline_catalog.get_dag`, `pipeline_catalog.load_dag`, `pipeline_catalog.config_guidance` |
| `project` | Scaffold a new Cursus pipeline project (phase-0 skeleton + action-item ledger). | `project.init`, `project.bring_up` |
| `strategies` | Inspect the builder strategy library — axes and knobs (`registry.strategy_registry`). | `strategies.list`, `strategies.show`, `strategies.list_axes`, `strategies.knobs`, `strategies.for_step_type` |
| `steps` | Per-step connection/I-O view: container paths, property refs, channels. | `steps.io`, `steps.patterns` |
| `author` | Guardrails + preflight checks for authoring a new step (rules, checklist, preflight). | `author.rules`, `author.checklist`, `author.config_constraints`, `author.preflight_config`, `author.check_script`, `author.preflight_step` |
| `tools` | Meta/discovery over the tool registry itself (help, by_phase, describe_tool). | `tools.help`, `tools.by_phase`, `tools.describe_tool` |

(The `tools` namespace is defined in `info.py`; the module filename and the namespace key
differ on purpose.) The generated [MCP tools reference](../reference/generated/mcp_tools.md)
lists every tool with its full schema.

### Lifecycle phases

Every tool is tagged with one or more lifecycle phases (`info.py`), which is how an agent
routes work rather than reading dozens of descriptions. The three phase names are the same
`planner` / `validator` / `programmer` taxonomy the agentic-workflow design assigns to its
three cooperating agents — the tool tags let a single agent (or three specialized ones)
route work by role instead of scanning every tool:

| Phase | Intent |
| --- | --- |
| `planner` | Discover, select, and assemble — explore steps/configs/DAGs and plan the pipeline. |
| `validator` | Check before building — alignment, dependencies, config/DAG integrity, scripts. |
| `programmer` | Compile and generate — turn a resolved DAG into a pipeline / execution doc. |

These intents are the exact `_PHASE_DESCRIPTIONS` strings `tools.help` reports. A typical
agent run flows PLAN → VALIDATE → PROGRAM: discover steps and assemble a DAG, check
alignment/dependencies/integrity, then compile the DAG into a SageMaker pipeline and
generate the execution document. The phase tags are not just documentation — `tools.by_phase`
filters the live registry by tag, so "give me the validator tools" is a single query.

### Destructive tools

Two shipped tools carry `destructive=True` and an agent host may gate them behind
confirmation:

- `compile.dag` — its optional `upsert=true` path mutates SageMaker state (build-only is
  the default and is non-destructive).
- `project.init` — creates a new project folder tree on disk.

## Self-documenting design

The surface teaches itself, so an agent never has to be pre-loaded with the whole tool
list. Three layers cooperate.

### `tools.help` — the front door

`tools.help` (`info.py`) introduces the *entire* toolset in one call: a short overview,
the three lifecycle phases with counts, and every tool grouped by namespace with its
one-line description. Optional filters narrow the output without changing its shape:

```bash
tools.help {}                                   # full overview of every namespace and tool
tools.help {"namespace": "compile"}             # just the compile.* tools
tools.help {"phase": "validator"}               # every validator-phase tool
tools.help {"namespace": "dag", "include_schema": true}   # dag tools + JSON input schemas
```

The result also carries `next_steps` pointing at `<namespace>.help`,
`tools.describe_tool`, and `tools.by_phase` — the golden discovery path is baked into the
payload.

### Auto `<ns>.help` — per-namespace overviews

The registry auto-generates a `<namespace>.help` tool for **every** namespace that
declares a `NAMESPACE` description (via `_make_namespace_help_tool`). So an agent working
in the compile space can call `compile.help` and get just the compile overview plus its
tools, without scanning the whole surface. Each generated help delegates to `tools.help`
with `namespace` pinned, so there is exactly one rendering path. The meta `tools`
namespace is excluded (`_NO_AUTO_HELP`) because its hand-written `tools.help` is already
the global front door, and a hand-written `<ns>.help` is never overwritten.

### `tools.by_phase` and `tools.describe_tool`

- `tools.by_phase(phase)` returns the tools tagged for one lifecycle phase — turning tool
  selection from "read 50+ descriptions" into a single filtered query.
- `tools.describe_tool(name)` returns one tool's full descriptor: description, `when` cue,
  `examples`, JSON input schema, phase `tags`, and the `destructive` flag.

Because this namespace reads only the live registry and has no engine dependencies, it
never fails to import and is cheap to call.

### Folding guidance into a single description

External MCP / OpenAI clients only receive one description *string* per tool — they cannot
see the structured `when` / `examples` fields. `render_description(td)` folds those in:

```text
<description>

When: <when>

Examples:
  - <example 1>
  - <example 2>
```

In-process agents get the fields structured (via help/describe); external agents get the
same guidance concatenated into the description, so nobody misses the usage cues.

## The server and exporters

Because the registry is the single source of truth, the same `ToolDef`s generate every
external descriptor shape.

### MCP server adapter

`src/cursus/mcp/server.py` is a *thin* adapter that mounts the framework-neutral tools onto
an actual Model Context Protocol server. The official `mcp` Python SDK is an **optional**
dependency, imported lazily, so importing `cursus.mcp` (the tools, schemas, and registry)
never requires the SDK — only the running server does. If the SDK is absent, a clear,
actionable error points at the extra to install.

```bash
python -m cursus.mcp.server      # run the cursus tools as a stdio MCP server
```

`build_server(name="cursus")` wires the server's `list_tools` from the registry (each entry
uses `render_description(td)` and `td.schema`), and routes `call_tool` **straight through**
`registry.call_tool`, serializing the returned envelope as a JSON `TextContent` block. So
the MCP server and in-process callers share exactly one code path and one result contract.

### Framework exporters

Two pure functions generate tool lists for other agent frameworks from the same registry:

| Exporter | Output shape |
| --- | --- |
| `export_openai_tools(namespace=None)` | OpenAI / Claude function-calling shape — `{"type": "function", "function": {"name", "description", "parameters"}}`, with phase `tags` under `function.metadata`. |
| `export_mcp_tools(namespace=None)` | MCP `list_tools` shape — `{"name", "description", "inputSchema"}`, with phase `tags` alongside. |

Both fold the `when` / `examples` guidance into the description via `render_description`, so
external OpenAI/Claude/MCP clients see the same usage hints an in-process agent gets.

### Mirroring the CLI and API

The [CLI](../cli.rst) mirrors the tool surface directly. `cursus mcp` (see
`src/cursus/cli/mcp_cli.py`) exposes three subcommands, all of which read the same registry:

```bash
cursus mcp help                       # the guided overview — same data as the tools.help tool
cursus mcp help --namespace compile   # zoom into one namespace (auto-shows examples)
cursus mcp help --phase validator     # filter by lifecycle phase
cursus mcp list-tools                 # print the registered tools
cursus mcp list-tools --format json   # machine-readable listing (no SDK needed)
cursus mcp serve                      # run the stdio MCP server (needs the optional mcp SDK)
```

`cursus mcp help` literally calls `call_tool("tools.help", …)` and renders the resulting
`ToolResult.data`, so the human overview and the agent overview can never drift. Engine and
SDK imports are lazy, so `cursus --help` and `cursus mcp list-tools` work even without the
optional `mcp` SDK installed.

The result: **one registry, three front ends.** The [Python API](../api/index.rst) exposes
the engine to scripts, the [CLI](../cli.rst) exposes it to humans, and `cursus.mcp` exposes
it to agents — each tool namespace maps onto the same subsystem the CLI and API already use
(`core.compiler`, `api.dag`, `step_catalog`, `validation`, `mods.exe_doc`, …).

## Dynamic-workflow orchestrators

Individual tools are primitives; real tasks are multi-step. `src/cursus/mcp/workflows/`
ships **Claude Code dynamic-workflow** scripts that orchestrate the tools into deterministic,
resumable, batch-capable sequences. They are JavaScript orchestration scripts for the Claude
Code `Workflow` runtime (an `export const meta = {...}` header plus the host-injected
`agent()` / `parallel()` / `pipeline()` / `phase()` primitives) — reference/runnable
artifacts, **not** importable Python. Each workflow's phases are non-skippable pipeline
stages, so a task cannot reach its final "green" stage without passing the same gates CI
runs.

| Workflow | What it does |
| --- | --- |
| `cursus-init-project.js` | Scaffold a brand-new project package (the phase-0 skeleton that is byte-identical across projects) plus an action-item ledger of what remains. |
| `cursus-configure-pipeline.js` | Author or repair a project's `generate_config.py` — produce a pipeline `config.json` for a DAG of **existing** step types, with per-node and cross-node (whole-graph) validation gates. |
| `cursus-author-step.js` | DAG-driven authoring of a **new** step type between a producer and a consumer, gated on spec alignment (the resolver's 6-component compatibility score), preflight constructibility, and script/interface checks. |
| `cursus-new-project.js` | End-to-end bring-up: composes the three workflows above so a team runs one command and ends with a compile-ready project. |

The workflows call the cursus tools as **MCP tools** (`author.checklist`,
`validate.step_interface`, `steps.io`, `compile.preview`, …). If the MCP server is
unreachable in the harness, the gate stages fall back to the `cursus` CLI (e.g.
`cursus validate step-interface …`) — they never fall back to hand-editing the registry,
because `.step.yaml` interface files are the source of truth (see
[Registry & discovery](registry_and_discovery.md)). Two of these flows also have a
deterministic, offline MCP-tool equivalent: `project.init` scaffolds statelessly, and
`project.bring_up` returns the `cursus-new-project` invocation for a caller that wants the
whole chain. See `src/cursus/mcp/workflows/README.md` for the full stage-by-stage design.

## The Kiro runtime

The workflow scripts target the Claude Code `Workflow` runtime, whose primitives are
injected by the Claude Code host — under a different harness the scripts are inert.
`src/cursus/mcp/workflows/kiro/` is the **Kiro alternative**: it re-implements those same
primitives (`agent` / `parallel` / `pipeline` / `phase` / `log` plus the `args` / `budget`
globals) on top of `kiro-cli`, so the *unmodified* parent workflow scripts run under Kiro
as well.

Key pieces (`kiro/README.md`):

| File | Role |
| --- | --- |
| `kiro-workflow-runtime.js` | The engine — re-implements the CC workflow primitives on `kiro-cli`. Each `agent()` call is one fresh sub-agent turn, matching CC semantics; schema-forced output is emulated with a JSON-schema prompt suffix plus parse/validate and a bounded re-prompt. |
| `kiro-acp-client.js` | A dependency-free JS ACP client (JSON-RPC 2.0 over newline-delimited stdio) that spawns and drives one long-lived `kiro-cli acp` process, used with `--transport acp`. |
| `run-workflow.js` | The CLI — loads an unmodified CC workflow `.js`, binds the primitives as globals, runs it, and prints the return value as JSON on STDOUT while progress goes to STDERR. The Kiro counterpart of the CC `Workflow` tool. |

Because the primitive API is kept identical, one workflow file runs in both harnesses. When
Kiro reaches the MCP-tool stages, it can inject the `cursus mcp serve` stdio server into the
ACP session (`--mcp-cursus --transport acp`) so the same `author.*` / `validate.*` /
`compile.*` tools are reachable — the surface stays the same no matter which agent runtime
drives it.

## See also

- [Registry & interface-first discovery](registry_and_discovery.md) — the step registry the
  `catalog.*` and `author.*` tools read.
- [DAG & compilation](dag_and_compilation.md) — the engine behind `dag.*` and `compile.*`.
- [Dependency resolution](dependency_resolution.md) — the resolver behind
  `validate.deps_resolve` / `validate.deps_explain` and the edge-alignment score.
- [MCP tools reference](../reference/generated/mcp_tools.md) — every tool with its schema.
- [CLI reference](../cli.rst) — `cursus mcp help` / `list-tools` / `serve`.
- [Python API reference](../api/index.rst).
