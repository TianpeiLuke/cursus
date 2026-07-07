# Drive Cursus with the Agent (MCP) Tools

Cursus exposes its whole pipeline engine as a set of **namespaced, JSON-in / JSON-out
tools** — the same surface a coding agent uses to do what a human engineer does by hand:
discover steps, assemble and validate a DAG, work out the config a pipeline needs,
compile it into a SageMaker pipeline, and author brand-new steps. This tutorial walks
that surface end to end.

Everything lives under [`cursus.mcp`](../api/index.rst). The design is deliberately layered:

- **The tools themselves** (`cursus.mcp.tools.*`) are pure Python functions with JSON
  schemas. They have no dependency on any agent framework.
- **The registry** (`cursus.mcp.registry`) is the single source of truth: one
  `name -> ToolDef` map keyed by dotted name (`"catalog.list_steps"`).
- **`call_tool`** is the in-process invoker. Call it from Python, from a notebook, or
  from your own agent loop.
- **The MCP server** (`cursus.mcp.server`) is a thin, optional adapter that mounts the
  exact same tools onto a real Model Context Protocol server.

Because all three read one registry, the CLI, the in-process API, and the MCP server can
never drift apart — and the surface is **self-documenting**: one tool describes all the
others.

## The shape of a tool

Every tool is declared once as a `ToolDef` (see `cursus.mcp.registry`):

| Field | Meaning |
|-------|---------|
| `name` | Dotted, unique name — `"<namespace>.<verb>"`, e.g. `compile.dag`. |
| `description` | What the tool does and when to call it. |
| `schema` | JSON Schema for the arguments (`type`, `properties`, `required`). |
| `handler` | The Python function that runs it. |
| `destructive` | `True` if it mutates external state (e.g. upserting a SageMaker pipeline). |
| `tags` | Lifecycle phase(s): `planner`, `validator`, `programmer`. |
| `when` | A one-line "call this when…" cue. |
| `examples` | Copy-paste invocation strings. |

Every call returns a `ToolResult` (see `cursus.mcp.envelope`) — a uniform success/error
envelope, so a tool **never raises across the boundary and never prints**:

```python
@dataclass
class ToolResult:
    ok: bool                       # did it succeed?
    data: Any                      # JSON-serializable payload (on success)
    error: str                     # message (on failure)
    code: str                      # machine-readable code: not_found, invalid_input, ...
    warnings: list[str]            # non-fatal notes
    next_steps: list[dict]         # in-band "what to call next" guidance (on success)
    remedy: dict                   # in-band recovery hint (on failure)
    meta: dict                     # side-band info (tool name, counts, timings)
```

`next_steps` and `remedy` are the important part for an agent: the golden path and the
recovery hint travel **on the result**, so an agent doesn't need it baked into a prompt.

### The three lifecycle phases

Tools are tagged by where they fall in the pipeline lifecycle. This is how an agent
picks the right tool without scanning every description:

| Phase | Intent |
|-------|--------|
| `planner` | Discover, select, assemble — explore steps/configs/DAGs and plan the pipeline. |
| `validator` | Check before building — alignment, dependencies, config/DAG integrity, scripts. |
| `programmer` | Compile and generate — turn a resolved DAG into a pipeline / execution doc. |

## The namespaces

The surface is grouped into namespaces. This tutorial focuses on the core path
(`catalog` → `dag` → `config` → `compile` → `author`), but the full set is:

| Namespace | Purpose |
|-----------|---------|
| `catalog` | Discover/search steps, configs, and builders (wraps the step catalog + registry). |
| `dag` | Construct, validate, and serialize pipeline DAGs (`cursus.api.dag`). |
| `config` | Schema-driven config requirements and loading (`cursus.api.factory` + `core.config_fields`). |
| `compile` | Compile/validate/preview a DAG into a SageMaker pipeline (`cursus.core.compiler`). |
| `validate` | Alignment, dependency-resolution, builder, and step-interface checks. |
| `execdoc` | MODS execution-document generation and validation. |
| `pipeline_catalog` | Recommend/select/load pre-built shared DAGs. |
| `project` | Scaffold a new project package (`project.init`, `project.bring_up`). |
| `strategies` | Inspect the builder-strategy library — axes and knobs. |
| `steps` | Per-step connection / I-O view: container paths, property refs, channels. |
| `author` | Guardrails + preflight checks for authoring a new step. |
| `tools` | Meta/discovery over the registry itself (`tools.help`, `tools.by_phase`, `tools.describe_tool`). |

Each non-meta namespace also gets an **auto-generated `<namespace>.help`** tool (e.g.
`compile.help`, `catalog.help`) that returns just that namespace's tools, pre-scoped —
so an agent working in one area can zoom straight in.

For the full, always-current listing, see the generated
[MCP tools reference](../reference/generated/mcp_tools.md).

## Discovering the surface (self-documenting)

The very first thing an agent (or you) should do is orient. There are three ways in.

### 1. From the CLI: `cursus mcp help`

`cursus mcp help` renders exactly what the `tools.help` tool returns — a short overview,
the phase counts, and every tool grouped by namespace:

```bash
# Full overview: every namespace and tool
cursus mcp help

# Zoom into one namespace (auto-shows usage examples)
cursus mcp help --namespace compile

# Only the tools for one lifecycle phase
cursus mcp help --phase validator

# Attach each tool's JSON input schema (call-ready detail)
cursus mcp help --namespace dag --schema

# Machine-readable
cursus mcp help --format json
```

To just list the registered tools (no schemas, no SDK required):

```bash
cursus mcp list-tools
cursus mcp list-tools --namespace catalog
cursus mcp list-tools --format json
```

Both commands are lazy about imports, so `cursus mcp help` works even when the optional
`mcp` SDK is not installed. See the [CLI reference](../cli.rst) for the full command tree.

### 2. From an agent: the `tools.*` meta namespace

The same discovery is available as tools, so an agent can orient itself in-band without
leaving the tool loop:

- **`tools.help`** — START HERE. One call returns the whole toolset: overview, phases,
  and every tool grouped by namespace. Optional `namespace` / `phase` filters and an
  `include_schema` flag zoom from overview down to call-ready detail.
- **`tools.by_phase`** — the tools tagged for one phase (`planner` / `validator` /
  `programmer`).
- **`tools.describe_tool`** — one tool's full descriptor: description, `when` cue,
  examples, JSON schema, tags, and destructive flag.

```python
from cursus.mcp import call_tool

# Orient
overview = call_tool("tools.help", {})
overview.data["total_tools"]          # how many tools exist right now
overview.data["namespaces"]           # each namespace + its tools

# Zoom into a namespace
call_tool("compile.help", {})         # the auto-generated per-namespace help

# Learn one tool's exact arguments before calling it
spec = call_tool("tools.describe_tool", {"name": "compile.dag"})
spec.data["schema"]                   # the JSON input schema
spec.data["examples"]                 # copy-paste invocations
```

## Calling a tool from Python

`call_tool(name, args)` is the whole in-process API. It validates arguments against the
tool's schema, runs the handler, and returns a `ToolResult` — an unknown tool, invalid
arguments, a handled `ToolError`, or an unexpected exception are **all** converted to
error envelopes, so a call never crashes your caller.

```python
from cursus.mcp import call_tool, list_tools, get_registry

# List every tool (optionally filter to one namespace)
for td in list_tools(namespace="catalog"):
    print(td.name, "-", td.description)

# The registry itself is a plain dict you can inspect
reg = get_registry()                        # name -> ToolDef
reg["catalog.list_steps"].schema            # the JSON schema

# Invoke
result = call_tool("catalog.list_steps", {})
if result.ok:
    print(result.data["count"], "steps")
else:
    print("failed:", result.code, result.error)
    print("remedy:", result.remedy)         # in-band recovery hint

# Serialize for an agent transport
result.to_dict()                            # drops empty optional fields
```

## Walkthrough: build a pipeline the way an engineer would

The rest of this tutorial does one concrete task with tools only — take two steps,
wire them, work out the config, and compile — mirroring the manual flow in
[Core concepts](../getting_started/core_concepts.md).

### Step 1 — discover steps (`catalog.*`)

```python
from cursus.mcp import call_tool

# What steps exist?
call_tool("catalog.list_steps", {})
call_tool("catalog.list_steps", {"job_type": "training"})

# Partial name? Fuzzy-search for scored matches.
call_tool("catalog.search", {"query": "xgboost"})

# Full metadata for one step (config/builder names, SageMaker type, framework)
call_tool("catalog.step_info", {"step_name": "XGBoostTraining"})

# The I/O contract — declared dependency ports and output ports (needed to wire edges)
call_tool("catalog.step_spec", {"step_name": "XGBoostTraining"})

# What config fields does it expect?
call_tool("catalog.config_fields", {"step_name": "XGBoostTraining"})
```

`catalog.step_info` gives *naming/component* metadata; `catalog.step_spec` gives the
actual *ports* (dependencies + outputs) so you know what wires into what. Use both
before drawing an edge.

### Step 2 — construct and validate a DAG (`dag.*`)

DAGs cross the tool boundary as plain JSON: `{"nodes": [...], "edges": [[src, dst], ...]}`.
`src` runs before `dst`.

```python
dag_args = {
    "nodes": ["TabularPreprocessing", "XGBoostTraining"],
    "edges": [["TabularPreprocessing", "XGBoostTraining"]],
}

# Build it -> serialized, round-trippable JSON (version, metadata, the serialized `dag`
# holding nodes/edges, and statistics)
built = call_tool("dag.construct", dag_args)

# Validate integrity BEFORE compiling: cycles, dangling edges, isolated nodes,
# and undeclared_edge_nodes (an edge endpoint you never listed in `nodes` — a
# likely typo that would otherwise become a phantom, unconfigured node).
check = call_tool("dag.validate_integrity", dag_args)
check.data["is_valid"]        # True / False
check.data["issues"]          # {category: [messages]}

# Topologically sorted execution plan (order + per-step dependencies)
call_tool("dag.resolve_plan", dag_args)

# Immediate neighbors of one node
call_tool("dag.dependencies", {"step": "XGBoostTraining", **dag_args})

# Persist to a JSON file (validates before writing) — or omit `path` for inline JSON
call_tool("dag.serialize", {**dag_args, "path": "config/dag.json"})
call_tool("dag.deserialize", {"path": "config/dag.json"})
```

The `undeclared_edge_nodes` check is worth calling out: `PipelineDAG` silently
auto-creates any edge endpoint that isn't already a node, so a misspelled edge name would
otherwise spawn a phantom node instead of erroring. `dag.validate_integrity` surfaces it.

### Step 3 — work out the config a DAG needs (`config.*`)

The `config.*` tools do schema-driven **introspection** — they tell you *what* config a
DAG needs. They deliberately expose only the stateless, JSON-clean operations.

```python
dag_json = {
    "dag": {
        "nodes": ["TabularPreprocessing", "XGBoostTraining"],
        "edges": [["TabularPreprocessing", "XGBoostTraining"]],
    }
}

# The primary planning call: base config fields, base processing config fields,
# per-step (non-inherited) fields, the node -> config-class map, and which steps
# still need user input.
reqs = call_tool("config.requirements", dag_json)
reqs.data["config_class_map"]        # node -> config class name
reqs.data["base_config_requirements"]
reqs.data["step_requirements"]       # per node
reqs.data["pending_steps"]           # still need values

# Inspect a single config class in detail (split into required vs optional)
call_tool("config.field_info", {
    "config_class": "XGBoostTrainingConfig",
    "categorized": True,
})

# Summarize a saved merged-config file (field names/counts, not full values)
call_tool("config.load", {"path": "config/config_NA.json"})
```

**Where config generation actually happens.** Actually *writing* the merged `config.json`
requires live, in-process Pydantic config objects — it can't be driven from a stateless
JSON boundary. `config.merge_save` returns an explanatory `unsupported` error that points
you at the in-process engine API (`cursus.core.config_fields.merge_and_save_configs`
built via `DAGConfigFactory`). That in-process authoring is exactly what the shipped
`cursus-configure-pipeline` workflow orchestrates (see below).

### Step 4 — compile to a SageMaker pipeline (`compile.*`)

Compilation takes a DAG (inline `dag` and/or a `dag_file` path) plus a `config_file`.
Building a pipeline definition is **non-destructive**; only `upsert` mutates SageMaker
state, so that path is gated behind a `destructive=True` flag.

```python
common = {"config_file": "config/config_NA.json", "dag_file": "config/dag.json"}

# Validate: can every node resolve to a config and a step builder?
v = call_tool("compile.validate", common)
v.data["summary"]                    # is_valid, missing_configs, unresolvable_builders, ...

# Preview: per-node config/builder resolution with confidence scores + ambiguities
call_tool("compile.preview", common)

# Compile the pipeline definition (build only — NOT pushed to SageMaker)
built = call_tool("compile.dag", {**common, "pipeline_name": "xgboost-training"})
built.data["pipeline_name"], built.data["step_count"], built.data["upserted"]  # upserted=False

# A detailed ConversionReport (per-node resolution_details, avg_confidence) — still build-only
call_tool("compile.with_report", common)

# Re-run ONE failed node in isolation with manual S3 inputs
call_tool("compile.single_node", {
    **common,
    "target_node": "XGBoostTraining",
    "manual_inputs": {"input_path": "s3://bucket/run-1/preprocessing/output/"},
})

# Pure helper: a SageMaker-valid pipeline name (returns {"generated", "generated_is_valid"};
# dots are sanitized to dashes to satisfy SageMaker's name constraints)
call_tool("compile.name", {"base": "xgboost-training"})   # generated -> "xgboost-training-1-0-pipeline"
```

To actually push the pipeline to SageMaker, set `upsert=True` on `compile.dag` and pass a
`role` ARN — this is the one destructive call in the core path, so an agent or server can
gate it behind confirmation:

```python
call_tool("compile.dag", {
    **common,
    "role": "arn:aws:iam::123456789012:role/SageMakerRole",
    "upsert": True,                  # DESTRUCTIVE — mutates SageMaker state
})
```

Notice the `compile.validate` result carries `next_steps` pointing at `compile.dag` when
the DAG is valid, or at `compile.preview` / `catalog.search` when it isn't — the tool
hands the agent its own next move.

## Authoring a brand-new step (`author.*`)

Authoring a step is different from configuring one: you write **one `.step.yaml` + one
config class + one script**, and the builder and registry are derived by construction —
writing the `.step.yaml` *is* the registration. The `author.*` namespace gives an agent
the SOP, the guardrails, and offline constructibility proofs, all sourced from the live
enforcement objects so guidance can never drift from what CI enforces. There is **no
code generator** — the agent writes the files itself; these tools tell it what's legal and
prove it worked.

```python
# START HERE: the ordered author -> validate -> integrate SOP as DATA, naming the exact
# tool to call at each step. The routing branch (bound handler + exemplar) is derived
# from the live strategy registry.
call_tool("author.checklist", {"sagemaker_step_type": "Training"})
call_tool("author.checklist", {"sagemaker_step_type": "Processing", "step_assembly": "step_args"})

# The restriction set for one topic, read off the live enforcement objects
call_tool("author.rules", {"topic": "naming"})       # PascalCase / Config / StepBuilder / valid step types
call_tool("author.rules", {"topic": "packaging"})    # source_dir + SAIS preamble
call_tool("author.rules", {"topic": "closure"})      # registry-by-construction + contract<->spec alignment
# topics: naming | packaging | sdk_carveout | reuse_class | closure

# Config VALUE guidance (allowed_values + case-sensitivity + required-no-default fields)
call_tool("author.config_constraints", {"step_name": "TabularPreprocessing"})

# Validate concrete config VALUES against the live config class (model_validate)
call_tool("author.preflight_config", {
    "step_name": "TabularPreprocessing",
    "values": {"job_type": "training", "label_name": "label"},
})

# Check the SCRIPT against its contract BOTH ways (args parsed + required env vars read)
call_tool("author.check_script", {"step_name": "TabularPreprocessing"})

# Prove the step is CONSTRUCTIBLE (binds + synthesizes) — the same four gates CI runs
call_tool("author.preflight_step", {"step_name": "TabularPreprocessing"})
call_tool("author.preflight_step", {"all": True})    # the whole CI merge gate
```

`author.preflight_step` runs a flat list of the four merge gates — interface validation,
registry derivation/parity, `RegistryBindingValidator` B3 (handler binds + builder
synthesizes + config-field coverage), and `resolve_strategy` routability. SDK-delegation
and no-builder rows report *skip-not-error* offline, so a step is judged CONSTRUCTIBLE,
not merely parseable, before a code review. Related namespaces the checklist calls into
include `strategies.for_step_type` (which handler + knobs bind for a step type) and
`steps.io` (a step's wired inputs/outputs/env/job-args).

## Running the MCP server

Everything above works in-process with no server. To expose the same tools to an external
agent over the Model Context Protocol, run the stdio server. It requires the **optional**
`mcp` SDK (the tool functions, schemas, and registry all work without it — only this
adapter needs it):

```bash
# via the CLI
cursus mcp serve

# or directly
python -m cursus.mcp.server
```

If the SDK is missing, both paths raise a clear, actionable error telling you to
`pip install mcp`. The server (`cursus.mcp.server.build_server`) generates its
`list_tools` from the registry and routes `call_tool` straight through
`cursus.mcp.registry.call_tool`, so the server, the CLI, and in-process callers all share
**one code path and one result contract**. Each tool's `when` cue and `examples` are
folded into the description string it advertises (`render_description`), so an external
MCP client sees the usage guidance too.

Building a tool list for other agent frameworks uses the same registry:

```python
from cursus.mcp import export_mcp_tools, export_openai_tools

export_mcp_tools()        # MCP list_tools shape: name / description / inputSchema
export_openai_tools()     # OpenAI / Claude function-calling shape
```

## The shipped dynamic-workflow orchestrators

The tools are the primitives; Cursus also ships **saved orchestration scripts** that
sequence them into deterministic, resumable, gated flows. They live in
`src/cursus/mcp/workflows/` (with a full README) and are JavaScript scripts for the
Claude Code `Workflow` runtime — reference/runnable artifacts, not importable Python. The
two central ones:

- **`cursus-configure-pipeline.js`** — the common job: produce a pipeline `config.json`
  for a DAG of **existing** step types. It drives an agent to author (or repair) a
  project's `generate_config.py` (the `DAGConfigFactory` pattern), gating each node with
  `author.config_constraints` + `author.preflight_config`, then running whole-graph checks
  (`compile.validate`, `compile.preview`, `validate.deps_resolve`) that catch the
  cross-node failures per-node validation misses.
- **`cursus-author-step.js`** — DAG-driven creation of a **new** step type. The input is a
  DAG with exactly one new, unregistered node between a producer and a consumer; the
  workflow enforces spec-alignment on both edges as non-skippable gates, calling
  `author.checklist` + `strategies.for_step_type` + `steps.io` + `validate.deps_resolve` +
  `validate.step_interface` at each stage, and short-circuits rather than authoring a step
  that shouldn't exist.

Two more compose on top: `cursus-init-project.js` (scaffold a phase-0 project — also
available statelessly as the `project.init` tool) and `cursus-new-project.js` (end-to-end
bring-up, also reachable via `project.bring_up`). Each stage calls the cursus tools as MCP
tools, falling back to the `cursus` CLI if the server is unreachable.

To use one, copy the script into a Claude Code workflows location
(`.claude/workflows/` in a project, or `~/.claude/workflows/`) so it runs as, e.g.,
`/cursus-configure-pipeline`, or invoke the `Workflow` tool with its `scriptPath`
directly. The `src/cursus/mcp/workflows/README.md` documents each stage, its arguments,
and the gates it enforces.

## Recap

- The registry (`cursus.mcp.registry`) is the single source of truth; the CLI, the
  in-process `call_tool`, and the MCP server all read it.
- Start with `cursus mcp help` (CLI) or `tools.help` (agent) — the surface documents
  itself, down to per-tool JSON schemas and examples.
- The core path is `catalog.*` → `dag.*` → `config.*` → `compile.*`; authoring adds
  `author.*`. Building is non-destructive; only `compile.dag`'s `upsert` mutates state.
- Every call returns a `ToolResult` with `next_steps` (golden path) or `remedy`
  (recovery) travelling in-band.
- `cursus mcp serve` / `python -m cursus.mcp.server` exposes the identical tools over MCP.
- Shipped workflow scripts orchestrate the tools into gated, resumable flows.

## See also

- [MCP tools reference](../reference/generated/mcp_tools.md) — the full, generated tool listing.
- [Step catalog reference](../reference/generated/step_catalog.md) — the steps `catalog.*` discovers.
- [Pipeline catalog reference](../reference/generated/pipeline_catalog.md) — the shared DAGs `pipeline_catalog.*` serves.
- [CLI reference](../cli.rst) — `cursus mcp help / list-tools / serve`.
- [API reference](../api/index.rst) — `cursus.mcp` modules.
- [Core concepts](../getting_started/core_concepts.md) — the manual flow these tools mirror.
