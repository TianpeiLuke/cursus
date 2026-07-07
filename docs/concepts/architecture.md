# System Architecture

Cursus turns a **pipeline graph plus a set of step configurations** into a
production-ready **SageMaker pipeline**. This page is the map: it names the
layered subsystems, shows how a request flows from a DAG to a compiled
`sagemaker.workflow.pipeline.Pipeline`, and points to the deeper concept pages
for each layer.

If you only remember one sentence: **a DAG names the steps and their edges, the
configs supply the parameters, and the compiler + assembler + dependency
resolver wire them into a real SageMaker pipeline — everything else (the step
library, the catalog, the registry, validation, MCP, CLI) exists to make that
one transformation discoverable, correct, and automatable.**

```{figure} ../images/cursus_system_structure.png
:alt: Cursus subsystems — a Pipeline DAG feeds the Compiler API and Assembler, which use the Step Catalog/Registry and Step Library, the Dependency Resolver, and the Validation System.
:width: 100%

The Cursus subsystems. A **Pipeline DAG** enters the **Compiler API + Assembler** (1);
the **Step Catalog** and **Step Registry** resolve each node to a step from the **Step
Library** (2); the **Dependency Resolver** (3) wires producer outputs to consumer
inputs; and the **Validation System** (4) checks alignment and runtime behavior.
```

## The 10,000-foot view

```
                        YOU AUTHOR
        ┌────────────────────────┐        ┌───────────────────────────┐
        │  PipelineDAG            │        │  config.json / .yaml      │
        │  nodes + edges          │        │  one config per DAG node  │
        │  (api/dag)              │        │  (steps/configs)          │
        └───────────┬────────────┘        └─────────────┬─────────────┘
                    │                                    │
                    └──────────────┬─────────────────────┘
                                   ▼
              ┌──────────────────────────────────────────────┐
              │  PipelineDAGCompiler        (core/compiler)   │
              │   • build a DynamicPipelineTemplate           │
              │   • detect required config classes            │
              └───────────────────────┬──────────────────────┘
                                      ▼
              ┌──────────────────────────────────────────────┐
              │  DynamicPipelineTemplate    (core/compiler)   │
              │   • node  -> config     (StepConfigResolver)  │
              │   • node  -> builder    (StepCatalog)         │
              └───────────────────────┬──────────────────────┘
                                      ▼
              ┌──────────────────────────────────────────────┐
              │  StepCatalog / registry  (step_catalog +      │
              │  registry)                                    │
              │   • load_builder_class("XGBoostTraining")     │
              │   • builder SYNTHESIZED at runtime from the   │
              │     step's .step.yaml  (no builder_*.py)      │
              └───────────────────────┬──────────────────────┘
                                      ▼
              ┌──────────────────────────────────────────────┐
              │  PipelineAssembler          (core/assembler)  │
              │   • builder.create_step(**kwargs) per node    │
              │   • wire edges via dependency resolution      │
              └───────────────────────┬──────────────────────┘
                                      │ uses
                                      ▼
              ┌──────────────────────────────────────────────┐
              │  UnifiedDependencyResolver  (core/deps)       │
              │   • match one step's outputs to the next      │
              │     step's declared inputs (semantic match)   │
              │   • emit PropertyReference wiring             │
              └───────────────────────┬──────────────────────┘
                                      ▼
                        ┌──────────────────────────┐
                        │  sagemaker ...Pipeline    │
                        │  pipeline.definition()    │
                        │  pipeline.start()         │
                        └──────────────────────────┘

  CROSS-CUTTING (used at every stage, not in the linear flow):
   • steps/interfaces  — the .step.yaml contract+spec for each step type
   • pipeline_catalog  — pre-built DAGs you can load instead of hand-building
   • validation        — proves scripts/interfaces/wiring line up (alignment)
   • mcp               — the same engine exposed as JSON tools for LLM agents
   • cli               — the same engine exposed as `cursus` subcommands
```

## Subsystem responsibilities

| Subsystem | Responsibility | Package path |
|-----------|----------------|--------------|
| **DAG / graph model** | The topology data structure: nodes, edges, topological sort, JSON (de)serialization. | `src/cursus/api/dag` (`PipelineDAG`, `PipelineDAGResolver`, `export_dag_to_json`) |
| **Compiler** | Entry point. Wraps a DAG + config into a `DynamicPipelineTemplate`, resolves each node to a config class and a builder, and produces a `Pipeline`. | `src/cursus/core/compiler` (`PipelineDAGCompiler`, `DynamicPipelineTemplate`, `compile_dag_to_pipeline`) |
| **Assembler** | Instantiates each step builder, calls `create_step(...)`, and stitches the steps together into the SageMaker `Pipeline`. | `src/cursus/core/assembler` (`PipelineAssembler`, `PipelineTemplateBase`) |
| **Dependency resolution** | Semantic matching of one step's declared outputs to the next step's declared inputs; produces `PropertyReference` wiring. | `src/cursus/core/deps` (`UnifiedDependencyResolver`, `SemanticMatcher`, `RegistryManager`, `PropertyReference`) |
| **Base interfaces** | The shared abstractions: `BasePipelineConfig`, `StepBuilderBase`, `StepInterface` (unified contract+spec), the `TemplateStepBuilder` facade + `PatternHandler` strategies, and enums. | `src/cursus/core/base` (`config_base.py`, `builder_base.py`, `step_interface.py`, `builder_templates.py`, `enums.py`) |
| **Step catalog** | Discovery. Given a step name, find its config class, interface, script, and (synthesized) builder across the package and any workspaces. O(1) lookups. | `src/cursus/step_catalog` (`StepCatalog`, `builder_discovery.py`, `config_discovery.py`) |
| **Registry** | The canonical step-name table (`STEP_NAMES`) and the construction-strategy routing table — both *derived* from the `.step.yaml` interfaces, not hand-maintained. | `src/cursus/registry` (`step_names.py`, `interface_registry_loader.py`, `strategy_registry.py`) |
| **Step library** | The actual step types: one declarative `.step.yaml` interface + one config class + (usually) one script per step. | `src/cursus/steps/interfaces`, `src/cursus/steps/configs`, `src/cursus/steps/scripts` |
| **Pipeline catalog** | Pre-built, reusable DAGs (`*.dag.json`) plus a router that recommends/loads them and helpers that compile them. | `src/cursus/pipeline_catalog` (`shared_dags/`, `core/`) |
| **Validation** | Alignment testing — proves that a step's script, interface, and registry binding line up, and that DAG edges are resolvable. | `src/cursus/validation` (`validation/alignment`) |
| **MCP** | The whole engine exposed as framework-neutral JSON-in/JSON-out tools for LLM agents. | `src/cursus/mcp` (`registry.py`, `tools/`, `server.py`) |
| **CLI** | The whole engine exposed as `cursus` subcommands for humans. | `src/cursus/cli` (`compile_cli.py`, `dag_cli.py`, `validate_cli.py`, …) |

## The request flow, step by step

### 1. You author two things

A **DAG** (the shape) and a **config file** (the parameters). The DAG is a
`PipelineDAG` of node names and directed edges:

```python
from cursus.api.dag import PipelineDAG

dag = PipelineDAG()
dag.add_node("CradleDataLoading")
dag.add_node("TabularPreprocessing")
dag.add_node("XGBoostTraining")
dag.add_edge("CradleDataLoading", "TabularPreprocessing")
dag.add_edge("TabularPreprocessing", "XGBoostTraining")
```

Node names correspond to step types in the [step catalog](../reference/generated/step_catalog.md).
The config file carries one configuration object per node (loaded and matched by
`StepConfigResolver`). You don't have to hand-build the DAG — the
[pipeline catalog](../reference/generated/pipeline_catalog.md) ships ready-made
`*.dag.json` graphs (`load_shared_dag`, `recommend_dag`).

### 2. The compiler wraps it in a template

`PipelineDAGCompiler` (`core/compiler/dag_compiler.py`) is the top-level entry
point. The one-call convenience function is `compile_dag_to_pipeline` (also
exported as `cursus.compile_dag`):

```python
from cursus.core.compiler import PipelineDAGCompiler

compiler = PipelineDAGCompiler(config_path="config.json")
pipeline = compiler.compile(dag, pipeline_name="fraud-detection")

# Or, to get a diagnostics report alongside the pipeline:
pipeline, report = compiler.compile_with_report(dag)
```

Internally the compiler builds a `DynamicPipelineTemplate`
(`core/compiler/dynamic_template.py`), a subclass of `PipelineTemplateBase` that
implements the template's abstract methods *dynamically* — it works for any DAG
rather than being hand-written per pipeline. On construction it auto-detects the
config classes the DAG requires (`_detect_config_classes`).

### 3. The template resolves node -> config and node -> builder

For each DAG node the template answers two questions:

- **Which config?** `StepConfigResolver`
  (`step_catalog/adapters/config_resolver.py`, re-exported as
  `core.compiler.StepConfigResolver`) matches the node name to a loaded config
  object from your config file.
- **Which builder?** The `StepCatalog` loads the builder class:
  `StepCatalog().load_builder_class("XGBoostTraining")`.

### 4. The catalog synthesizes the builder

This is where the 2.0.0 **classless-factory** design shows up. There is no
`builder_xgboost_training_step.py` file. Instead, for any registry step that has
a `.step.yaml` interface, `step_catalog/builder_discovery.py` **synthesizes** a
builder class at runtime — `type(f"{Name}StepBuilder", (TemplateStepBuilder,),
{"STEP_NAME": Name})` — and caches it per process. Every step builder is the
same shared `TemplateStepBuilder` facade (`core/base/builder_templates.py`)
whose construction methods (`create_step`, `_get_inputs`, `_get_outputs`,
`_get_environment_variables`) delegate to one of five `PatternHandler`
strategies:

| Handler | Builds |
|---------|--------|
| `ProcessingHandler` | Processing steps (both the `code=` and `processor.run()`→`step_args` assembly modes) |
| `TrainingHandler` | Training steps |
| `ModelCreationHandler` | CreateModel steps |
| `TransformHandler` | Batch Transform steps |
| `SDKDelegationHandler` | MODS/SAIS predefined steps (Cradle / Redshift / Registration / DataUploading) |

The right handler is chosen by `resolve_handler(sagemaker_step_type,
step_assembly, knobs)` (defined in `core/base/builder_templates.py`), which in
turn consults the strategy table via `resolve_strategy(axis, name)` in
`registry/strategy_registry.py`. Per-step behavior that isn't shared lives as
declarative **knobs** in the `.step.yaml` `patterns:` section — not as overridden
Python. There is no `cursus.steps.builders` module and no `builder_*.py` files;
the only way to obtain a builder is through the catalog
(`StepCatalog().load_builder_class("XGBoostTraining")`), which returns the
synthesized `TemplateStepBuilder` subclass.

The registry itself (`STEP_NAMES` and its derived maps `CONFIG_STEP_REGISTRY`,
`BUILDER_STEP_NAMES`, `SPEC_STEP_TYPES`) is likewise *derived by construction*
from the `.step.yaml` `registry:` blocks
(`registry/interface_registry_loader.py::build_registry_from_interfaces`) rather
than read from a standalone table — so the registry can never drift from the
interface files.

### 5. The assembler builds and wires the steps

`PipelineAssembler` (`core/assembler/pipeline_assembler.py`) takes the resolved
`{node: builder}` and `{node: config}` maps and, in topological order:

1. Instantiates each builder and calls `builder.create_step(**kwargs)` to
   produce a real SageMaker step. The call site has **no `isinstance` gate**, so
   a runtime-synthesized builder is indistinguishable from a hand-written one.
2. For each DAG edge, asks the dependency resolver how the upstream step's
   outputs satisfy the downstream step's declared inputs.

### 6. The dependency resolver connects the edges

`UnifiedDependencyResolver` (`core/deps/dependency_resolver.py`) does the
intelligent wiring. Each step's `.step.yaml` declares its **dependencies**
(what it needs) and **outputs** (what it produces), including `compatible_sources`
and `property_path`. The resolver's `SemanticMatcher` matches an upstream
`OutputDecl` to a downstream `DependencyDecl` by logical name/alias and type,
and emits a `PropertyReference` — the SageMaker runtime handle that carries one
step's output into the next step's input. `RegistryManager` /
`SpecificationRegistry` scope these specs so multiple pipelines don't collide.

Because the wiring keys entirely on `.step.yaml` **spec data** carried on
`builder.spec` (plus the DAG node name) and never on a Python class, collapsing
the former per-step `*StepBuilder` classes into one facade leaves every edge
intact.

### 7. Out comes a SageMaker pipeline

The assembler returns a `sagemaker.workflow.pipeline.Pipeline`. You can inspect
`pipeline.definition()` or run `pipeline.start()`. The public compiler API
(`PipelineDAGCompiler`, `compile_dag_to_pipeline`, `compile_with_report`) is
stable across the 2.0.0 rewrite: the same DAG + the same config compiles to the
same pipeline.

## The unified step interface (`.step.yaml`)

The keystone of the architecture is the single declarative interface per step,
introduced in 1.8.0 and completed in 2.0.0. Before 1.8.0 a step needed a
`*_contract.py` (script I/O) **and** a `*_spec.py` (dependency demand/supply)
**and** a `builder_*.py`. Now a step is:

```
steps/interfaces/xgboost_training.step.yaml   # contract + spec + registry block
steps/configs/config_xgboost_training_step.py # the typed config class
steps/scripts/xgboost_training.py             # the script (if any)
```

The `.step.yaml` is parsed into a single Pydantic `StepInterface`
(`core/base/step_interface.py`) that is a superset of the old `ScriptContract`
and `StepSpecification`. It exposes:

- `.contract` — `entry_point`, `expected_input_paths`, `expected_output_paths`,
  `expected_arguments`, `required_env_vars`, etc.
- `.spec` — `step_type`, `node_type`, `dependencies`, `outputs`, and helpers like
  `list_required_dependencies()` / `get_output_by_name_or_alias()`.

Contract↔spec alignment (inputs match dependency keys, outputs match output
keys, paths are valid SageMaker paths) is enforced as a **construction-time
Pydantic invariant** — a bad interface fails to load, rather than failing a
separate validation tier. This one object is the message passed between the
dependency resolver, the builder, and the assembler.

## Cross-cutting subsystems

These aren't stages in the linear flow; they wrap or observe it.

### Validation (alignment)

`validation/alignment` proves the pieces line up. In the 2.0.0 boundary model,
validation checks the three things construction *can't* self-verify:

- **B1 — script↔interface fidelity** (AST analysis of the script against its
  `.step.yaml`).
- **B2 — cross-step DAG resolvability** (every edge's inputs are satisfiable,
  including the SageMaker property-path check).
- **B3 — registry↔handler↔config binding**
  (`validation/alignment/validators/registry_binding_validator.py`): the handler
  resolves, the builder loads/synthesizes, and the config covers the fields the
  handler reads. If B3 passes, the step is constructible.

Run it via `cursus alignment` — the Unified Alignment Tester (see the
[CLI](../cli.rst)). (`cursus validate` covers the complementary local
script-testing and author-time `.step.yaml` checks.)

### Pipeline catalog

`pipeline_catalog` is a queryable store of pre-built DAGs (`shared_dags/*.dag.json`,
indexed by `catalog_index.json`) plus:

- `router.py` — `recommend_dag` / `auto_select_dag` scoring over the index,
- `core/builders.py` — `build_and_compile` (SAIS) and `build_mods_pipeline`
  (generates a MODS template class on demand),
- `load_shared_dag` / `search_dags` — load or search by feature/framework.

So instead of hand-building a DAG you can `load_shared_dag(...)` and pass it
straight to the compiler. See the
[pipeline catalog reference](../reference/generated/pipeline_catalog.md).

### MCP (agent surface)

`cursus.mcp` exposes the engine as JSON-in / JSON-out **tools** grouped into
namespaces (`catalog`, `dag`, `config`, `compile`, `validate`, `execdoc`,
`pipeline_catalog`, `steps`, `strategies`, plus `author`, `project`, and a
`tools` meta-namespace for tool introspection). It is framework-neutral: the same
`ToolDef` registry (`mcp/registry.py`, a `name -> ToolDef` dict) drives an MCP
server (`mcp/server.py`), OpenAI function-calling, and Claude tool use.

```python
from cursus.mcp import get_registry, call_tool

result = call_tool("catalog.search", {"query": "xgboost"})
result.ok    # True / False
result.data  # JSON-serializable payload
```

Each call returns a `ToolResult` envelope (`mcp/envelope.py`) with `.ok`,
`.data`, `.error`, `.remedy`, and `.next_steps` fields. Argument schemas are
validated per tool — for example `catalog.list_steps` accepts only
`workspace_id` / `job_type` filters, while `catalog.search` requires a `query`.

Because a step is now one `.step.yaml` + one config (no builder file), an agent
can author a step and configure a pipeline entirely by tool-calling. The optional
stdio server (`pip install "cursus[mcp]"`, then `cursus-mcp`) is **read-only by
default** — state-changing tools are opt-in via env vars — and exposes host-legal
tool names (dots become `__`) with per-tool safety annotations. See
[the agent (MCP) tool surface](mcp_surface.md) and the
[MCP tools reference](../reference/generated/mcp_tools.md).

### CLI

`cursus.cli` is the human counterpart to MCP — the same capabilities as `cursus`
subcommands (`compile`, `dag`, `config`, `validate`, `alignment`, `exec-doc`,
`pipeline-catalog`, `projects`, `catalog`, `registry`, `steps`, `strategies`,
`mcp`). See the [CLI reference](../cli.rst).

## How the layers depend on each other

```
   cli ─┐                      ┌─ mcp
        └──────────┬───────────┘
                   ▼
        core/compiler  ──▶  core/assembler  ──▶  core/deps
                   │                 │                 │
                   ▼                 ▼                 ▼
             step_catalog ◀───▶  registry  ◀───▶  core/base
                   │                 │                 │
                   ▼                 ▼                 ▼
       steps/{interfaces, configs, scripts}   +   pipeline_catalog
                                     ▲
                                     │
                              validation (alignment)
```

- `core/base` is the foundation everyone imports (`StepInterface`,
  `StepBuilderBase`, `BasePipelineConfig`, the `TemplateStepBuilder` facade,
  enums). It depends on nothing else in this list.
- `step_catalog` + `registry` are the discovery/derivation layer: they read the
  `steps/interfaces/*.step.yaml` files and expose lookups.
- `core/compiler`, `core/assembler`, `core/deps` are the runtime engine.
- `cli` and `mcp` are two thin surfaces over that engine; `pipeline_catalog`
  feeds it pre-built DAGs; `validation` observes it.

## Where to go next

- [DAG and compilation](../concepts/dag_and_compilation.md) — the graph model
  and the compile flow in depth.
- [API reference](../api/index.rst) — the classes and functions named here.
- [Step catalog](../reference/generated/step_catalog.md) — the full list of step
  types.
- [Pipeline catalog](../reference/generated/pipeline_catalog.md) — the pre-built
  DAGs.
- [MCP tools](../reference/generated/mcp_tools.md) — the agent tool surface.
- [CLI](../cli.rst) — the `cursus` command groups.
