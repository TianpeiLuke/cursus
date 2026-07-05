# Core Concepts

A quick tour of the pieces you'll meet in Cursus. Each links to deeper material in the
[Concepts](../concepts/index.md) section.

## The DAG

A `PipelineDAG` is the graph you author: **nodes are step names**, **edges are data
dependencies** ("this step's output feeds that step's input"). It is intentionally thin —
it says *what* runs and *in what order*, not *how*. The DAG can be built in Python
(`add_node` / `add_edge`), loaded from JSON (`cursus dag`), or pulled ready-made from the
[pipeline catalog](../reference/generated/pipeline_catalog.md).

## The config

A **config JSON** supplies each node's settings and, crucially, a
`metadata.config_types` map from each DAG node to a **configuration class** (a Pydantic
model such as `XGBoostTrainingConfig`). The config is the single source of truth for a
step's parameters; the compiler reads it to instantiate steps.

## Step interfaces

Every step type is described by one declarative **`<step>.step.yaml`** interface that
unifies two things: the script **contract** (input/output paths, environment variables,
CLI job arguments) and the **spec** (typed dependencies and outputs used for dependency
resolution). Builders are synthesized from these interfaces at runtime — there are no
hand-written builder classes to keep in sync. Browse them all in the
[Step Interface Catalog](../reference/generated/step_catalog.md).

## The registry

The **registry** is the canonical table of step names and their associated config
classes, builders, and SageMaker step types. It is *derived* from the step interfaces
(interface-first discovery), and it is **extensible**: a consumer can define its own steps
in a folder outside the installed package (a "step pack") and have Cursus discover them
as native. The overlay is strictly additive: a pack can only *add* steps — the built-in
steps are never removed or replaced. On a deliberate name clash the pack step shadows the
built-in ("plugin-wins") and a warning is emitted so the collision is visible.

## Compilation

The **compiler** (`PipelineDAGCompiler`) is where it comes together. Given a DAG and a
config, it: resolves the dependencies between nodes, looks up each node's interface and
synthesized builder, constructs the concrete SageMaker step objects, and assembles them
into a `sagemaker.workflow.pipeline.Pipeline`. Along the way it validates that the DAG and
config actually agree (missing configs, unresolvable builders, and dependency gaps are
reported before you deploy).

## The pipeline catalog

The **pipeline catalog** is a library of 44 pre-built, validated DAGs indexed by
framework, task type, and complexity. Its router (`recommend_dag` / `load_shared_dag` /
`build_and_compile`) lets you start from a proven pipeline instead of a blank graph.

## The agent surface (MCP)

Everything above is also exposed as a framework-neutral **MCP tool surface** (70 tools)
so an LLM agent can discover steps, construct and validate DAGs, generate configs,
compile, and even author new steps — the same operations as the CLI/API, as JSON-in /
JSON-out tools. See the [MCP Tool Reference](../reference/generated/mcp_tools.md).

---

Ready for the details? Continue to [Concepts](../concepts/index.md), or jump to the
[Step](../reference/generated/step_catalog.md) / [Pipeline](../reference/generated/pipeline_catalog.md) /
[MCP](../reference/generated/mcp_tools.md) references.
