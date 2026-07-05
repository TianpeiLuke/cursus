# Quickstart

This walkthrough builds a SageMaker pipeline three ways — from the **pipeline catalog**,
from the **CLI**, and from the **Python API**. Pick whichever fits how you work; they all
compile through the same engine.

Every path needs two ingredients:

- a **DAG** — the graph of step names and the edges between them, and
- a **config** — a JSON file whose `metadata.config_types` maps each DAG node to a
  configuration class, plus each node's settings.

Compilation is offline. You only need AWS credentials to *deploy* (`--upsert`) or *run*
(`--start`) the result.

## Path 1 — Start from the pipeline catalog (recommended)

Cursus ships **44 pre-built, dependency-resolvable DAGs** across 8 frameworks. Let the
router recommend one, see what config it expects, then build it.

```bash
# 1. List the frameworks the catalog covers
cursus pipeline-catalog list

# 2. Recommend a DAG for your framework/task
cursus pipeline-catalog recommend --framework xgboost

# 3. Inspect a specific DAG (nodes, edges, required config)
cursus pipeline-catalog get-dag xgboost_complete_e2e
```

```python
from cursus.pipeline_catalog import recommend_dag, load_shared_dag
from cursus import PipelineDAGCompiler

# recommend_dag returns a ranked list of matches (dicts with 'id', 'score', ...)
recommendations = recommend_dag(framework="xgboost", task_type="end_to_end")
dag = load_shared_dag(recommendations[0]["id"])

pipeline, report = PipelineDAGCompiler(config_path="config.json").compile_with_report(dag)
print(pipeline.name, "-", len(pipeline.steps), "steps")
print("avg confidence:", report.avg_confidence)
```

Browse the whole catalog in the [Pipeline Catalog reference](../reference/generated/pipeline_catalog.md).

## Path 2 — Compile from the CLI

If you already have a DAG JSON and a config JSON (e.g. checked into a repo), the CLI is
the reproducible, no-glue path:

```bash
# compile only (writes the SageMaker pipeline definition to a file)
cursus compile -d my_dag.json -c my_config.json -o pipeline.json

# validate DAG <-> config alignment without compiling
cursus compile -d my_dag.json -c my_config.json --validate-only

# compile AND deploy to SageMaker, then start an execution
cursus compile -d my_dag.json -c my_config.json \
    --upsert --start --role arn:aws:iam::123456789012:role/MySageMakerRole
```

Not sure what a DAG or config file looks like? Generate scaffolding and inspect requirements:

```bash
cursus dag --help          # validate / resolve dependency edges in DAG JSON
cursus config --help       # inspect the config a DAG requires
cursus projects list       # list pipeline projects under a root directory
```

The full command tree — every group, option, and argument — is in the
[CLI reference](../cli.rst).

## Path 3 — Build a DAG in Python

For full programmatic control, construct the DAG and drive the compiler directly:

```python
from cursus import PipelineDAGCompiler
from cursus.api.dag import PipelineDAG

# 1. Describe the graph: nodes are step names, edges are data dependencies
dag = PipelineDAG()
for node in ["CradleDataLoading", "TabularPreprocessing", "XGBoostTraining"]:
    dag.add_node(node)
dag.add_edge("CradleDataLoading", "TabularPreprocessing")
dag.add_edge("TabularPreprocessing", "XGBoostTraining")

# 2. Compile against a config file (which maps each node -> a config class)
compiler = PipelineDAGCompiler(
    config_path="config.json",
    anchor_file=__file__,   # anchors docker source_dir resolution to your project
)

# 3. Validate, then compile with a detailed report
result = compiler.validate_dag_compatibility(dag)
if not result.is_valid:
    raise SystemExit(result)

pipeline, report = compiler.compile_with_report(dag)
print(pipeline.name, "compiled:", len(pipeline.steps), "steps")

# 4. Deploy / run when ready
# pipeline.upsert(role_arn="arn:aws:iam::...:role/MySageMakerRole")
# pipeline.start()
```

## What just happened?

Cursus resolved the **dependencies** between your steps (which output of
`TabularPreprocessing` feeds `XGBoostTraining`), looked up each step's declarative
**interface** (`<step>.step.yaml`) and **builder** to construct the right SageMaker step
type, and assembled them into a `sagemaker.workflow.pipeline.Pipeline`.

## Where to go next

- [Core concepts](core_concepts.md) — the DAG, configs, the registry, and how compilation works.
- [Step Interface Catalog](../reference/generated/step_catalog.md) — every step you can put in a DAG.
- [CLI reference](../cli.rst) and [API reference](../api/index.rst) — the complete surfaces.
