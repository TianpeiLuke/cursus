# Build & Compile Your First Pipeline

This tutorial walks you end-to-end through the core Cursus workflow: you will
**describe a pipeline as a small DAG**, **write a matching configuration**, and
**compile it into a SageMaker `Pipeline` object** — first with the `cursus compile`
CLI, then with the Python API (`PipelineDAG` + `PipelineDAGCompiler`).

We build a three-step topology that most tabular ML pipelines share:

```
data load  ─▶  preprocess  ─▶  train
```

By the end you will understand the two input artifacts (a **DAG JSON** and a
**config JSON**), how Cursus resolves each DAG node to a config class and a step
builder, how to read the **compilation report**, and how to (optionally) deploy the
compiled pipeline to SageMaker.

```{contents}
:local:
:depth: 2
```

## What "compile" actually means

Cursus separates *structure* from *configuration*:

| Artifact | What it holds | Who owns it |
|----------|---------------|-------------|
| **DAG** | The graph — which steps exist (`nodes`) and how they depend on each other (`edges`) | You, the pipeline author |
| **Config** | The per-step settings — instance types, hyperparameters, S3 locations, job types | You, plus shared defaults |
| **Step catalog** | The registry of step *types* and their builders | Cursus (and any step packs you add) |

Compilation is the process of **matching each DAG node to a config**, **matching each
config to a step builder**, wiring the data dependencies, and emitting a
`sagemaker.workflow.pipeline.Pipeline`. That work is driven by
`PipelineDAGCompiler` (see `src/cursus/core/compiler/dag_compiler.py`). For the
conceptual model behind this, see [DAG & compilation](../concepts/dag_and_compilation.md).

## Prerequisites

- Cursus installed (`pip install cursus[all]`) — see [Installation](../getting_started/installation.md).
- The `cursus` CLI on your `PATH` (verify with `cursus --version`).
- For the compile-only and validate steps you do **not** need AWS credentials. You
  only need them for the optional deploy step at the very end (`--upsert` / `--start`).

## Step 1 — Describe the DAG

A `PipelineDAG` (`src/cursus/api/dag/base_dag.py`) is just a set of node names and a
list of directed edges. Two methods do all the work:

- `add_node(name)` — declare a step by name.
- `add_edge(src, dst)` — declare that `dst` depends on `src`.

### Node naming matters

Cursus resolves a node to a config partly by its **name**. The convention is
`<StepType>` or `<StepType>_<job_type>`, where `<StepType>` is a real step type from
the [step catalog](../reference/generated/step_catalog.md). We use three real step
types here:

| Node | Step type | Job type | Role |
|------|-----------|----------|------|
| `CradleDataLoading_training` | `CradleDataLoading` | `training` | load data |
| `TabularPreprocessing_training` | `TabularPreprocessing` | `training` | preprocess |
| `XGBoostTraining` | `XGBoostTraining` | — | train |

Naming a node after its step type lets the resolver bind it with high confidence, and
lets validation cross-check that the bound config really is that step type. (If you
deliberately use an off-convention name, you bind it explicitly via
`metadata.config_types` — covered below.)

### Build it in Python and serialize

```python
from cursus.api.dag import PipelineDAG, export_dag_to_json

dag = PipelineDAG()
dag.add_node("CradleDataLoading_training")
dag.add_node("TabularPreprocessing_training")
dag.add_node("XGBoostTraining")

dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")

# Optional: catch typos before you compile. add_edge auto-creates missing
# endpoints in lenient mode, so a misspelled edge name silently spawns a
# phantom node. This surfaces any endpoint you never declared with add_node.
undeclared = dag.validate_node_declarations()
assert not undeclared, f"Undeclared edge endpoints (likely typos): {undeclared}"

export_dag_to_json(dag, "dag.json")
```

```{tip}
Pass `strict=True` to `PipelineDAG(...)` to turn undeclared edge endpoints into an
immediate `ValueError` at `add_edge` time, instead of auto-creating a phantom node.
```

### The DAG JSON shape

`export_dag_to_json` (backed by `PipelineDAGWriter`) writes a versioned document. The
only part the compiler reads is `dag.nodes` and `dag.edges`; the rest is metadata and
computed statistics:

```json
{
  "version": "1.0.0",
  "created_at": "2026-06-26T00:00:00+00:00",
  "metadata": {},
  "dag": {
    "nodes": [
      "CradleDataLoading_training",
      "TabularPreprocessing_training",
      "XGBoostTraining"
    ],
    "edges": [
      ["CradleDataLoading_training", "TabularPreprocessing_training"],
      ["TabularPreprocessing_training", "XGBoostTraining"]
    ]
  },
  "statistics": {
    "node_count": 3,
    "edge_count": 2,
    "has_cycles": false,
    "entry_nodes": ["CradleDataLoading_training"],
    "exit_nodes": ["XGBoostTraining"],
    "max_depth": 2,
    "isolated_nodes": []
  }
}
```

Key rules enforced by the reader/writer (`src/cursus/api/dag/pipeline_dag_serializer.py`):

- `dag.nodes` must be a list; `dag.edges` must be a list of `[src, dst]` pairs.
- Every edge endpoint must be a real node (no dangling edges) and the graph must be
  acyclic — writing an empty or cyclic DAG raises `ValueError`.
- The reader rejects documents whose **major** schema version it does not support.

You can also hand-write `dag.json` — the fields above are all the compiler needs. Load
one back at any time with `import_dag_from_json("dag.json")`.

## Step 2 — Write the matching config

The config JSON has two top-level keys, `configuration` and `metadata`:

```json
{
  "configuration": {
    "shared": {
      "author": "your-alias",
      "bucket": "your-s3-bucket",
      "role": "arn:aws:iam::123456789012:role/YourSageMakerRole",
      "region": "NA",
      "pipeline_name": "my-first-pipeline",
      "pipeline_version": "0.1.0"
    },
    "specific": {
      "CradleDataLoading_training": {
        "job_type": "training"
      },
      "TabularPreprocessing_training": {
        "job_type": "training"
      },
      "XGBoostTraining": {
      }
    }
  },
  "metadata": {
    "config_types": {
      "CradleDataLoading_training": "CradleDataLoadingConfig",
      "TabularPreprocessing_training": "TabularPreprocessingConfig",
      "XGBoostTraining": "XGBoostTrainingConfig"
    }
  }
}
```

How the pieces map:

- **`configuration.shared`** — fields common to every step (author, bucket, role,
  region, pipeline name/version). Configs inherit these, so you set them once.
- **`configuration.specific.<Node>`** — per-node overrides and required fields (for
  example `job_type`, hyperparameters, source dirs). The key is the **DAG node name**.
- **`metadata.config_types`** — the critical `node → config class` map. It tells the
  loader which config class to instantiate for each node. `XGBoostTraining` →
  `XGBoostTrainingConfig`, and so on. This is also the escape hatch for
  off-convention node names: an entry here is treated as a deliberate, user-authored
  binding and is **not** flagged as a misresolution during validation (see
  `ValidationEngine.validate_dag_compatibility` in
  `src/cursus/core/compiler/validation.py`).

```{note}
The config class names (`CradleDataLoadingConfig`, `TabularPreprocessingConfig`,
`XGBoostTrainingConfig`) must be classes the step catalog can discover, and each
`specific` block must satisfy that class's required fields. Real configs carry many
more fields than the minimal skeleton above (S3 locations, instance types, framework
versions, data-source specs). See [Config system](../concepts/config_system.md) and
the [step catalog](../reference/generated/step_catalog.md) for each step's required
inputs.
```

### The reliable way to produce a config: build objects, then merge-and-save

Because real configs have many required and interdependent fields, most authors don't
hand-write the JSON. Instead you construct the config *objects* in Python and let
Cursus serialize them into exactly the `shared` / `specific` / `metadata.config_types`
shape shown above, using `merge_and_save_configs`:

```python
from cursus.core.config_fields import merge_and_save_configs

# Each config is an instance of its step's config class, populated with your
# settings. (Import the concrete config classes from cursus.steps.configs and fill
# in the required fields for your steps.)
configs = [
    data_load_cfg,      # -> CradleDataLoadingConfig
    preprocess_cfg,     # -> TabularPreprocessingConfig
    train_cfg,          # -> XGBoostTrainingConfig
]

merge_and_save_configs(configs, "config.json")
```

`merge_and_save_configs` (see `src/cursus/core/config_fields/__init__.py`) factors
common fields into `shared`, keeps per-step fields under `specific`, and writes the
`metadata.config_types` map for you — producing a `config.json` the compiler consumes
directly.

## Step 3 — Compile with the CLI

With `dag.json` and `config.json` in hand, the fastest path is the
`cursus compile` command (`src/cursus/cli/compile_cli.py`). Its two required options
are the DAG file and the config file:

```bash
cursus compile -d dag.json -c config.json
```

Expected output (compile-only, no AWS calls):

```text
✓ DAG loaded: 3 nodes, 2 edges
✓ Config loaded: 2 step configurations
✓ Pipeline compiled successfully

Pipeline: my-first-pipeline-0-1-0-pipeline
Steps: 3 SageMaker steps created
```

When you don't pass `-n/--pipeline-name`, the name is generated from the config's
`pipeline_name` and `pipeline_version` by `generate_pipeline_name`
(`src/cursus/core/compiler/name_generator.py`) as `<pipeline_name>-<version>-pipeline`,
sanitized to SageMaker's naming rules (dots and underscores become hyphens) — hence
`my-first-pipeline-0-1-0-pipeline`. Passing `-n my-first-pipeline` uses that literal
name instead.

### Validate first (recommended)

Before compiling, run in **validate-only** mode. This resolves configs and builders
and reports problems *without* building the pipeline — fast feedback on missing
configs, unresolvable builders, and node/config mismatches:

```bash
cursus compile -d dag.json -c config.json --validate-only
```

A healthy run prints:

```text
Validation Results:
✓ All DAG nodes have matching configurations
✓ All step builders resolved successfully
✓ No dependency issues found

Validation passed! Ready for compilation.
```

If something is wrong, you get an actionable breakdown and a non-zero exit code — for
example a node missing from `metadata.config_types` or a config whose class the step
catalog can't resolve to a builder:

```text
❌ Validation failed!

Missing configurations:
  - XGBoostTraining

Unresolvable builders:
  - TabularPreprocessing_training (TabularPreprocessing)
```

### Read the compilation report

Add `--show-report` to compile *and* print how every node was resolved. This uses
`PipelineDAGCompiler.compile_with_report`, which returns a `ConversionReport`
(`src/cursus/core/compiler/validation.py`):

```bash
cursus compile -d dag.json -c config.json --show-report
```

```text
✓ Pipeline compiled successfully

📋 Compilation Report:
   Pipeline: my-first-pipeline-0-1-0-pipeline
   Steps: 3
   Average confidence: 1.00
   Warnings: 0

   Resolution Details:
     CradleDataLoading_training → CradleDataLoadingConfig (CradleDataLoadingStepBuilder, confidence: 1.00)
     TabularPreprocessing_training → TabularPreprocessingConfig (TabularPreprocessingStepBuilder, confidence: 1.00)
     XGBoostTraining → XGBoostTrainingConfig (XGBoostTrainingStepBuilder, confidence: 1.00)
```

Reading it:

- **Steps** — number of DAG nodes turned into SageMaker steps.
- **Average confidence** — mean of the per-node resolution confidence scores. A node
  whose confidence is below `0.80` produces a warning suggesting you rename it (or add
  an explicit `metadata.config_types` entry) for a cleaner match.
- **Warnings** — low-confidence resolutions and any ambiguous bindings.
- **Resolution Details** — for each node, the chosen `config_type`, the `builder_type`
  that will emit the step, and the confidence.

### Save the pipeline definition

To capture the compiled SageMaker pipeline definition as JSON (without deploying),
add `-o`:

```bash
cursus compile -d dag.json -c config.json -o pipeline_definition.json
```

### Useful CLI options

| Option | Effect |
|--------|--------|
| `-d, --dag-file` | Path to the DAG JSON (**required**) |
| `-c, --config-file` | Path to the config JSON (**required**) |
| `-n, --pipeline-name` | Override the generated pipeline name |
| `--role` | IAM role ARN for the pipeline |
| `-o, --output` | Save the pipeline definition to a JSON file |
| `--validate-only` | Resolve & validate, do not compile |
| `--show-report` | Compile and print the resolution report |
| `--format {text,json}` | Console output format (`json` is script-friendly) |
| `--upsert` | Create/update the pipeline in SageMaker (needs AWS) |
| `--start` | Start an execution (requires `--upsert`) |

Full option reference: [CLI](../cli.rst).

## Step 4 — Compile with the Python API

The CLI is a thin wrapper over the Python API. Use the API directly when you want to
inspect the report programmatically, embed compilation in your own tooling, or pass a
live `PipelineSession`.

```python
from cursus.api.dag import PipelineDAG
from cursus.core.compiler import PipelineDAGCompiler

# 1. Build (or import_dag_from_json) the DAG
dag = PipelineDAG()
dag.add_node("CradleDataLoading_training")
dag.add_node("TabularPreprocessing_training")
dag.add_node("XGBoostTraining")
dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")

# 2. Create the compiler, pointing it at your config file
compiler = PipelineDAGCompiler(config_path="config.json")

# 3. (Optional) validate before compiling
result = compiler.validate_dag_compatibility(dag)
print(result.summary())          # e.g. "✅ Validation passed"
if not result.is_valid:
    print(result.detailed_report())
    raise SystemExit(1)

# 4. Compile and get the report
pipeline, report = compiler.compile_with_report(dag, pipeline_name="my-first-pipeline")

print(report.summary())
print(report.detailed_report())  # per-node config/builder/confidence
print("Pipeline name:", pipeline.name)
print("Step count:", len(pipeline.steps))
```

A few things worth knowing about `PipelineDAGCompiler`:

- **`config_path`** is the only required argument. `role` and `sagemaker_session` are
  optional — you need them for deployment, not for compilation.
- It accepts either a `PipelineDAG` instance **or** a path to a DAG JSON file wherever
  it takes a `dag` argument (it calls `import_dag_from_json` for you).
- `project_root` / `anchor_file` help Cursus resolve your step source directories. When
  omitted they are inferred from the config file's location; the self-documenting form
  is `anchor_file=__file__` from the module that builds the pipeline.

### Preview resolution without compiling

If you just want to see how nodes will bind — for example while iterating on node
names — call `preview_resolution`, which returns a `ResolutionPreview`:

```python
preview = compiler.preview_resolution(dag)
print(preview.display())
```

### The one-call shortcut

If you don't need the report object, the top-level `compile_dag_to_pipeline`
convenience function does everything in a single call and is re-exported from the
package root:

```python
from cursus import compile_dag_to_pipeline   # alias: cursus.compile_dag

pipeline = compile_dag_to_pipeline(
    dag=dag,                 # or dag_path="dag.json"
    config_path="config.json",
    pipeline_name="my-first-pipeline",
)
```

## Understanding resolution & common errors

When you compile, Cursus performs two matchings per node:

1. **node → config.** By the `metadata.config_types` map, an explicit `specific` key,
   or name-based matching against available configs. A confident, convention-following
   name (`XGBoostTraining`) resolves cleanly; an off-convention name needs an explicit
   `config_types` entry.
2. **config → builder.** The config's class name maps (via the step registry and step
   catalog) to the step builder that emits the SageMaker step. Each step type in the
   registry declares both a `config_class` and a `builder_step_name` — e.g.
   `XGBoostTraining` maps to `XGBoostTrainingConfig` and `XGBoostTrainingStepBuilder`,
   and `TabularPreprocessing` to `TabularPreprocessingConfig` /
   `TabularPreprocessingStepBuilder`. That registry pairing is what the report's
   `config_type → builder_type` lines reflect.

Validation surfaces the failures of either step. The most common ones:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Missing configurations: <node>` | No config resolves to that node | Add a `specific.<node>` block **and** a `metadata.config_types.<node>` entry |
| `Unresolvable builders: <node> (<StepType>)` | Config class has no builder in the catalog | Use a supported config class, or add its step (see [Author a step](author_a_step.md)) |
| `Config-node mismatch: node base '<X>' names a step type, but the bound config <C> resolves to step type '<Y>'` | Node name encodes one step type but its config is another | Rename the node, or add an explicit `metadata.config_types` binding |
| Low **confidence** warnings in the report | Fuzzy name match | Rename the node to `<StepType>` / `<StepType>_<job_type>`, or bind it explicitly |

For the full picture of the validation layers, see [Validation](../concepts/validation.md).

## Step 5 (optional) — Deploy to SageMaker

Everything so far runs offline. To create/update the pipeline in the SageMaker
service, you need valid AWS credentials and an execution role, then add `--upsert`.
Add `--start` to also kick off an execution (it requires `--upsert`):

```bash
# Create or update the pipeline in SageMaker
cursus compile -d dag.json -c config.json \
  --role arn:aws:iam::123456789012:role/YourSageMakerRole --upsert

# Compile, upsert, and start an execution in one shot
cursus compile -d dag.json -c config.json \
  --role arn:aws:iam::123456789012:role/YourSageMakerRole --upsert --start
```

On success the CLI prints the pipeline ARN and, for `--start`, the execution ARN plus
a SageMaker console link to monitor it.

The Python equivalent uses the `sagemaker` SDK's `Pipeline` methods on the object you
compiled:

```python
pipeline.upsert(role_arn="arn:aws:iam::123456789012:role/YourSageMakerRole")
execution = pipeline.start()
print(execution.arn)
```

```{warning}
`--upsert` and `--start` make real AWS calls and can incur cost. Validate and compile
locally first; only deploy once the report looks right.
```

## Recap

You built a three-node DAG, wrote a config that maps each node to a config class via
`metadata.config_types`, and compiled it to a SageMaker pipeline — through both the
`cursus compile` CLI and `PipelineDAGCompiler.compile_with_report`. You also learned to
validate before compiling and to read resolution confidence in the report.

Next steps:

- [Build a pipeline from the catalog](pipeline_from_catalog.md) — start from a ready-made pipeline instead of hand-building a DAG.
- [Author a step](author_a_step.md) — add your own step type to the catalog.
- [DAG & compilation](../concepts/dag_and_compilation.md) and [Config system](../concepts/config_system.md) — the concepts behind this workflow.
- [Quickstart](../getting_started/quickstart.md) and [API reference](../api/index.rst).
