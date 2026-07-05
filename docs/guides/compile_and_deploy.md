# Compile, Deploy & Run a Pipeline

This guide is the end-to-end recipe for turning a **DAG** and a **configuration
file** into a runnable SageMaker pipeline: compile the definition, save it,
*upsert* it into the SageMaker service, and start an execution. It covers both
the `cursus compile` CLI and the equivalent Python API, plus how to read the
compilation report and interpret exit codes.

If you are new to what "compilation" means here, read
[DAG & Compilation](../concepts/dag_and_compilation.md) first. For the full
command surface see the [CLI reference](../cli.rst); for class/function
signatures see the [API reference](../api/index.rst).

## What compilation does

Compilation takes two inputs and produces one output:

| Input | What it is |
| --- | --- |
| **DAG file** (`-d`) | A serialized [`PipelineDAG`](../concepts/dag_and_compilation.md) — nodes (step names) and edges (dependencies), stored as JSON. |
| **Config file** (`-c`) | A merged cursus config JSON with two top-level sections: `metadata` (whose `config_types` maps each saved step name to its config class) and `configuration` (with `shared` fields plus per-step `specific` overrides). Produced by the config tooling — see [Generate configs](generate_configs.md). |

The compiler resolves each DAG node to a configuration and a step builder (via
the [Step catalog](../reference/generated/step_catalog.md)), assembles the
SageMaker steps, and returns a `sagemaker.workflow.pipeline.Pipeline` object.
That object can then be serialized, upserted, and executed.

The stages, in order, are:

1. **Load** the DAG and config.
2. **Validate** node → config → builder resolution (optional, always available).
3. **Compile** to a `Pipeline` object.
4. **Save** the pipeline definition JSON (optional, `-o`).
5. **Upsert** to SageMaker (optional, `--upsert`).
6. **Start** an execution (optional, `--start`, requires `--upsert`).

## The `cursus compile` command

### Flags

The command is defined in `src/cursus/cli/compile_cli.py`. The flags you will
use most:

| Flag | Short | Required | Meaning |
| --- | --- | --- | --- |
| `--dag-file` | `-d` | yes | Path to the serialized DAG JSON file (must exist). |
| `--config-file` | `-c` | yes | Path to the configuration JSON file (must exist). |
| `--output` | `-o` | no | Save the compiled pipeline definition to this JSON path. |
| `--upsert` | | no | Create/update the pipeline in the SageMaker service. |
| `--start` | | no | Start an execution after upserting. **Requires `--upsert`.** |
| `--role` | | no | IAM role ARN used for pipeline execution / upsert. |
| `--pipeline-name` | `-n` | no | Override the generated pipeline name. |
| `--validate-only` | | no | Validate compatibility only; do not compile. |
| `--show-report` | | no | Compile and print a detailed compilation report. |
| `--format` | | no | Console output format: `text` (default) or `json`. |

`--dag-file` and `--config-file` use Click's `exists=True`, so a missing path
fails argument parsing before any work starts.

### Recipes

```bash
# 1. Validate only — check that every node resolves, then stop
cursus compile -d dag.json -c config.json --validate-only

# 2. Compile and inspect (console output only, nothing deployed)
cursus compile -d dag.json -c config.json

# 3. Compile with a detailed resolution report
cursus compile -d dag.json -c config.json --show-report

# 4. Compile and save the pipeline definition to disk
cursus compile -d dag.json -c config.json -o pipeline_definition.json

# 5. Deploy (upsert) to SageMaker
cursus compile -d dag.json -c config.json --upsert \
  --role arn:aws:iam::123456789012:role/SageMakerRole

# 6. Full workflow — compile + upsert + start an execution
cursus compile -d dag.json -c config.json --upsert --start \
  --role arn:aws:iam::123456789012:role/SageMakerRole
```

The flags compose: `-o`, `--upsert`, and `--start` run in that fixed order in a
single invocation, so recipe 6 saves nothing but upserts then starts. Add `-o`
to also persist the definition in the same call.

### Typical text output

A full `--upsert --start` run prints each stage as it completes:

```text
✓ DAG loaded: 5 nodes, 4 edges
✓ Config loaded: 2 step configurations
✓ Pipeline compiled successfully

Pipeline: my-pipeline-1-0-0
Steps: 5 SageMaker steps created

Upserting to SageMaker...
✓ Pipeline created/updated
  Pipeline Name: my-pipeline-1-0-0
  Pipeline ARN: arn:aws:sagemaker:us-east-1:123456789012:pipeline/my-pipeline-1-0-0

Starting execution...
✓ Execution started
  Execution ARN: arn:aws:sagemaker:us-east-1:123456789012:pipeline/my-pipeline-1-0-0/execution/abc123
  Execution ID: abc123
  Status: Executing

Monitor execution at:
  https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/pipelines/my-pipeline-1-0-0/executions/abc123
```

The console link is derived from the execution ARN, so it is only shown when the
ARN carries a region.

The `Config loaded: N step configurations` line is a naive count of the config
file's top-level JSON keys (those not starting with `_`). A standard merged
config has exactly two top-level keys — `metadata` and `configuration` — so this
line reports `2`, not the number of steps; it is a cosmetic echo and does not
affect compilation.

## Reading the compilation report

There are two distinct reports, produced by two different modes.

### Validation report (`--validate-only`)

`--validate-only` builds a `PipelineDAGCompiler`, calls
`validate_dag_compatibility(dag)`, and prints the resulting `ValidationResult`.
It does **not** produce a `Pipeline`. The result carries:

| Field | Meaning |
| --- | --- |
| `is_valid` | `True` only if there are no missing configs, unresolvable builders, config errors, or dependency issues. |
| `missing_configs` | DAG nodes with no matching configuration. |
| `unresolvable_builders` | Nodes whose step builder could not be resolved. |
| `config_errors` | Per-config validation errors (map of config name → messages). |
| `dependency_issues` | Unsatisfied input/output dependency problems. |
| `warnings` | Non-fatal issues. |

`ValidationResult` is a Pydantic model; call `result.summary()` for a one-line
status or `result.detailed_report()` for the full breakdown.

A passing run prints confirmation and returns; a failing run lists the offending
nodes and exits non-zero:

```text
Validation Results:
❌ Validation failed!

Missing configurations:
  - model_eval

Unresolvable builders:
  - custom_transform
```

With `--format json` the same data is emitted as a JSON object with a
`"status": "validation_complete"` field plus `is_valid`, `dag_nodes`,
`dag_edges`, `missing_configs`, `unresolvable_builders`, and `warnings`.

### Compilation report (`--show-report`)

`--show-report` calls `compiler.compile_with_report(...)`, which returns a
`(Pipeline, ConversionReport)` tuple. The report describes how each node was
resolved:

| Field | Meaning |
| --- | --- |
| `pipeline_name` | Name of the generated pipeline. |
| `steps` | List of step (node) names. |
| `avg_confidence` | Mean resolution confidence across nodes. |
| `warnings` | Includes a warning for any node resolved with confidence `< 0.8`. |
| `resolution_details` | Per node: `config_type`, `builder_type`, and `confidence`. |

```text
✓ Pipeline compiled successfully

📋 Compilation Report:
   Pipeline: my-pipeline-1-0-0
   Steps: 5
   Average confidence: 0.94
   Warnings: 1

   Warnings:
     - Low confidence resolution for node 'custom_transform': 0.62

   Resolution Details:
     data_load → CradleDataLoadConfig (CradleDataLoadingStepBuilder, confidence: 1.00)
     preprocess → TabularPreprocessingConfig (TabularPreprocessingStepBuilder, confidence: 1.00)
     ...
```

Low-confidence nodes are a signal to make your node names align more closely
with config types, or to add explicit metadata — see
[Dependency resolution](../concepts/dependency_resolution.md).

## Exit codes

`compile_pipeline` communicates failure by raising `SystemExit(1)`; Click turns
that into the process exit status. There are only two outcomes:

| Exit code | When |
| --- | --- |
| `0` | Every requested stage succeeded (including a passing `--validate-only`). |
| `1` | Any failure: DAG load error, config load error, failed validation, compilation error, save error, upsert error, execution-start error, or `--start` without `--upsert`. |

Because each stage has its own guard, the first failing stage exits `1`
immediately and later stages do not run. The `--start`-requires-`--upsert` check
happens up front:

```bash
$ cursus compile -d dag.json -c config.json --start
❌ Error: --start flag requires --upsert flag
$ echo $?
1
```

Use `--format json` in scripts for machine-readable status; the exit code is
still the primary success/failure signal.

## The Python API

The CLI is a thin wrapper over the compiler API in
`src/cursus/core/compiler/dag_compiler.py`. Two entry points exist.

### One-call compilation: `compile_dag_to_pipeline`

For the simple case, `compile_dag_to_pipeline` loads/accepts a DAG and returns a
ready `Pipeline`:

```python
from cursus.api.dag import import_dag_from_json
from cursus.core.compiler import compile_dag_to_pipeline

dag = import_dag_from_json("dag.json")

pipeline = compile_dag_to_pipeline(
    dag=dag,
    config_path="config.json",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    pipeline_name="my-pipeline",   # optional override
)
```

You can pass `dag_path="dag.json"` instead of a `dag` instance and skip the
explicit import. On any failure it raises `PipelineAPIError`.

### Full control: `PipelineDAGCompiler`

For validation, previews, and reports, construct the compiler directly:

```python
from cursus.api.dag import import_dag_from_json
from cursus.core.compiler import PipelineDAGCompiler

dag = import_dag_from_json("dag.json")

compiler = PipelineDAGCompiler(
    config_path="config.json",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
)

# Validate before compiling
result = compiler.validate_dag_compatibility(dag)
if not result.is_valid:
    print(result.summary())
    raise SystemExit(1)

# Compile and get the resolution report
pipeline, report = compiler.compile_with_report(dag)
print(report.summary())
```

`PipelineDAGCompiler` also exposes:

- `compile(dag, pipeline_name=...)` — compile without a report.
- `preview_resolution(dag)` — a `ResolutionPreview` (fields `node_config_map`,
  `config_builder_map`, `resolution_confidence`, `ambiguous_resolutions`,
  `recommendations`) of the node → config → builder mappings and confidences,
  without building the pipeline. Call `.display()` for a formatted view.
- `get_supported_step_types()` — list of step type names the catalog can resolve.
- `validate_config_file()` — a quick structural check of the config file,
  returning a dict of validation results.

When your steps live in your own project (custom `source_dir`s or an external
step-pack), pass `anchor_file=__file__` (or `project_root=Path(__file__).parent`)
and, if needed, `workspace_dirs=[...]` so path resolution and step discovery
anchor to your project. See [Define a step pack](define_a_step_pack.md).

### Save, upsert, and start

The returned `pipeline` is a standard SageMaker `Pipeline`, so deployment uses
the SDK methods the CLI itself calls:

```python
# Save the definition JSON to disk (this is what -o does)
with open("pipeline_definition.json", "w") as f:
    f.write(pipeline.definition())

# Create or update the pipeline in SageMaker (this is what --upsert does)
response = pipeline.upsert(role_arn="arn:aws:iam::123456789012:role/SageMakerRole")
print(response["PipelineArn"])

# Start an execution (this is what --start does)
execution = pipeline.start()
print(execution.arn)
```

`pipeline.upsert()`, `pipeline.definition()`, and `pipeline.start()` come from
the SageMaker Python SDK; Cursus does not wrap them. The CLI reads the pipeline
ARN from `response.get("PipelineArn")` and the execution ARN from
`execution.arn`, exactly as shown above.

```{note}
`upsert` and `start` require valid AWS credentials and an IAM role with
SageMaker permissions. Compilation, validation, and saving a definition are all
fully offline and need no AWS access — do those first to catch config problems
before touching the service.
```

## Preparing the DAG file

If you build a `PipelineDAG` in Python, serialize it to the JSON that `-d`
expects with `export_dag_to_json`:

```python
from cursus.api.dag import PipelineDAG
from cursus.api.dag.pipeline_dag_serializer import export_dag_to_json

dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_edge("data_load", "preprocess")

export_dag_to_json(dag, "dag.json")   # now usable with: cursus compile -d dag.json
```

`import_dag_from_json` (used by both the CLI and the API) is the inverse. For
ready-made DAGs, browse the
[Pipeline catalog](../reference/generated/pipeline_catalog.md).

## Troubleshooting

| Symptom | Likely cause & fix |
| --- | --- |
| `Failed to load DAG from ...` | The DAG JSON is malformed or not a serialized `PipelineDAG`. Re-export with `export_dag_to_json`. |
| Validation lists **missing configurations** | A DAG node has no matching config key. Add the config or rename the node — see [Generate configs](generate_configs.md). |
| Validation lists **unresolvable builders** | No step builder maps to the resolved config type. Check the [Step catalog](../reference/generated/step_catalog.md). |
| Low `avg_confidence` / low-confidence warnings | Node names don't align well with config types; rename nodes or add metadata. |
| `--start` rejected | You passed `--start` without `--upsert`. Add `--upsert`. |
| Upsert/start errors | AWS credentials or the IAM `--role` are missing or lack SageMaker permissions. |

## See also

- [DAG & Compilation](../concepts/dag_and_compilation.md) — concepts behind the compiler.
- [Validate a pipeline](validate_a_pipeline.md) — deeper validation workflows.
- [CLI reference](../cli.rst) — every `cursus` command.
- [API reference](../api/index.rst) — `PipelineDAGCompiler` and friends.
- [Step catalog](../reference/generated/step_catalog.md) — how nodes resolve to builders.
- [MCP tools](../reference/generated/mcp_tools.md) — programmatic access to the same operations.
