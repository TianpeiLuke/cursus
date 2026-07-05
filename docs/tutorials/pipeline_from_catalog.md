# Build a Pipeline from the Catalog

This tutorial shows how to go from *"I need a tabular XGBoost pipeline"* to a
compiled SageMaker pipeline **without writing a DAG by hand**. Cursus ships a
data-driven **pipeline catalog**: a set of pre-built shared DAGs plus a queryable
index that describes each one (framework, task type, complexity, cost, features,
and agent guidance). You query the catalog, pick a DAG, load it, and compile it
against your config.

You will:

1. Recommend a DAG from the command line with `cursus pipeline-catalog`.
2. Inspect a concrete DAG (`xgboost_complete_e2e`) node-by-node.
3. Understand every field in `catalog_index.json` and how to filter on
   framework / task_type / complexity.
4. Do the same thing in Python with `recommend_dag`, `load_shared_dag`, and
   `build_and_compile`.

> **Prerequisites.** A working Cursus install and (for the final compile step) a
> SageMaker session, an execution role, and a pipeline config JSON. If you only
> want to *explore* the catalog, no AWS credentials are needed — the recommend /
> inspect steps read local JSON only.

Related reading: [Pipeline catalog reference](../reference/generated/pipeline_catalog.md) ·
[Step catalog](../reference/generated/step_catalog.md) ·
[CLI reference](../cli.rst) · [Concepts](../concepts/index.md) ·
[API reference](../api/index.rst) · [MCP tools](../reference/generated/mcp_tools.md)

---

## What the catalog actually is

The catalog has three moving parts, all under
`src/cursus/pipeline_catalog/`:

| Part | Location | Role |
| --- | --- | --- |
| Shared DAG files | `shared_dags/<subdir>/*.dag.json` | The actual node/edge definitions (the exact path per DAG comes from the index's `path` field; subdirs are `bedrock`, `dummy`, `lightgbm`, `mtl`, `pytorch`, `singleton`, `xgboost`, so the folder is not always the framework name) |
| Catalog index | `shared_dags/catalog_index.json` | One metadata entry per DAG (the queryable layer) |
| Router / builders | `core/router.py`, `core/builders.py` | Rank DAGs, load them, and compile them |

Everything you need is re-exported from the package root, so imports stay short:

```python
from cursus.pipeline_catalog import (
    recommend_dag,      # rank DAGs against requirements
    auto_select_dag,    # pick the single best match (or None)
    load_shared_dag,    # DAG id -> PipelineDAG object
    build_and_compile,  # dag_path + config_path -> compiled Pipeline
    search_dags,        # feature/framework search
    get_catalog_index,  # raw index dict
)
```

These names are defined in
`src/cursus/pipeline_catalog/__init__.py` — use them as shown; do not invent
variants.

---

## Step 1 — Recommend a DAG from the CLI

The CLI group is `cursus pipeline-catalog` (registered in
`src/cursus/cli/__init__.py`). It has three subcommands:

| Subcommand | Purpose |
| --- | --- |
| `recommend` | Rank catalog DAGs against your requirements |
| `list` | Show frameworks and how many DAGs each has |
| `get-dag <dag_id>` | Print nodes, edges, and requirements for one DAG |

Start by asking for a tabular XGBoost recommendation:

```bash
cursus pipeline-catalog recommend --data-type tabular --framework xgboost
```

Typical output (top matches shown, scores are 0–1):

```text
Top 5 recommended pipeline DAG(s):

  1. xgboost_complete_e2e  (score: 1.0)
      framework: xgboost | nodes: 10
      when to use: Tabular binary/multiclass classification with XGBoost. ...
  2. xgboost_complete_e2e_dummy  (score: 1.0)
      framework: xgboost | nodes: 10
  ...
```

The `recommend` command exposes semantic flags (see
`src/cursus/cli/pipeline_catalog_cli.py`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-type` | — | `text`, `tabular`, or `mixed` |
| `--has-labels / --no-labels` | `--has-labels` | Whether labeled data already exists |
| `--needs-llm / --no-llm` | `--no-llm` | Whether a Bedrock LLM is needed for labeling/enrichment |
| `--multi-task / --single-task` | `--single-task` | Multiple output tasks |
| `--incremental / --first-time` | `--first-time` | Incremental retraining vs. first run |
| `--framework` | — | `pytorch`, `xgboost`, `lightgbm`, `lightgbmmt`, or `any` |
| `--gpu / --no-gpu` | `--gpu` | Whether GPU instances are available |
| `--format` | `text` | `text` or `json` |

`--framework` is a **hard filter**: when you pass a real framework (not `any`),
only that framework's DAGs are considered, so a requested framework can never be
crowded out of the top-N by higher-scoring DAGs from other frameworks. This is
implemented in `recommend_for_agent` in `core/router.py`.

Use `--format json` when you want to pipe results into another tool:

```bash
cursus pipeline-catalog recommend --data-type tabular --framework xgboost --format json
```

List which frameworks exist and how many DAGs each has:

```bash
cursus pipeline-catalog list
```

```text
Pipeline catalog frameworks (DAG counts):
  bedrock: 4
  dummy: 2
  generic: 1
  ...
  xgboost: 15
```

(Frameworks are printed sorted; the full set is `bedrock`, `dummy`, `generic`,
`lightgbm`, `lightgbmmt`, `pytorch`, `xgboost`, and `xgboost_mt`.)

> **Note.** The `recommend` and `list` commands are semantic wrappers around
> `pipeline_catalog_tool` (`core/agent_tool.py`) — the same engine used by the
> agent / [MCP tools](../reference/generated/mcp_tools.md). The CLI and the
> agent surface share one recommendation implementation.

---

## Step 2 — Inspect the chosen DAG

Once a DAG id looks promising, print its structure with `get-dag`:

```bash
cursus pipeline-catalog get-dag xgboost_complete_e2e
```

This returns JSON with the DAG's `nodes`, `edges`, `input_requirements`,
`constraints`, `cost`, and `agent_context` (the underlying handler loads the
full `.dag.json` for node/edge details — see the `get_dag` action of
`pipeline_catalog_tool` in `core/agent_tool.py`).

Our example DAG `xgboost_complete_e2e` lives at
`shared_dags/xgboost/complete_e2e.dag.json` and has **10 nodes / 11 edges**:

**Nodes**

```
CradleDataLoading_training        TabularPreprocessing_training
XGBoostTraining                   ModelCalibration_calibration
Package                           Registration
Payload                           CradleDataLoading_calibration
TabularPreprocessing_calibration  XGBoostModelEval_calibration
```

**Edges (data flow)**

```
CradleDataLoading_training     -> TabularPreprocessing_training
TabularPreprocessing_training  -> XGBoostTraining
CradleDataLoading_calibration  -> TabularPreprocessing_calibration
XGBoostTraining                -> XGBoostModelEval_calibration
TabularPreprocessing_calibration -> XGBoostModelEval_calibration
XGBoostModelEval_calibration   -> ModelCalibration_calibration
ModelCalibration_calibration   -> Package
XGBoostTraining                -> Package
XGBoostTraining                -> Payload
Package                        -> Registration
Payload                        -> Registration
```

Reading the graph:

- **Two independent data legs.** One `CradleDataLoading` → `TabularPreprocessing`
  leg feeds *training*; a second, parallel leg feeds *calibration/eval*. That is
  why the config needs data for both a training window and a (different-period)
  calibration window.
- **Train once, fan out.** `XGBoostTraining` feeds three consumers:
  `XGBoostModelEval_calibration` (evaluate on held-out data),
  `Package` (bundle the model), and `Payload` (build a registration payload).
- **Calibrate, then package.** Evaluation output flows into
  `ModelCalibration_calibration`, whose output is also packaged.
- **Register.** `Package` and `Payload` converge on `Registration`, the single
  exit point (a MIMS registration step — note `requires_mims: true` in the
  entry's constraints).

Each node name is a **step type** that the compiler resolves against the
[step catalog](../reference/generated/step_catalog.md). The DAG only declares the
*shape*; concrete instance configuration comes from your config JSON at compile
time. See [Concepts](../concepts/index.md) for how DAG + config compile into a
SageMaker pipeline.

---

## Step 3 — Understand `catalog_index.json`

`catalog_index.json` is the queryable layer. Its top level looks like:

```json
{
  "version": "3.1",
  "generated": "2026-06-24T23:57:14Z",
  "total_dags": 44,
  "frameworks": ["bedrock", "dummy", "generic", "lightgbm",
                 "lightgbmmt", "pytorch", "xgboost", "xgboost_mt"],
  "dags": [ /* one entry per DAG */ ]
}
```

Here is the full `xgboost_complete_e2e` entry, which we will use as the field
reference:

```json
{
  "id": "xgboost_complete_e2e",
  "path": "xgboost/complete_e2e.dag.json",
  "description": "Complete XGBoost end-to-end pipeline with training, calibration, packaging, registration, and evaluation",
  "framework": "xgboost",
  "complexity": "comprehensive",
  "task_type": "end_to_end",
  "node_count": 10,
  "edge_count": 11,
  "features": ["training", "calibration", "packaging", "registration", "evaluation"],
  "input_requirements": {
    "data_types": ["tabular"],
    "text_support": false,
    "multi_task": false,
    "output_type": "binary_or_multiclass",
    "data_source": "cradle_mds",
    "requires_llm": false
  },
  "constraints": {
    "requires_gpu": false,
    "min_instance": "ml.m5.xlarge",
    "supports_multi_gpu": false,
    "requires_mims": true
  },
  "cost": {
    "estimated_hours": "1-3 (CPU training)",
    "cost_driver": "cpu_training",
    "instance_cost_tier": "medium",
    "scalability": "data_size_linear"
  },
  "used_by_projects": ["cap_dnr_eu_xgboost", "pda_eu_xgboost", "pda_na_xgboost"],
  "agent_context": { "when_to_use": "...", "config_guidance": { "...": "..." } }
}
```

### Field reference

| Field | Type | What it means / how it's used |
| --- | --- | --- |
| `id` | string | Stable DAG identifier — the argument to `load_shared_dag(id)` and `get-dag <id>` |
| `path` | string | Location of the `.dag.json` relative to `shared_dags/` |
| `description` | string | Human-readable summary |
| `framework` | string | ML framework; matched by the `--framework` hard filter |
| `complexity` | string | One of `simple`, `standard`, `advanced`, `comprehensive` |
| `task_type` | string | Keyword like `end_to_end`, `training`, `incremental_training_with_llm_scoring` (may be empty) |
| `node_count` / `edge_count` | int | DAG size at a glance |
| `features` | list | Capability tags (`training`, `calibration`, `evaluation`, `edx_uploading`, …) — the primary scoring signal |
| `input_requirements` | object | Data expectations: `data_types`, `text_support`, `multi_task`, `output_type`, `data_source`, `requires_llm` |
| `constraints` | object | Runtime needs: `requires_gpu`, `min_instance`, `supports_multi_gpu`, `requires_mims`, `requires_bedrock_access`, `requires_edx_access` |
| `cost` | object | `estimated_hours`, `cost_driver`, `instance_cost_tier`, `scalability` (informational, not fabricated benchmarks) |
| `used_by_projects` | list | Existing projects that use this DAG (social proof / examples) |
| `agent_context` | object | Rich LLM-facing guidance: `when_to_use`, `when_not_to_use`, `differentiators`, `prerequisites`, `config_guidance`, `decision_tree` |

### `agent_context.config_guidance` — what you must supply

This is the field you will care about most when moving toward a real compile. For
`xgboost_complete_e2e` it says:

- `user_must_provide`: `cradle_sql`, `label_column`, `feature_columns`, `author`,
  `service_name`
- `safe_defaults`: `n_estimators`, `max_depth`, `learning_rate`, `instance_types`
- `common_pitfalls`: e.g. *"Feature columns must match between training and
  inference"* and *"Calibration data should be from a different time period than
  training"* (which explains the two data legs from Step 2).

Fetch just this block from the agent tool if you are scripting:

```python
from cursus.pipeline_catalog.core.agent_tool import pipeline_catalog_tool

guidance = pipeline_catalog_tool(action="get_config_guidance",
                                 dag_id="xgboost_complete_e2e")
print(guidance["config_guidance"]["user_must_provide"])
```

### How the scoring works

`recommend_dag` (in `core/router.py`) assigns each DAG a score in `[0, 1]` from
four weighted signals:

| Signal | Weight | Rule |
| --- | --- | --- |
| Feature overlap | 0.5 | `|required ∩ dag.features| / |required|` |
| Framework match | 0.25 | 1.0 for exact match; 0.1 for a partial id match |
| Task type match | 0.15 | 1.0 if `task_type` substring-matches the DAG's `task_type` |
| Complexity match | 0.1 | 1.0 for exact; scaled down for adjacent tiers |

When a filter is omitted, that signal contributes its full weight (no penalty).
DAGs scoring at or below `0.2` are dropped, and results are returned
highest-first. The `complexity` tiers used for adjacency are, in order:
`simple`, `standard`, `advanced`, `comprehensive`.

### Picking by framework / task_type / complexity

- **Framework:** pass `framework="xgboost"`. Exact matches earn the full 0.25.
- **Task type:** pass `task_type="end_to_end"` to prefer full pipelines, or
  `task_type="training"` for train-only DAGs, or `task_type="incremental"` to
  favor incremental-retraining DAGs (substring match, so partial keywords work).
- **Complexity:** pass `complexity="simple"` for a quick prototype
  (e.g. `xgboost_simple`, 5 nodes) or `complexity="comprehensive"` for a full
  train→calibrate→register pipeline like `xgboost_complete_e2e`.

---

## Step 4 — Do it in Python

### 4a. Rank candidates with `recommend_dag`

```python
from cursus.pipeline_catalog import recommend_dag

results = recommend_dag(
    framework="xgboost",
    features=["training", "calibration", "registration"],
    task_type="end_to_end",
    complexity="comprehensive",
    max_results=5,
)

for r in results:
    print(f"{r['id']:40s} score={r['score']:<5} {r['reasoning']}")
```

Each result is a copy of the catalog entry with two extra keys: `score`
(rounded to 3 decimals) and `reasoning` (a human-readable list of what matched).

### 4b. Auto-select the single best match

If you would rather let the router decide, `auto_select_dag` returns a
`(dag_id, PipelineDAG, score)` tuple — or `None` if nothing clears the
`min_score` threshold (default `0.6`):

```python
from cursus.pipeline_catalog import auto_select_dag

selection = auto_select_dag(
    framework="xgboost",
    features=["training", "calibration", "registration"],
    task_type="end_to_end",
)
if selection is None:
    raise SystemExit("No DAG matched confidently — loosen your requirements.")

dag_id, dag, score = selection
print(f"Selected {dag_id} (score={score:.2f})")
```

Note `auto_select_dag` already loads the `PipelineDAG` for you.

### 4c. Load a DAG explicitly with `load_shared_dag`

If you know the id (from Step 1), skip scoring and load directly:

```python
from cursus.pipeline_catalog import load_shared_dag

dag = load_shared_dag("xgboost_complete_e2e")
print(dag.nodes)   # the 10 node names from Step 2
print(dag.edges)   # the 11 edges
```

`load_shared_dag` looks the id up in the index, resolves the `path`, and calls
`import_dag_from_json` (`src/cursus/api/dag/pipeline_dag_serializer.py`) to
return a `PipelineDAG`. An unknown id raises `ValueError` listing the available
ids.

### 4d. Search by feature/framework with `search_dags`

`search_dags` is a lighter, non-scored alternative to the router when you just
want entries containing certain features:

```python
from cursus.pipeline_catalog import search_dags

matches = search_dags(features=["calibration", "registration"], framework="xgboost")
for m in matches:
    print(m["id"], m.get("_score"))
```

It filters by framework, keeps only DAGs with at least one matching feature, and
sorts by feature overlap (`_score`).

### 4e. Compile with `build_and_compile`

Finally, turn a DAG plus your config into a runnable SageMaker pipeline.
`build_and_compile` (in `core/builders.py`) takes **file paths**, not the loaded
object: it calls `import_dag_from_json(dag_path)` internally and drives
`PipelineDAGCompiler`.

To feed it a catalog DAG, resolve the shared-DAG path from the index:

```python
from pathlib import Path
from cursus.pipeline_catalog import build_and_compile, get_catalog_index
from cursus.pipeline_catalog.shared_dags import SHARED_DAGS_DIR

index = get_catalog_index()
entry = next(d for d in index["dags"] if d["id"] == "xgboost_complete_e2e")
dag_path = str(SHARED_DAGS_DIR / entry["path"])

pipeline, report = build_and_compile(
    dag_path=dag_path,
    config_path="pipeline_config/config.json",  # your config (see config_guidance)
    sagemaker_session=pipeline_session,          # a SageMaker PipelineSession
    role=execution_role,                         # IAM role ARN
)

print(f"Compiled '{pipeline.name}' with {len(pipeline.steps)} steps")
```

`build_and_compile` returns `(Pipeline, ConversionReport)` (the report type is
`ConversionReport`, defined in `src/cursus/core/compiler/validation.py`). The
`Pipeline` is a standard `sagemaker.workflow.pipeline.Pipeline` you can `upsert()`
and `start()`; the report describes how each DAG node was resolved to a step — its
`resolution_details` maps each step name to its resolution info, and
`avg_confidence` gives the mean resolver confidence. Call `report.summary()` for a
one-line result or `report.detailed_report()` for the full per-step breakdown.

> **Config, not covered here.** Your `config_path` must satisfy the DAG's
> `config_guidance` (Step 3) — for `xgboost_complete_e2e` that means at least
> `cradle_sql`, `label_column`, `feature_columns`, `author`, and `service_name`.
> Building that config is a separate topic; see the [Concepts](../concepts/index.md)
> and [guides](../api/index.rst) pages.

### Optional: generate a reusable pipeline class

If you are deploying via MODS Lambda rather than compiling in a notebook,
`build_mods_pipeline` (also in `core/builders.py`) generates a
`@MODSTemplate`-decorated class with the standard
`__init__(sagemaker_session, execution_role, regional_alias)` /
`generate_pipeline()` interface. It resolves `dag_path` / `config_path` relative
to the calling module. Use `build_and_compile` for interactive work and
`build_mods_pipeline` for packaged deployment.

---

## End-to-end recap

```python
from cursus.pipeline_catalog import auto_select_dag, build_and_compile, get_catalog_index
from cursus.pipeline_catalog.shared_dags import SHARED_DAGS_DIR

# 1. Let the router pick the best XGBoost end-to-end DAG.
dag_id, dag, score = auto_select_dag(
    framework="xgboost",
    features=["training", "calibration", "registration"],
    task_type="end_to_end",
)

# 2. Resolve its shared-DAG path from the index.
entry = next(d for d in get_catalog_index()["dags"] if d["id"] == dag_id)
dag_path = str(SHARED_DAGS_DIR / entry["path"])

# 3. Compile against your config.
pipeline, report = build_and_compile(
    dag_path=dag_path,
    config_path="pipeline_config/config.json",
    sagemaker_session=pipeline_session,
    role=execution_role,
)
```

You picked a DAG from a data-driven catalog instead of authoring one, inspected
its exact shape, understood every index field that drives selection, and
compiled it into a SageMaker pipeline — all through confirmed Cursus APIs.

### Where to go next

- [Pipeline catalog reference](../reference/generated/pipeline_catalog.md) — the
  full generated catalog listing.
- [Step catalog](../reference/generated/step_catalog.md) — the step types that
  DAG nodes resolve to.
- [MCP tools](../reference/generated/mcp_tools.md) — call the same
  recommendation engine from an agent.
- [CLI reference](../cli.rst) · [API reference](../api/index.rst) ·
  [Tutorials index](./index.md)
