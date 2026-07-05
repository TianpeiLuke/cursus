# The Pipeline Catalog

The **pipeline catalog** is Cursus's library of ready-to-use pipeline shapes. It is
a *data-driven* store: every pipeline is a `*.dag.json` file (nodes + edges +
metadata), and a single `catalog_index.json` makes the whole collection queryable.
A small `core/` package sits on top to **recommend** a DAG for your requirements,
**build/compile** it into a SageMaker `Pipeline`, and expose the whole thing as a
**tool** that LLM agents (and the CLI/MCP surfaces) can call.

This page explains how the catalog is laid out, how the deterministic router scores
and filters DAGs, how the builders turn a DAG file into a runnable pipeline, and how
this design replaced the older class-based catalog.

Related reading: [DAG & Compilation](../concepts/dag_and_compilation.md) ·
[Registry & Discovery](../concepts/registry_and_discovery.md) ·
[Step interfaces](../concepts/step_interfaces.md) ·
[Pipeline catalog reference](../reference/generated/pipeline_catalog.md) ·
[MCP tools](../reference/generated/mcp_tools.md) · [CLI](../cli.rst).

## Why a data-driven catalog

Earlier, each pipeline was a hand-written Python class under `pipelines/<name>.py`,
with a parallel MODS layer and a separate graph-based "recommendation engine" for
discovery. That meant three things to keep in sync per pipeline and a lot of
boilerplate (each MODS class was roughly 100–300 lines).

The current design replaces all of that with **data**:

- A pipeline is just a **JSON file** — no class, no code.
- Discovery reads a **single index** (`catalog_index.json`) instead of walking a
  connection graph.
- Compilation is handled by the one shared `PipelineDAGCompiler`, so every DAG —
  catalogued or ad-hoc — goes through the same path.

The net effect: adding a pipeline means dropping a `.dag.json` file and an index
entry, not writing (and maintaining) a class.

## Layout

```
src/cursus/pipeline_catalog/
├── __init__.py                 # public API (re-exports core + shared_dags)
├── core/
│   ├── pipeline_factory.py     # create_pipeline(dag_id | dag_path, config) -> (Pipeline, report)
│   ├── builders.py             # build_and_compile(...)  +  build_mods_pipeline(...)
│   ├── router.py               # recommend_dag / auto_select_dag / recommend_for_agent
│   └── agent_tool.py           # pipeline_catalog_tool(...) + TOOL_SCHEMA (LLM tool interface)
└── shared_dags/
    ├── catalog_index.json      # queryable index of all DAGs (source of truth for discovery)
    ├── <framework>/*.dag.json  # the DAG definitions themselves
    └── __init__.py             # load_shared_dag / search_dags / get_catalog_index / ...
```

DAGs are grouped into directories: `bedrock/`, `dummy/`, `lightgbm/`, `mtl/`,
`pytorch/`, `xgboost/`, plus `singleton/`. Directory names are *not* one-to-one with
framework tags: the multi-task DAGs (frameworks `lightgbmmt` and `xgboost_mt`) live
under `mtl/`, and the single `generic` DAG (`cradle_data_loading_singleton`) lives
under `singleton/`. The frameworks currently indexed are `bedrock`, `dummy`,
`generic`, `lightgbm`, `lightgbmmt`, `pytorch`, `xgboost`, and `xgboost_mt` (see the
`frameworks` field at the top of `catalog_index.json`, alongside `version`,
`generated`, and `total_dags`).

### The DAG file

Each `*.dag.json` has two top-level keys — `dag` (the graph) and `metadata`:

```json
{
  "dag": {
    "nodes": ["DummyTraining", "Package", "Payload", "Registration"],
    "edges": [
      ["DummyTraining", "Package"],
      ["DummyTraining", "Payload"],
      ["Package", "Registration"],
      ["Payload", "Registration"]
    ]
  },
  "metadata": {
    "name": "dummy_e2e_basic",
    "framework": "dummy",
    "complexity": "simple",
    "task_type": "end_to_end",
    "features": ["end_to_end", "dummy", "testing", "packaging", "registration"],
    "node_count": 4,
    "edge_count": 4
  }
}
```

The real `metadata` block carries more than this (`description`, `entry_points`,
`exit_points`, `input_requirements`, `constraints`, `cost`, `agent_context`) — the
same rich fields the index mirrors — but `dag` + `metadata` are the only two
top-level keys.

Node names are step types resolved through the [step registry](../concepts/registry_and_discovery.md);
edges are `[source, destination]` pairs. The file is loaded into a `PipelineDAG`
by `import_dag_from_json` (from `cursus.api.dag`) and then handed to the compiler.

### The index

`shared_dags/catalog_index.json` is the discovery source of truth. Its top level
carries `version`, `total_dags`, and the `frameworks` list; `dags` is an array of
per-DAG entries. Each entry is rich metadata used by the router and the agent tool:

| Field | Meaning |
|---|---|
| `id` | Catalog identifier, e.g. `bedrock_pytorch_incremental_edx` |
| `path` | Relative path to the `.dag.json` under `shared_dags/` |
| `description` | One-line human summary of the pipeline |
| `framework` | `pytorch`, `xgboost`, `lightgbm`, `lightgbmmt`, `bedrock`, `dummy`, ... |
| `complexity` | `simple` / `standard` / `advanced` / `comprehensive` |
| `task_type` | Free-text keyword, e.g. `end_to_end`, `incremental_training_with_llm_scoring` |
| `features` | Capability tags, e.g. `training`, `calibration`, `edx_uploading`, `bedrock_realtime_processing` |
| `node_count` / `edge_count` | Graph size (kept in sync with the DAG file) |
| `input_requirements` | `data_types`, `text_support`, `multi_task`, `requires_llm`, ... |
| `constraints` | `requires_gpu`, `min_instance`, `requires_mims`, `requires_bedrock_access`, ... |
| `cost` | `estimated_hours`, `cost_driver`, `instance_cost_tier`, `scalability` |
| `used_by_projects` | List of projects that reference this DAG (may be empty) |
| `agent_context` | `when_to_use`, `when_not_to_use`, `differentiators`, `prerequisites`, `config_guidance`, `decision_tree` |

The `agent_context` block is where the catalog's "knowledge" lives: rather than a
separate connection graph, each DAG carries its own `when_to_use` / `when_not_to_use`
guidance, `differentiators`, `prerequisites`, `config_guidance`, and a `decision_tree`.
That embedded rationale is what lets an LLM agent both pick *and justify* a pipeline
from index data alone, without loading every graph.

The `shared_dags/__init__.py` module gives you raw access:

```python
from cursus.pipeline_catalog.shared_dags import (
    get_catalog_index,   # load the whole index dict
    get_all_shared_dags, # {dag_id: metadata}
    load_shared_dag,     # dag_id -> PipelineDAG (raises ValueError if unknown)
    search_dags,         # filter by features / framework
)

index = get_catalog_index()
dag = load_shared_dag("bedrock_pytorch_incremental_edx")
```

`search_dags(features=?, framework=?)` is the simplest filter: it drops DAGs whose
`framework` differs, drops DAGs with zero feature overlap, and ranks the rest by
overlap ratio (`|required ∩ dag_features| / |required|`).

## The router: deterministic recommendation

`core/router.py` provides three pure functions. None of them call an LLM — scoring
is fully deterministic, so the same inputs always yield the same ranking.

### `recommend_dag` — keyword scoring

`recommend_dag(framework, features, task_type, complexity, max_results=5)` returns a
ranked list of index entries, each augmented with a `score` (0–1) and a human-readable
`reasoning` string. The score is a weighted sum:

| Signal | Weight | Rule |
|---|---|---|
| Feature overlap | 0.5 | `|required ∩ dag_features| / |required|` (full 0.5 if no `features` given) |
| Framework match | 0.25 | 1.0 for exact `framework` match; 0.1 partial credit if the framework name appears in the DAG `id` |
| Task type match | 0.15 | 1.0 if `task_type` is a substring of (or contains) the DAG's `task_type`; 0.07 partial if it appears in the `id` |
| Complexity match | 0.1 | 1.0 exact; reduced by distance for adjacent complexity tiers |

Entries scoring at or below `0.2` are dropped, and the survivors are sorted
descending and truncated to `max_results`.

```python
from cursus.pipeline_catalog import recommend_dag

results = recommend_dag(
    framework="pytorch",
    features=["bedrock", "training", "edx_uploading"],
    task_type="incremental",
)
for r in results:
    print(r["id"], r["score"], "-", r["reasoning"])
```

Because unspecified signals contribute their **full** weight (a missing `features`
filter adds the whole 0.5, and so on), an empty query gives every DAG a baseline
score rather than filtering everything out.

### `auto_select_dag` — single best match

`auto_select_dag(framework, features, task_type, min_score=0.6)` runs `recommend_dag`
with `max_results=1` and returns the winner **only if** its score clears `min_score`.
On success it loads the actual graph; on failure it returns `None`:

```python
from cursus.pipeline_catalog import auto_select_dag

result = auto_select_dag(framework="xgboost", features=["training", "calibration"])
if result:
    dag_id, dag, score = result   # dag is a compiled-ready PipelineDAG
else:
    ...  # no confident match; fall back to manual selection
```

### `recommend_for_agent` — semantic constraints + hard framework filter

`recommend_for_agent` is the agent-friendly entry point. Instead of feature strings
it takes **semantic constraints** and multiplies a starting score of `1.0` by
penalties/bonuses as constraints are checked:

- `data_type` (`text` / `tabular` / `mixed`) — a `text` request skips DAGs without
  `text_support` (e.g. XGBoost); pure `tabular` on a PyTorch DAG is dampened.
- `has_labels` — no labels and no LLM available heavily penalizes (can't train).
- `needs_llm` — mismatches between "LLM needed" and a DAG's `requires_llm` are penalized.
- `multi_task` — penalizes single-task DAGs when multi-task is requested, and vice versa.
- `incremental` — boosts DAGs whose `task_type` contains `incremental` or that carry
  `edx_uploading`; penalizes the rest.
- `gpu_available=False` — **excludes** any DAG whose `constraints.requires_gpu` is true.
- `data_volume` — large volumes dampen realtime-Bedrock DAGs (batch is preferred).

The `framework` argument is a **hard filter**, applied *before* scoring and
truncation:

```python
# from router.py
for dag in index["dags"]:
    if framework and dag.get("framework") != framework:
        continue
    ...
```

Applying the filter first is deliberate: a requested framework can never be crowded
out of the top-N by higher-scoring DAGs from other frameworks, and the results never
silently fall back to an unrelated framework. The function returns up to five DAGs,
each carrying its full metadata plus `score` and `reasoning`, including the
`agent_context` an agent needs to justify a choice.

## The builders: from DAG to Pipeline

`core/builders.py` (and the sibling `core/pipeline_factory.py`) turn a DAG file into
a SageMaker `Pipeline`. All of them funnel through `import_dag_from_json` +
`PipelineDAGCompiler.compile_with_report(...)` — see
[DAG & Compilation](../concepts/dag_and_compilation.md) for what the compiler does.

### `build_and_compile` — SAIS / notebook path

`build_and_compile` compiles a DAG file (given its path) and returns `(Pipeline, ConversionReport)`
directly, with no class in between. This is the path for SageMaker Studio (SAIS)
notebooks where you build and `upsert` a pipeline inline:

```python
from cursus.pipeline_catalog import build_and_compile

pipeline, report = build_and_compile(
    dag_path="pipeline_config/dag_training_NA.json",
    config_path="pipeline_config/config_training_NA.json",
    sagemaker_session=pipeline_session,
    role=role,
)
pipeline.upsert()
```

An optional `project_root` argument anchors Docker `source_dir` resolution on your
project folder; when omitted, the compiler infers it from `config_path`.

`create_pipeline` (from `pipeline_factory.py`) is a close cousin that accepts
**either** a catalog `dag_id` (loaded via `load_shared_dag`) **or** an ad-hoc
`dag_path`:

```python
from cursus.pipeline_catalog import create_pipeline, list_available_pipelines

list_available_pipelines()   # -> every catalogued DAG id

pipeline, report = create_pipeline(
    dag_id="bedrock_pytorch_incremental_edx",   # or dag_path="/path/to/some.dag.json"
    config_path="config.json",
    sagemaker_session=session,
    role=role,
)
```

### `build_mods_pipeline` — the `@MODSTemplate` class factory

The MODS Lambda runtime expects a **class** with the standard MODS interface, not a
loose function call. `build_mods_pipeline` generates that class from declarative
inputs, replacing the old hand-written per-pipeline MODS classes:

```python
from cursus.pipeline_catalog import build_mods_pipeline

MungedAddressPipelineNA = build_mods_pipeline(
    author="bjjin",
    version="0.0.5",
    description="Munged Address Detection DistilBERT Training Pipeline",
    dag_path="pipeline_config/dag_NA.json",      # relative to the calling module
    config_path="pipeline_config/config_NA.json",
)

# In the MODS Lambda:
pipeline = MungedAddressPipelineNA(
    sagemaker_session=sess, execution_role=role
).generate_pipeline()
```

Key behaviours, all grounded in the implementation:

- The returned class is decorated with `@MODSTemplate(author, description, version)`.
  If the real `mods.mods_template.MODSTemplate` isn't importable, the module falls
  back to a lightweight decorator that just stamps `_mods_author` / `_mods_description`
  / `_mods_version` onto the class — so the catalog imports cleanly outside MODS.
- The generated class exposes `__init__(sagemaker_session, execution_role,
  regional_alias="NA")` and `generate_pipeline() -> Pipeline`.
- `dag_path` and `config_path` are resolved **relative to the calling module's
  directory** (via `inspect.stack()`), and that same directory is passed to the
  compiler as `project_root`. Because the generated template lives inside the project
  folder, this anchors Docker `source_dir` resolution across Lambda/MODS, SAIS, and
  pip-installed deployments without extra environment variables.
- The class name defaults to one derived from `description` (or you can pass
  `class_name`).

## The agent tool: catalog as a callable

`core/agent_tool.py` wraps the catalog as **one tool** compatible with MCP, OpenAI
function calling, and Claude tool use. It exports `TOOL_SCHEMA` (the JSON schema an
agent registers) and `pipeline_catalog_tool(...)` (the implementation).

The tool dispatches on an `action` field:

| `action` | What it returns |
|---|---|
| `recommend` | Ranked DAGs via `recommend_for_agent` (top 5) — each item carries `dag_id`, `score`, `framework`, `description`, `node_count`, `cost`, `when_to_use`, `differentiators`; the response also has a top-level `total_matches` and a `next_step` hint |
| `get_dag` | Full graph (`nodes`, `edges`) plus `input_requirements`, `constraints`, `cost`, `agent_context` for one `dag_id` |
| `get_config_guidance` | `prerequisites`, `config_guidance`, `common_pitfalls`, and a `decision_tree` for one `dag_id` |
| `list_frameworks` | `{framework: count}` across the index |
| `list_features` | Sorted list of every feature tag in the catalog |

```python
from cursus.pipeline_catalog.core import pipeline_catalog_tool, TOOL_SCHEMA

pipeline_catalog_tool(action="recommend", data_type="text", needs_llm=True)
pipeline_catalog_tool(action="get_dag", dag_id="bedrock_pytorch_incremental_edx")
pipeline_catalog_tool(action="get_config_guidance", dag_id="bedrock_pytorch_incremental_edx")
```

In the `recommend` action, a `framework` of `"any"` (or `None`) means "no filter";
any other value is forwarded to `recommend_for_agent` as the hard filter described
above. Every branch returns a plain, JSON-serializable dict with a `status` field,
so the same code backs multiple surfaces.

## How the surfaces line up

The recommendation engine is written **once** and reused everywhere, so the CLI, MCP
tools, and any agent share identical ranking behaviour.

### CLI

`cursus pipeline-catalog` (in `cli/pipeline_catalog_cli.py`) wraps
`pipeline_catalog_tool`:

```bash
# Rank DAGs for your requirements
cursus pipeline-catalog recommend --data-type tabular --framework xgboost

# List frameworks with DAG counts
cursus pipeline-catalog list

# Inspect one DAG's nodes/edges + requirements
cursus pipeline-catalog get-dag bedrock_pytorch_incremental_edx
```

`recommend` accepts `--data-type`, `--has-labels/--no-labels`, `--needs-llm/--no-llm`,
`--multi-task/--single-task`, `--incremental/--first-time`, `--framework`,
`--gpu/--no-gpu`, and `--format text|json`. See the [CLI reference](../cli.rst) for
the full command surface.

### MCP

The `pipeline_catalog.*` MCP namespace exposes the same engine as discrete tools:
`pipeline_catalog.recommend`, `pipeline_catalog.get_dag`,
`pipeline_catalog.config_guidance`, `pipeline_catalog.auto_select`,
`pipeline_catalog.list`, and `pipeline_catalog.load_dag` (plus the auto-generated
`pipeline_catalog.help`). It delegates to `pipeline_catalog_tool` and
`auto_select_dag`, importing engine modules lazily so a missing optional dependency
only fails that one call. See [MCP tools](../reference/generated/mcp_tools.md).

## Public API surface

The builders, factory, catalog-access helpers, and `recommend_dag` /
`auto_select_dag` are importable straight from `cursus.pipeline_catalog`. The
agent-oriented symbols — `recommend_for_agent`, `pipeline_catalog_tool`, and
`TOOL_SCHEMA` — live only under `cursus.pipeline_catalog.core` (which also re-exports
everything in the top block):

| Symbol | Purpose |
|---|---|
| `build_and_compile(dag_path, config_path, sagemaker_session, role)` | Compile a DAG file → `(Pipeline, report)` (SAIS/notebook) |
| `build_mods_pipeline(author, version, description, dag_path, config_path)` | Generate a `@MODSTemplate` pipeline class (MODS Lambda) |
| `create_pipeline(dag_id \| dag_path, config_path, ...)` | Compile a catalogued or ad-hoc DAG |
| `list_available_pipelines()` | All catalogued DAG ids |
| `load_shared_dag(dag_id)` | Load a catalogued DAG → `PipelineDAG` |
| `search_dags(features=?, framework=?)` | Filter/rank the index |
| `get_all_shared_dags()`, `get_catalog_index()` | Raw index access |
| `recommend_dag`, `auto_select_dag` | Deterministic recommendation (top-level + `core`) |
| `recommend_for_agent` (via `core`) | Semantic-constraint recommendation for agents |
| `pipeline_catalog_tool`, `TOOL_SCHEMA` (via `core`) | LLM tool interface |

## Adding a pipeline

1. Drop a `shared_dags/<framework>/<name>.dag.json` file — `{"dag": {"nodes": [...],
   "edges": [[src, dst], ...]}, "metadata": {...}}`. Every edge endpoint must be a
   declared node.
2. Add a matching entry to `shared_dags/catalog_index.json` (`id`, `path`,
   `framework`, `node_count`, `edge_count`, `features`, `input_requirements`,
   `constraints`, `agent_context`, ...). Keep `node_count` / `edge_count` in sync
   with the DAG.

No Python class, no MODS subclass, no discovery-graph wiring — the JSON *is* the
pipeline, and the router, builders, and agent tool pick it up automatically.

## See also

- [DAG & Compilation](../concepts/dag_and_compilation.md) — how a `PipelineDAG` becomes a SageMaker `Pipeline`.
- [Registry & Discovery](../concepts/registry_and_discovery.md) — how node names resolve to step types.
- [Step interfaces](../concepts/step_interfaces.md) — the per-step contracts the nodes reference.
- [Pipeline catalog reference](../reference/generated/pipeline_catalog.md) · [Step catalog](../reference/generated/step_catalog.md) · [MCP tools](../reference/generated/mcp_tools.md) · [CLI](../cli.rst).
