# Cursus Pipeline Catalog

A **declarative DAG store + builders** for Cursus pipelines. Pipelines are not
Python classes — each is a `*.dag.json` file (nodes + edges + metadata), compiled
on demand into a SageMaker `Pipeline` by the single `PipelineDAGCompiler`. A small
`core/` package provides the build, recommend, and agent-tool entry points.

> **History.** This module was refactored (2026-06) from a class-based catalog
> (one `pipelines/<name>.py` class per pipeline, a parallel `mods_api.py` MODS
> layer, and a Zettelkasten "connection graph / recommendation engine" discovery
> layer) into the DAG-store design below. The companion change is the
> [StepInterface migration](https://code.amazon.com/packages/AmazonCursus) that
> unified each step's contract+spec into one `.step.yaml`. Together they make the
> whole pipeline-definition layer data-driven.

## Structure

```
pipeline_catalog/
├── __init__.py                 # public API (re-exports core + shared_dags)
├── core/
│   ├── pipeline_factory.py     # create_pipeline(dag_id|dag_path, config) -> (Pipeline, report)
│   ├── builders.py             # build_and_compile(...)  +  build_mods_pipeline(...)
│   ├── router.py               # recommend_dag / auto_select_dag / recommend_for_agent
│   └── agent_tool.py           # pipeline_catalog_tool(...) + TOOL_SCHEMA (LLM tool interface)
├── shared_dags/
│   ├── catalog_index.json      # queryable index of all DAGs (the source of truth for discovery)
│   ├── <framework>/*.dag.json  # the DAG definitions (nodes, edges, metadata)
│   └── __init__.py             # load_shared_dag / get_all_shared_dags / search_dags / get_catalog_index
├── pipeline_exe/               # execution-document helpers
└── mods_pipelines/             # (namespace stub — class-based MODS layer removed; use build_mods_pipeline)
```

There are **42 shared DAGs** across `xgboost/` (15), `pytorch/` (9), `bedrock/`
(10), `mtl/` (3), `lightgbm/` (2), `dummy/` (2), and `singleton/` (1).

## Quick Start

### Build & compile a pipeline (SAIS notebook)

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

### Compile a *catalogued* DAG by id

```python
from cursus.pipeline_catalog import create_pipeline, list_available_pipelines

list_available_pipelines()        # -> ['bedrock_pytorch_incremental_edx', 'xgboost_complete_e2e', ...] (42)

pipeline, report = create_pipeline(
    dag_id="bedrock_pytorch_incremental_edx",   # or dag_path="/path/to/some.dag.json"
    config_path="config.json",
    sagemaker_session=session,
    role=role,
)
```

### Generate a MODS pipeline class (for the MODS Lambda)

`build_mods_pipeline` returns a `@MODSTemplate`-decorated class with the standard
MODS interface (`__init__(sagemaker_session, execution_role, regional_alias)` +
`generate_pipeline()`). It replaces the old per-pipeline MODS classes and `mods_api`.

```python
from cursus.pipeline_catalog import build_mods_pipeline

MungedAddressPipelineNA = build_mods_pipeline(
    author="bjjin",
    version="0.0.5",
    description="Munged Address Detection DistilBERT Training Pipeline",
    dag_path="pipeline_config/dag_NA.json",     # relative to the calling module
    config_path="pipeline_config/config_NA.json",
)
# In the MODS Lambda:
pipeline = MungedAddressPipelineNA(sagemaker_session=sess, execution_role=role).generate_pipeline()
```

## Discovery

The catalog index (`shared_dags/catalog_index.json`) carries rich metadata per DAG
— `framework`, `complexity`, `features`, `input_requirements`, `constraints`,
`cost`, `agent_context`. Three ways to query it:

### 1. Direct search

```python
from cursus.pipeline_catalog import search_dags, get_all_shared_dags

search_dags(framework="pytorch")                          # all pytorch DAGs
search_dags(features=["training", "calibration"], framework="xgboost")   # ranked by feature overlap
get_all_shared_dags()                                     # {dag_id: metadata}
```

### 2. Recommendation / auto-selection

```python
from cursus.pipeline_catalog.core import recommend_dag, auto_select_dag

# Ranked recommendations (score 0-1 + reasoning):
recommend_dag(framework="pytorch", features=["bedrock", "training", "edx_uploading"], task_type="incremental")

# Auto-select the single best match (returns None below min_score):
result = auto_select_dag(framework="xgboost", features=["training", "calibration"], min_score=0.6)
if result:
    dag_id, dag, score = result
```

### 3. Agent / LLM tool interface

`core/agent_tool.py` exposes the catalog as a single tool (MCP / OpenAI function
calling / Claude tool_use compatible) via `TOOL_SCHEMA` + `pipeline_catalog_tool`.

```python
from cursus.pipeline_catalog.core import pipeline_catalog_tool, TOOL_SCHEMA

pipeline_catalog_tool(action="recommend", data_type="text", needs_llm=True)
pipeline_catalog_tool(action="get_dag", dag_id="bedrock_pytorch_incremental_edx")
pipeline_catalog_tool(action="get_config_guidance", dag_id="bedrock_pytorch_incremental_edx")
pipeline_catalog_tool(action="list_frameworks")
pipeline_catalog_tool(action="list_features")
```

`recommend_for_agent` (which `action="recommend"` wraps) selects on **semantic
constraints** — `data_type` (text/tabular/mixed), `has_labels`, `needs_llm`,
`multi_task`, `incremental`, `gpu_available` — and returns each DAG's
`agent_context` (`when_to_use`, `prerequisites`, `config_guidance`) so an agent can
reason about the choice.

## Public API (`from cursus.pipeline_catalog import ...`)

| Symbol | Purpose |
|---|---|
| `build_and_compile(dag_path, config_path, session, role)` | Compile a DAG file → `(Pipeline, report)` (SAIS) |
| `build_mods_pipeline(author, version, description, dag_path, config_path)` | Generate a `@MODSTemplate` pipeline class (MODS Lambda) |
| `create_pipeline(dag_id \| dag_path, config_path, ...)` | Compile a catalogued or file DAG |
| `list_available_pipelines()` | All 42 catalogued DAG ids |
| `load_shared_dag(dag_id)` | Load a catalogued DAG → `PipelineDAG` |
| `search_dags(features=?, framework=?)` | Filter/rank the index |
| `get_all_shared_dags()`, `get_catalog_index()` | Raw index access |
| `recommend_dag`, `auto_select_dag`, `recommend_for_agent` (via `core`) | Recommendation |
| `pipeline_catalog_tool`, `TOOL_SCHEMA` (via `core`) | LLM tool interface |

## Adding a Pipeline

1. Add a `<framework>/<name>.dag.json` under `shared_dags/` — `{"dag": {"nodes": [...],
   "edges": [[src, dst], ...]}, "metadata": {...}}`. **Every edge endpoint must be a
   declared node** (a dangling endpoint raises `KeyError` at load).
2. Add an entry to `shared_dags/catalog_index.json` (`id`, `path`, `framework`,
   `node_count`, `edge_count`, `features`, `input_requirements`, `constraints`,
   `agent_context`, …). Keep `node_count`/`edge_count` in sync with the DAG.
3. Run the catalog tests: `pytest tests/pipeline_catalog`. The suite loads every
   catalogued DAG, verifies node/edge counts match the index, and asserts no
   dangling edge endpoints — so a malformed DAG fails fast.

## Tests

`tests/pipeline_catalog/` (52 tests) covers the index↔disk consistency, that every
DAG loads into a valid `PipelineDAG`, the router scoring, the agent-tool actions,
and `build_mods_pipeline` class generation.

---

**The Cursus Pipeline Catalog: declarative DAGs, compiled on demand.**
