# Pipeline Catalog — MODS Pipelines (deprecated namespace)

The class-based MODS layer that lived here — the `mods_api` module
(`create_mods_pipeline`, `create_mods_pipeline_by_name`, `PIPELINE_REGISTRY`, the
per-framework convenience functions) and per-pipeline `*_new.py` example classes —
was **removed in the 2026-06 pipeline-catalog refactor**. It depended on the
hardcoded `pipeline_catalog.pipelines.*` pipeline classes, which were also deleted
when the catalog became a declarative DAG store.

## Use this instead

MODS-enhanced pipelines are generated on demand from a DAG + config:

```python
from cursus.pipeline_catalog import build_mods_pipeline

MyPipeline = build_mods_pipeline(
    author="...",
    version="...",
    description="...",
    dag_path="dag.json",
    config_path="config.json",
)
# MyPipeline is a @MODSTemplate-decorated class with the standard MODS interface:
#   __init__(sagemaker_session, execution_role, regional_alias) + generate_pipeline()
```

`build_mods_pipeline` (in `../core/builders.py`) is the single source — no per-pipeline
classes, no separate registry. See the catalog [README](../README.md) for the full
DAG-store + builders + router design.
