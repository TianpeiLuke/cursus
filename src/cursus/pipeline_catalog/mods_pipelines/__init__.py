"""
Pipeline Catalog - MODS Pipelines (namespace stub)

The class-based MODS layer (``mods_api`` + per-pipeline ``*_new.py`` classes +
``PIPELINE_REGISTRY``) was removed in the 2026-06 pipeline-catalog refactor, along
with the hardcoded ``pipeline_catalog.pipelines.*`` classes it wrapped.

MODS-enhanced pipelines are now generated from a DAG + config via the canonical
entry point:

    from cursus.pipeline_catalog import build_mods_pipeline

    MyPipeline = build_mods_pipeline(
        author="...",
        version="...",
        description="...",
        dag_path="dag.json",
        config_path="config.json",
    )

This module is retained only as a namespace; it intentionally exposes no
class-based MODS API. See ``../README.md`` and ``core/builders.py``.
"""

__all__: list = []
