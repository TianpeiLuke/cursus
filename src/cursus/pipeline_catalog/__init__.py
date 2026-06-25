"""
Pipeline Catalog — queryable DAG store with common pipeline builders.

Usage:
    # Quick compile (SAIS notebook):
    from cursus.pipeline_catalog import build_and_compile
    pipeline, report = build_and_compile(dag_path="dag.json", config_path="config.json", ...)

    # Generate MODS class (for MODS Lambda):
    from cursus.pipeline_catalog import build_mods_pipeline
    MyPipeline = build_mods_pipeline(author="...", version="...", dag_path="...", config_path="...")

    # Search catalog:
    from cursus.pipeline_catalog import search_dags
    results = search_dags(features=["bedrock", "training"], framework="pytorch")
"""

from .core import (
    create_pipeline,
    list_available_pipelines,
    build_and_compile,
    build_mods_pipeline,
    recommend_dag,
    auto_select_dag,
)
from .shared_dags import (
    load_shared_dag,
    search_dags,
    get_all_shared_dags,
    get_catalog_index,
)
from .mods_pipelines import (
    discover_mods_pipelines,
    load_mods_pipeline,
    get_registered_mods_pipelines,
)

__all__ = [
    # Builders
    "build_and_compile",
    "build_mods_pipeline",
    "recommend_dag",
    "auto_select_dag",
    "create_pipeline",
    "list_available_pipelines",
    # Catalog
    "load_shared_dag",
    "search_dags",
    "get_all_shared_dags",
    "get_catalog_index",
    # MODS
    "discover_mods_pipelines",
    "load_mods_pipeline",
    "get_registered_mods_pipelines",
]
