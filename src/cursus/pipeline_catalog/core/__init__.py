"""Pipeline Catalog Core — factory, builders, and router."""

from .pipeline_factory import create_pipeline, list_available_pipelines
from .builders import build_and_compile, build_mods_pipeline
from .router import recommend_dag, auto_select_dag, recommend_for_agent
from .agent_tool import pipeline_catalog_tool, TOOL_SCHEMA

__all__ = [
    "create_pipeline",
    "list_available_pipelines",
    "build_and_compile",
    "build_mods_pipeline",
    "recommend_dag",
    "auto_select_dag",
    "recommend_for_agent",
    "pipeline_catalog_tool",
    "TOOL_SCHEMA",
]
