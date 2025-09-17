"""
Pipeline Execution Document Integration Module

This module provides simple integration between pipeline catalog pipelines
and the standalone execution document generator, maintaining complete independence
between pipeline generation and execution document generation modules.

The module provides:
- Simple pipeline execution document generation functions
- Registry-based pipeline discovery (no hardcoded mappings)
- Dynamic DAG loading with fallback mechanisms
- Utility functions for pipeline execution document handling

Note: This module is completely independent from BasePipeline.
For pipeline generation, use cursus.pipeline_catalog.core.base_pipeline.
For execution document generation, use this module or cursus.mods.exe_doc.generator directly.
"""

from .generator import generate_execution_document_for_pipeline
from .utils import (
    get_config_path_for_pipeline,
    load_shared_dag_for_pipeline,
    create_execution_doc_template_for_pipeline,
    get_pipeline_metadata,
    list_available_pipelines,
    validate_pipeline_setup,
)

__all__ = [
    "generate_execution_document_for_pipeline",
    "get_config_path_for_pipeline", 
    "load_shared_dag_for_pipeline",
    "create_execution_doc_template_for_pipeline",
    "get_pipeline_metadata",
    "list_available_pipelines", 
    "validate_pipeline_setup",
]
