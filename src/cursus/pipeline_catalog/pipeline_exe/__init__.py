"""
Pipeline Execution Document Integration Module

This module provides simple integration between pipeline catalog pipelines
and the standalone execution document generator.

The module provides:
- Simple pipeline execution document generation functions
- Direct mapping from pipeline names to configurations and DAGs
- Integration with standalone execution document generator
- Utility functions for pipeline execution document handling
"""

from .generator import generate_execution_document_for_pipeline
from .utils import (
    get_config_path_for_pipeline,
    load_shared_dag_for_pipeline,
    create_execution_doc_template_for_pipeline,
)

__all__ = [
    "generate_execution_document_for_pipeline",
    "get_config_path_for_pipeline", 
    "load_shared_dag_for_pipeline",
    "create_execution_doc_template_for_pipeline",
]
