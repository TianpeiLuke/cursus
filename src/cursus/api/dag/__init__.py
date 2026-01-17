"""
Pipeline DAG API module.

This module provides the core DAG classes for building and managing
pipeline topologies with intelligent dependency resolution.
"""

from .base_dag import PipelineDAG
from .pipeline_dag_resolver import PipelineDAGResolver, PipelineExecutionPlan
from .pipeline_dag_serializer import (
    PipelineDAGWriter,
    PipelineDAGReader,
    export_dag_to_json,
    import_dag_from_json,
)

__all__ = [
    # Core DAG classes
    "PipelineDAG",
    "PipelineDAGResolver",
    "PipelineExecutionPlan",
    # Serialization classes
    "PipelineDAGWriter",
    "PipelineDAGReader",
    # Convenience functions
    "export_dag_to_json",
    "import_dag_from_json",
]
