"""
Pipeline DAG API module.

This module provides the core DAG classes for building and managing
pipeline topologies with intelligent dependency resolution.
"""

from .base_dag import PipelineDAG
from .pipeline_dag_resolver import PipelineDAGResolver, PipelineExecutionPlan

__all__ = [
    # Core DAG classes
    "PipelineDAG",
    "PipelineDAGResolver",
    "PipelineExecutionPlan",
]
