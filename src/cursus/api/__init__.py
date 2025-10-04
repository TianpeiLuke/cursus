"""
Cursus API module.

This module provides the main API interfaces for Cursus functionality,
including DAG management and pipeline compilation.
"""

# Import DAG classes for direct access
from .dag import (
    PipelineDAG,
    PipelineDAGResolver,
    PipelineExecutionPlan,
    WorkspaceAwareDAG,
)

__all__ = [
    # DAG classes
    "PipelineDAG",
    "PipelineDAGResolver",
    "PipelineExecutionPlan",
    "WorkspaceAwareDAG",
]
