"""
Cursus API module.

This module provides the main API interfaces for Cursus functionality,
including DAG management and pipeline compilation.
"""

# Import DAG classes for direct access
from .dag import (
    PipelineDAG,
    EnhancedPipelineDAG,
    WorkspaceAwareDAG,
    EdgeType,
    DependencyEdge,
    ConditionalEdge,
    ParallelEdge,
    EdgeCollection
)

__all__ = [
    # DAG classes
    "PipelineDAG",
    "EnhancedPipelineDAG",
    "WorkspaceAwareDAG",
    
    # Edge types and management
    "EdgeType",
    "DependencyEdge",
    "ConditionalEdge", 
    "ParallelEdge",
    "EdgeCollection",
]
