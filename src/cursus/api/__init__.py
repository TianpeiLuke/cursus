"""
AutoPipe API module.

This module provides the main API interfaces for AutoPipe functionality,
including DAG management and pipeline compilation.
"""

# Import DAG classes for direct access
from .dag import (
    PipelineDAG,
    EnhancedPipelineDAG,
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
    
    # Edge types and management
    "EdgeType",
    "DependencyEdge",
    "ConditionalEdge", 
    "ParallelEdge",
    "EdgeCollection",
]
