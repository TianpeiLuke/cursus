"""
Pipeline DAG API module.

This module provides the core DAG classes for building and managing
pipeline topologies with intelligent dependency resolution.
"""

from .base_dag import PipelineDAG
from .edge_types import (
    EdgeType,
    DependencyEdge,
    ConditionalEdge,
    ParallelEdge,
    EdgeCollection
)
from .enhanced_dag import EnhancedPipelineDAG

__all__ = [
    # Core DAG classes
    "PipelineDAG",
    "EnhancedPipelineDAG",
    
    # Edge types and management
    "EdgeType",
    "DependencyEdge", 
    "ConditionalEdge",
    "ParallelEdge",
    "EdgeCollection",
]
