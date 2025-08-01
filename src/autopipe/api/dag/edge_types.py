"""
Edge types for enhanced pipeline DAG with typed dependencies.

This module defines the various types of edges that can exist between
pipeline steps, including typed dependency edges with confidence scoring.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EdgeType(Enum):
    """Types of edges in the pipeline DAG."""
    DEPENDENCY = "dependency"        # Standard dependency edge
    CONDITIONAL = "conditional"      # Conditional dependency
    PARALLEL = "parallel"           # Parallel execution hint
    SEQUENTIAL = "sequential"       # Sequential execution requirement


class DependencyEdge(BaseModel):
    """Represents a typed dependency edge between step ports."""
    source_step: str = Field(
        description="Name of the source step",
        min_length=1
    )
    target_step: str = Field(
        description="Name of the target step", 
        min_length=1
    )
    source_output: str = Field(
        description="Logical name of source output",
        min_length=1
    )
    target_input: str = Field(
        description="Logical name of target input",
        min_length=1
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for auto-resolved edges",
        ge=0.0,
        le=1.0
    )
    edge_type: EdgeType = Field(
        default=EdgeType.DEPENDENCY,
        description="Type of edge"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "use_enum_values": True
    }
    
    def to_property_reference_dict(self) -> Dict[str, Any]:
        """Convert edge to a property reference dictionary for SageMaker."""
        return {
            "Get": f"Steps.{self.source_step}.{self.source_output}"
        }
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if this edge has high confidence."""
        return self.confidence >= threshold
    
    def is_auto_resolved(self) -> bool:
        """Check if this edge was automatically resolved."""
        return self.confidence < 1.0
    
    def __str__(self):
        return f"{self.source_step}.{self.source_output} -> {self.target_step}.{self.target_input}"
    
    def __repr__(self):
        return (f"DependencyEdge(source='{self.source_step}.{self.source_output}', "
                f"target='{self.target_step}.{self.target_input}', "
                f"confidence={self.confidence:.3f})")


class ConditionalEdge(DependencyEdge):
    """Represents a conditional dependency edge."""
    condition: str = Field(
        default="",
        description="Condition expression"
    )
    edge_type: EdgeType = Field(
        default=EdgeType.CONDITIONAL,
        description="Type of edge"
    )
    
    @model_validator(mode='after')
    def validate_condition(self) -> 'ConditionalEdge':
        """Validate condition and log warning if empty."""
        if not self.condition:
            logger.warning(f"ConditionalEdge {self} has no condition specified")
        return self


class ParallelEdge(DependencyEdge):
    """Represents a parallel execution hint edge."""
    max_parallel: Optional[int] = Field(
        default=None,
        description="Maximum parallel executions",
        ge=1
    )
    edge_type: EdgeType = Field(
        default=EdgeType.PARALLEL,
        description="Type of edge"
    )


class EdgeCollection:
    """Collection of edges with utility methods."""
    
    def __init__(self):
        self.edges: Dict[str, DependencyEdge] = {}
        self._source_index: Dict[str, list] = {}  # source_step -> list of edge_ids
        self._target_index: Dict[str, list] = {}  # target_step -> list of edge_ids
    
    def add_edge(self, edge: DependencyEdge) -> str:
        """
        Add an edge to the collection.
        
        Args:
            edge: DependencyEdge to add
            
        Returns:
            Edge ID for the added edge
        """
        edge_id = f"{edge.source_step}:{edge.source_output}->{edge.target_step}:{edge.target_input}"
        
        # Check for duplicate edges
        if edge_id in self.edges:
            existing = self.edges[edge_id]
            if existing.confidence < edge.confidence:
                # Replace with higher confidence edge
                logger.info(f"Replacing edge {edge_id} with higher confidence "
                           f"({existing.confidence:.3f} -> {edge.confidence:.3f})")
            else:
                logger.debug(f"Ignoring duplicate edge {edge_id} with lower confidence")
                return edge_id
        
        self.edges[edge_id] = edge
        
        # Update indices
        if edge.source_step not in self._source_index:
            self._source_index[edge.source_step] = []
        self._source_index[edge.source_step].append(edge_id)
        
        if edge.target_step not in self._target_index:
            self._target_index[edge.target_step] = []
        self._target_index[edge.target_step].append(edge_id)
        
        logger.debug(f"Added edge: {edge}")
        return edge_id
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the collection."""
        if edge_id not in self.edges:
            return False
        
        edge = self.edges[edge_id]
        del self.edges[edge_id]
        
        # Update indices
        if edge.source_step in self._source_index:
            self._source_index[edge.source_step].remove(edge_id)
            if not self._source_index[edge.source_step]:
                del self._source_index[edge.source_step]
        
        if edge.target_step in self._target_index:
            self._target_index[edge.target_step].remove(edge_id)
            if not self._target_index[edge.target_step]:
                del self._target_index[edge.target_step]
        
        logger.debug(f"Removed edge: {edge}")
        return True
    
    def get_edges_from_step(self, step_name: str) -> list:
        """Get all edges originating from a step."""
        edge_ids = self._source_index.get(step_name, [])
        return [self.edges[edge_id] for edge_id in edge_ids]
    
    def get_edges_to_step(self, step_name: str) -> list:
        """Get all edges targeting a step."""
        edge_ids = self._target_index.get(step_name, [])
        return [self.edges[edge_id] for edge_id in edge_ids]
    
    def get_edge(self, source_step: str, source_output: str, 
                target_step: str, target_input: str) -> Optional[DependencyEdge]:
        """Get a specific edge by its components."""
        edge_id = f"{source_step}:{source_output}->{target_step}:{target_input}"
        return self.edges.get(edge_id)
    
    def list_all_edges(self) -> list:
        """Get list of all edges."""
        return list(self.edges.values())
    
    def list_auto_resolved_edges(self) -> list:
        """Get list of automatically resolved edges."""
        return [edge for edge in self.edges.values() if edge.is_auto_resolved()]
    
    def list_high_confidence_edges(self, threshold: float = 0.8) -> list:
        """Get list of high confidence edges."""
        return [edge for edge in self.edges.values() if edge.is_high_confidence(threshold)]
    
    def list_low_confidence_edges(self, threshold: float = 0.6) -> list:
        """Get list of low confidence edges that may need review."""
        return [edge for edge in self.edges.values() if edge.confidence < threshold]
    
    def get_step_dependencies(self, step_name: str) -> Dict[str, DependencyEdge]:
        """Get all dependencies for a step as a dictionary."""
        edges = self.get_edges_to_step(step_name)
        return {edge.target_input: edge for edge in edges}
    
    def validate_edges(self) -> list:
        """Validate all edges and return list of errors."""
        errors = []
        
        for edge_id, edge in self.edges.items():
            # Check for self-dependencies
            if edge.source_step == edge.target_step:
                errors.append(f"Self-dependency detected: {edge_id}")
            
            # Check confidence bounds
            if not 0.0 <= edge.confidence <= 1.0:
                errors.append(f"Invalid confidence {edge.confidence} for edge {edge_id}")
            
            # Check for empty names
            if not all([edge.source_step, edge.target_step, edge.source_output, edge.target_input]):
                errors.append(f"Empty component in edge {edge_id}")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the edge collection."""
        edges = list(self.edges.values())
        
        if not edges:
            return {
                'total_edges': 0,
                'auto_resolved_edges': 0,
                'high_confidence_edges': 0,
                'low_confidence_edges': 0,
                'average_confidence': 0.0,
                'edge_types': {}
            }
        
        confidences = [edge.confidence for edge in edges]
        edge_types = {}
        for edge in edges:
            edge_type = edge.edge_type.value
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'total_edges': len(edges),
            'auto_resolved_edges': len(self.list_auto_resolved_edges()),
            'high_confidence_edges': len(self.list_high_confidence_edges()),
            'low_confidence_edges': len(self.list_low_confidence_edges()),
            'average_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'edge_types': edge_types,
            'unique_source_steps': len(self._source_index),
            'unique_target_steps': len(self._target_index)
        }
    
    def __len__(self):
        return len(self.edges)
    
    def __iter__(self):
        return iter(self.edges.values())
    
    def __contains__(self, edge_id: str):
        return edge_id in self.edges
