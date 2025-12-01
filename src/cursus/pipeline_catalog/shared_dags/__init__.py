"""
Shared DAG Definitions for Pipeline Catalog

This module provides shared DAG creation functions that can be used by both
standard and MODS pipeline compilers, ensuring consistency while avoiding
code duplication.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from ...api.dag.base_dag import PipelineDAG

__all__ = ["DAGMetadata", "validate_dag_metadata", "get_all_shared_dags"]


class DAGMetadata(BaseModel):
    """Standard metadata structure for shared DAG definitions."""

    description: str = Field(
        ..., description="Description of the DAG's purpose and functionality"
    )
    complexity: str = Field(
        ..., description="Complexity level: simple, standard, advanced, comprehensive"
    )
    features: List[str] = Field(
        ...,
        description="List of features: training, evaluation, calibration, registration, etc.",
    )
    framework: str = Field(
        ..., description="Framework: xgboost, pytorch, generic, dummy"
    )
    node_count: int = Field(..., gt=0, description="Number of nodes in the DAG")
    edge_count: int = Field(..., ge=0, description="Number of edges in the DAG")
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata fields"
    )

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v):
        """Validate complexity level."""
        valid_complexities = {"simple", "standard", "advanced", "comprehensive"}
        if v not in valid_complexities:
            raise ValueError(
                f"Invalid complexity: {v}. Must be one of {valid_complexities}"
            )
        return v

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v):
        """Validate framework."""
        valid_frameworks = {
            "xgboost",
            "lightgbm",
            "lightgbmmt",
            "pytorch",
            "generic",
            "dummy",
        }
        if v not in valid_frameworks:
            raise ValueError(
                f"Invalid framework: {v}. Must be one of {valid_frameworks}"
            )
        return v

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Validate features list is not empty."""
        if not v:
            raise ValueError("Features list cannot be empty")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "description": self.description,
            "complexity": self.complexity,
            "features": self.features,
            "framework": self.framework,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            **self.extra_metadata,
        }


def validate_dag_metadata(metadata: DAGMetadata) -> bool:
    """
    Validate DAG metadata for consistency.

    Args:
        metadata: DAGMetadata instance to validate

    Returns:
        bool: True if metadata is valid

    Note:
        With Pydantic BaseModel, validation is automatic during instantiation.
        This function is kept for backward compatibility.
    """
    # Pydantic handles validation automatically, but we can add custom logic here
    try:
        # Trigger validation by accessing model fields
        _ = metadata.model_dump()
        return True
    except Exception as e:
        raise ValueError(f"DAG metadata validation failed: {e}")


def get_all_shared_dags() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available shared DAG definitions using auto-discovery.

    This function now uses DAGAutoDiscovery to automatically find all DAG files
    following the naming convention (create_*_dag + get_dag_metadata).

    BEFORE (Manual): Only 7 DAGs registered manually
    AFTER (Auto-discovery): All 34+ DAGs discovered automatically

    Returns:
        Dict mapping DAG identifiers to their metadata
    """
    from pathlib import Path
    from ..core.dag_discovery import DAGAutoDiscovery

    # Get the shared_dags directory (where this file is located)
    shared_dags_dir = Path(__file__).parent

    # Get src directory (4 levels up: shared_dags -> pipeline_catalog -> cursus -> src)
    # DAGAutoDiscovery expects package_root to be the src directory
    package_root = shared_dags_dir.parent.parent.parent

    # Initialize discovery for package DAGs only
    discovery = DAGAutoDiscovery(package_root=package_root)

    # Discover all DAGs
    all_dags = discovery.discover_all_dags()

    # Convert to the format expected by legacy code
    # Format: {dag_id: metadata_dict}
    shared_dags = {}
    for dag_id, dag_info in all_dags.items():
        # Handle None metadata gracefully
        metadata = dag_info.metadata or {}

        # Convert DAGInfo to metadata dict format
        shared_dags[dag_id] = {
            "description": metadata.get("description", ""),
            "complexity": dag_info.complexity,
            "features": dag_info.features,
            "framework": dag_info.framework,
            "node_count": dag_info.node_count,
            "edge_count": dag_info.edge_count,
            **metadata.get("extra_metadata", {}),
        }

    return shared_dags
