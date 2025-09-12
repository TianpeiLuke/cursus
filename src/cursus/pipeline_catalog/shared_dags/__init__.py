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
        valid_frameworks = {"xgboost", "pytorch", "generic", "dummy"}
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
    Get information about all available shared DAG definitions.

    Returns:
        Dict mapping DAG identifiers to their metadata
    """
    shared_dags = {}

    # XGBoost DAGs
    try:
        from .xgboost.simple_dag import get_dag_metadata as xgb_simple_meta

        shared_dags["xgboost.simple"] = xgb_simple_meta()
    except ImportError:
        pass

    try:
        from .xgboost.training_with_calibration_dag import (
            get_dag_metadata as xgb_training_meta,
        )

        shared_dags["xgboost.training_calibrated"] = xgb_training_meta()
    except ImportError:
        pass

    try:
        from .xgboost.training_with_evaluation_dag import (
            get_dag_metadata as xgb_training_eval_meta,
        )

        shared_dags["xgboost.training_evaluation"] = xgb_training_eval_meta()
    except ImportError:
        pass

    try:
        from .xgboost.complete_e2e_dag import get_dag_metadata as xgb_e2e_meta

        shared_dags["xgboost.complete_e2e"] = xgb_e2e_meta()
    except ImportError:
        pass

    # PyTorch DAGs
    try:
        from .pytorch.training_dag import get_dag_metadata as pytorch_training_meta

        shared_dags["pytorch.training"] = pytorch_training_meta()
    except ImportError:
        pass

    try:
        from .pytorch.standard_e2e_dag import get_dag_metadata as pytorch_e2e_meta

        shared_dags["pytorch.standard_e2e"] = pytorch_e2e_meta()
    except ImportError:
        pass

    # Dummy DAGs
    try:
        from .dummy.e2e_basic_dag import get_dag_metadata as dummy_e2e_basic_meta

        shared_dags["dummy.e2e_basic"] = dummy_e2e_basic_meta()
    except ImportError:
        pass

    return shared_dags
