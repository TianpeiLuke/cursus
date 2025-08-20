"""
Shared DAG Definitions for Pipeline Catalog

This module provides shared DAG creation functions that can be used by both
standard and MODS pipeline compilers, ensuring consistency while avoiding
code duplication.
"""

from typing import Dict, Any, List
from ...api.dag.base_dag import PipelineDAG

__all__ = [
    "DAGMetadata",
    "validate_dag_metadata",
    "get_all_shared_dags"
]


class DAGMetadata:
    """Standard metadata structure for shared DAG definitions."""
    
    def __init__(
        self,
        description: str,
        complexity: str,
        features: List[str],
        framework: str,
        node_count: int,
        edge_count: int,
        **kwargs
    ):
        self.description = description
        self.complexity = complexity  # simple, standard, advanced
        self.features = features  # training, evaluation, calibration, registration, etc.
        self.framework = framework  # xgboost, pytorch, generic
        self.node_count = node_count
        self.edge_count = edge_count
        self.extra_metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "description": self.description,
            "complexity": self.complexity,
            "features": self.features,
            "framework": self.framework,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            **self.extra_metadata
        }


def validate_dag_metadata(metadata: DAGMetadata) -> bool:
    """
    Validate DAG metadata for consistency.
    
    Args:
        metadata: DAGMetadata instance to validate
        
    Returns:
        bool: True if metadata is valid
        
    Raises:
        ValueError: If metadata is invalid
    """
    valid_complexities = {"simple", "standard", "advanced"}
    valid_frameworks = {"xgboost", "pytorch", "generic"}
    
    if metadata.complexity not in valid_complexities:
        raise ValueError(f"Invalid complexity: {metadata.complexity}. Must be one of {valid_complexities}")
    
    if metadata.framework not in valid_frameworks:
        raise ValueError(f"Invalid framework: {metadata.framework}. Must be one of {valid_frameworks}")
    
    if metadata.node_count <= 0:
        raise ValueError(f"Invalid node_count: {metadata.node_count}. Must be positive")
    
    if metadata.edge_count < 0:
        raise ValueError(f"Invalid edge_count: {metadata.edge_count}. Must be non-negative")
    
    if not isinstance(metadata.features, list) or not metadata.features:
        raise ValueError("Features must be a non-empty list")
    
    return True


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
        from .xgboost.training_dag import get_dag_metadata as xgb_training_meta
        shared_dags["xgboost.training"] = xgb_training_meta()
    except ImportError:
        pass
    
    try:
        from .xgboost.end_to_end_dag import get_dag_metadata as xgb_e2e_meta
        shared_dags["xgboost.end_to_end"] = xgb_e2e_meta()
    except ImportError:
        pass
    
    # PyTorch DAGs
    try:
        from .pytorch.training_dag import get_dag_metadata as pytorch_training_meta
        shared_dags["pytorch.training"] = pytorch_training_meta()
    except ImportError:
        pass
    
    try:
        from .pytorch.end_to_end_dag import get_dag_metadata as pytorch_e2e_meta
        shared_dags["pytorch.end_to_end"] = pytorch_e2e_meta()
    except ImportError:
        pass
    
    return shared_dags
