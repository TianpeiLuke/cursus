"""
Shared DAG definition for Simple Training Pipeline

This module provides the shared DAG definition for a simple training workflow
that uses dummy data loading and includes only the training path up to PyTorch training.
This is a simplified version of the complete E2E pipeline focusing only on the
core training components.

The DAG includes:
1) Dummy Data Loading (training)
2) Tabular Preprocessing (training)
3) PyTorch Model Training
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_simple_training_dag() -> PipelineDAG:
    """
    Create a DAG for simple training pipeline.

    This DAG represents a simplified training workflow that includes only
    the core training path from dummy data loading through PyTorch training, without
    calibration, packaging, registration, or evaluation steps.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add core training nodes
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("PyTorchTraining")  # PyTorch training step

    # Training flow - simple linear path
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")

    logger.info(
        f"Created simple training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the simple training DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Simple training pipeline with dummy data loading and PyTorch training",
        complexity="simple",
        features=["dummy_data_loading", "training"],
        framework="pytorch",
        node_count=3,
        edge_count=2,
        extra_metadata={
            "name": "simple_training",
            "task_type": "training_only",
            "entry_points": ["DummyDataLoading_training"],
            "exit_points": ["PyTorchTraining"],
            "required_configs": [
                "DummyDataLoading_training",
                "TabularPreprocessing_training",
                "PyTorchTraining",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the simple training DAG.

    Args:
        dag: The DAG to validate

    Returns:
        Dict containing validation results
    """
    metadata = get_dag_metadata()

    validation_result = {"is_valid": True, "errors": [], "warnings": []}

    # Check node count
    if len(dag.nodes) != metadata.node_count:
        validation_result["errors"].append(
            f"Expected {metadata.node_count} nodes, found {len(dag.nodes)}"
        )
        validation_result["is_valid"] = False

    # Check edge count
    if len(dag.edges) != metadata.edge_count:
        validation_result["errors"].append(
            f"Expected {metadata.edge_count} edges, found {len(dag.edges)}"
        )
        validation_result["is_valid"] = False

    # Check required nodes exist
    required_configs = metadata.extra_metadata.get("required_configs", [])
    missing_nodes = set(required_configs) - set(dag.nodes)
    if missing_nodes:
        validation_result["errors"].append(f"Missing required nodes: {missing_nodes}")
        validation_result["is_valid"] = False

    # Check entry points exist
    entry_points = metadata.extra_metadata.get("entry_points", [])
    missing_entry_points = set(entry_points) - set(dag.nodes)
    if missing_entry_points:
        validation_result["errors"].append(
            f"Missing entry points: {missing_entry_points}"
        )
        validation_result["is_valid"] = False

    # Check exit points exist
    exit_points = metadata.extra_metadata.get("exit_points", [])
    missing_exit_points = set(exit_points) - set(dag.nodes)
    if missing_exit_points:
        validation_result["errors"].append(
            f"Missing exit points: {missing_exit_points}"
        )
        validation_result["is_valid"] = False

    # Validate training flow structure
    expected_edges = [
        ("DummyDataLoading_training", "TabularPreprocessing_training"),
        ("TabularPreprocessing_training", "PyTorchTraining"),
    ]

    for edge in expected_edges:
        if edge not in dag.edges:
            validation_result["errors"].append(f"Missing expected edge: {edge}")
            validation_result["is_valid"] = False

    return validation_result


def get_training_flow_info() -> Dict[str, Any]:
    """
    Get information about the training flow in this DAG.

    Returns:
        Dict containing training flow details
    """
    return {
        "flow_type": "linear_training",
        "steps": [
            {
                "step": "DummyDataLoading_training",
                "purpose": "Load dummy training data",
                "output": "Raw training dataset",
            },
            {
                "step": "TabularPreprocessing_training",
                "purpose": "Preprocess training data with train/val/test splits",
                "output": "Processed training data with splits",
            },
            {
                "step": "PyTorchTraining",
                "purpose": "Train PyTorch model using processed data",
                "output": "Trained PyTorch model artifacts",
            },
        ],
        "data_flow": "DummyDataLoading_training → TabularPreprocessing_training → PyTorchTraining",
        "characteristics": {
            "simple": True,
            "training_only": True,
            "dummy_data": True,
            "linear_flow": True,
            "no_calibration": True,
            "no_packaging": True,
            "no_registration": True,
            "no_evaluation": True,
        },
    }
