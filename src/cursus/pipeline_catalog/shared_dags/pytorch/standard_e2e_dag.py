"""
Shared DAG definition for PyTorch Standard End-to-End Pipeline

This module provides the shared DAG definition for a complete PyTorch workflow
that includes training, evaluation, packaging, and registration.
This DAG can be used by both regular and MODS pipeline variants to ensure consistency.

The DAG includes:
1) Data Loading (training)
2) Preprocessing (training)
3) PyTorch Model Training
4) Package Model
5) Payload Generation
6) Model Registration
7) Data Loading (validation)
8) Preprocessing (validation) 
9) Model Evaluation
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_pytorch_standard_e2e_dag() -> PipelineDAG:
    """
    Create a complete end-to-end PyTorch pipeline DAG.

    This DAG represents a comprehensive workflow for training,
    evaluating, packaging, and registering a PyTorch model.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - named to match configuration names exactly
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("PyTorchTraining")  # PyTorch training step
    dag.add_node("Package")  # Package step
    dag.add_node("Payload")  # Payload step
    dag.add_node("Registration")  # Model registration step
    dag.add_node("CradleDataLoading_validation")  # Data load for validation
    dag.add_node(
        "TabularPreprocessing_validation"
    )  # Tabular preprocessing for validation
    dag.add_node("PyTorchModelEval")  # Model evaluation step

    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")

    # Output flow
    dag.add_edge("PyTorchTraining", "Package")  # Model is packaged
    dag.add_edge("PyTorchTraining", "Payload")  # Model is used for payload generation
    dag.add_edge("Package", "Registration")  # Packaged model is registered
    dag.add_edge("Payload", "Registration")  # Payload is needed for registration

    # Evaluation flow
    dag.add_edge("CradleDataLoading_validation", "TabularPreprocessing_validation")
    dag.add_edge("TabularPreprocessing_validation", "PyTorchModelEval")
    dag.add_edge("PyTorchTraining", "PyTorchModelEval")  # Model is input to evaluation
    dag.add_edge(
        "PyTorchModelEval", "Registration"
    )  # Evaluation results are needed for registration

    logger.info(
        f"Created PyTorch standard E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the PyTorch standard end-to-end DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Complete PyTorch end-to-end pipeline with training, evaluation, packaging, and registration",
        complexity="comprehensive",
        features=["training", "evaluation", "packaging", "registration"],
        framework="pytorch",
        node_count=9,
        edge_count=9,
        extra_metadata={
            "name": "pytorch_standard_e2e",
            "task_type": "end_to_end",
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_validation",
            ],
            "exit_points": ["Registration"],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_validation",
                "TabularPreprocessing_training",
                "TabularPreprocessing_validation",
                "PyTorchTraining",
                "PyTorchModelEval",
                "Package",
                "Payload",
                "Registration",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the PyTorch standard end-to-end DAG.

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

    return validation_result
