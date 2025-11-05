"""
Shared DAG definition for Dummy Training with Inference and Wiki Generation Pipeline

This module provides the shared DAG definition for a pipeline that uses dummy training
(pretrained model) and focuses on inference, metrics computation, and wiki generation.
This DAG can be used by both regular and MODS pipeline variants to ensure consistency.

The DAG includes:
1) Dummy Data Loading (calibration)
2) Preprocessing (calibration)
3) Dummy Training (uses pretrained model)
4) Model Inference
5) Model Metrics Computation
6) Model Wiki Generation
7) Model Calibration
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_dummy_inference_with_wiki_dag() -> PipelineDAG:
    """
    Create a DAG for dummy training with inference and wiki generation.

    This DAG represents a workflow that uses a pretrained model (dummy training)
    and focuses on inference, metrics computation, and documentation generation.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add nodes
    dag.add_node("DummyDataLoading_calibration")  # Dummy data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration
    dag.add_node("DummyTraining")  # Dummy training step (uses pretrained model)
    dag.add_node("XGBoostModelInference")  # Model inference step
    dag.add_node("ModelMetricsComputation")  # Model metrics computation step
    dag.add_node("ModelWikiGenerator")  # Model wiki generator step
    dag.add_node("ModelCalibration_calibration")  # Model calibration step

    # Data loading and preprocessing flow
    dag.add_edge("DummyDataLoading_calibration", "TabularPreprocessing_calibration")

    # Inference and evaluation flow
    dag.add_edge("DummyTraining", "XGBoostModelInference")
    dag.add_edge("TabularPreprocessing_calibration", "XGBoostModelInference")
    dag.add_edge("XGBoostModelInference", "ModelMetricsComputation")
    dag.add_edge("ModelMetricsComputation", "ModelWikiGenerator")

    # Model calibration flow
    dag.add_edge("XGBoostModelInference", "ModelCalibration_calibration")

    logger.info(
        f"Created dummy inference with wiki DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the dummy training with inference and wiki generation DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Dummy training pipeline with inference, metrics computation, and wiki generation",
        complexity="standard",
        features=[
            "dummy_training",
            "dummy_data_loading",
            "inference",
            "metrics",
            "wiki_generation",
            "calibration",
        ],
        framework="dummy",
        node_count=7,
        edge_count=6,
        extra_metadata={
            "name": "dummy_inference_with_wiki",
            "task_type": "inference",
            "entry_points": [
                "DummyDataLoading_calibration",
                "DummyTraining",
            ],
            "exit_points": ["ModelWikiGenerator", "ModelCalibration_calibration"],
            "required_configs": [
                "DummyDataLoading_calibration",
                "TabularPreprocessing_calibration",
                "DummyTraining",
                "XGBoostModelInference",
                "ModelMetricsComputation",
                "ModelWikiGenerator",
                "ModelCalibration_calibration",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the dummy training with inference and wiki generation DAG.

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
