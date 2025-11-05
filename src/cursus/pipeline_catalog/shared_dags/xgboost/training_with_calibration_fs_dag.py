"""
Shared DAG definition for XGBoost Training with Calibration Pipeline using Feature Selection

This module provides the shared DAG definition for XGBoost training workflows
that include model calibration with feature selection. This DAG can be used by both regular and MODS
pipeline variants to ensure consistency.

The DAG includes:
1) Data Loading (training)
2) Preprocessing (training)
3) Feature Selection (training)
4) XGBoost Model Training
5) Model Calibration
6) Data Loading (calibration)
7) Preprocessing (calibration)
8) Feature Selection (calibration)
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_xgboost_training_with_calibration_fs_dag() -> PipelineDAG:
    """
    Create a DAG for training and calibrating an XGBoost model with feature selection.

    This DAG represents a workflow that includes training an XGBoost model
    with feature selection and then calibrating it with a separate calibration dataset
    using the same feature selection.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add nodes for training path
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("FeatureSelection_training")  # Feature selection for training
    dag.add_node("XGBoostTraining")  # XGBoost training step
    dag.add_node(
        "ModelCalibration_training"
    )  # Model calibration step with training variant

    # Add nodes for calibration path
    dag.add_node("CradleDataLoading_calibration")  # Data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration
    dag.add_node("FeatureSelection_calibration")  # Feature selection for calibration

    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "FeatureSelection_training")
    dag.add_edge("FeatureSelection_training", "XGBoostTraining")
    dag.add_edge("XGBoostTraining", "ModelCalibration_training")

    # Calibration flow
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
    dag.add_edge("TabularPreprocessing_calibration", "FeatureSelection_calibration")

    # Connect calibration data to model calibration
    dag.add_edge("FeatureSelection_calibration", "ModelCalibration_training")

    # Feature selection artifacts connection (training -> calibration)
    dag.add_edge("FeatureSelection_training", "FeatureSelection_calibration")

    logger.info(
        f"Created XGBoost training with calibration FS DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the XGBoost training with calibration FS DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="XGBoost training pipeline with feature selection and model calibration",
        complexity="advanced",
        features=[
            "training",
            "calibration",
            "data_loading",
            "preprocessing",
            "feature_selection",
        ],
        framework="xgboost",
        node_count=8,
        edge_count=8,
        extra_metadata={
            "name": "xgboost_training_with_calibration_fs",
            "task_type": "training",
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
            ],
            "exit_points": ["ModelCalibration_training"],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
                "TabularPreprocessing_training",
                "TabularPreprocessing_calibration",
                "FeatureSelection_training",
                "FeatureSelection_calibration",
                "XGBoostTraining",
                "ModelCalibration_training",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the XGBoost training with calibration FS DAG.

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

    # Validate feature selection dependencies
    feature_selection_dependencies = [
        ("FeatureSelection_training", "FeatureSelection_calibration"),
        ("TabularPreprocessing_training", "FeatureSelection_training"),
        ("TabularPreprocessing_calibration", "FeatureSelection_calibration"),
        ("FeatureSelection_training", "XGBoostTraining"),
        ("FeatureSelection_calibration", "ModelCalibration_training"),
    ]

    for source, target in feature_selection_dependencies:
        if (source, target) not in dag.edges:
            validation_result["warnings"].append(
                f"Expected edge from {source} to {target} for feature selection workflow"
            )

    return validation_result
