"""
Shared DAG definition for XGBoost Training with Evaluation Pipeline using Dummy Data Loading

This module provides the shared DAG definition for XGBoost training workflows
that include model evaluation and use dummy data loading instead of Cradle services.
This DAG can be used by both regular and MODS pipeline variants to ensure consistency.

The DAG includes:
1) Dummy Data Loading (training)
2) Preprocessing (training)
3) XGBoost Model Training
4) Dummy Data Loading (evaluation)
5) Preprocessing (evaluation)
6) Model Evaluation
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_xgboost_training_with_evaluation_dummy_dag() -> PipelineDAG:
    """
    Create a DAG for training and evaluating an XGBoost model with dummy data loading.

    This DAG represents a workflow that includes training an XGBoost model
    and then evaluating it with a separate evaluation dataset, using dummy
    data loading instead of Cradle services.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add nodes - using DummyDataLoading instead of CradleDataLoading
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("XGBoostTraining")  # XGBoost training step
    dag.add_node("DummyDataLoading_evaluation")  # Dummy data load for evaluation
    dag.add_node(
        "TabularPreprocessing_evaluation"
    )  # Tabular preprocessing for evaluation
    dag.add_node("XGBoostModelEval")  # Model evaluation step

    # Training flow
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")

    # Evaluation flow
    dag.add_edge("DummyDataLoading_evaluation", "TabularPreprocessing_evaluation")
    dag.add_edge("TabularPreprocessing_evaluation", "XGBoostModelEval")
    dag.add_edge("XGBoostTraining", "XGBoostModelEval")  # Model is input to evaluation

    logger.info(
        f"Created XGBoost training with evaluation dummy DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the XGBoost training with evaluation DAG using dummy data loading.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="XGBoost training pipeline with model evaluation using dummy data loading",
        complexity="standard",
        features=["dummy_data_loading", "training", "evaluation", "preprocessing"],
        framework="xgboost",
        node_count=6,
        edge_count=5,
        extra_metadata={
            "name": "xgboost_training_with_evaluation_dummy",
            "task_type": "training",
            "entry_points": [
                "DummyDataLoading_training",
                "DummyDataLoading_evaluation",
            ],
            "exit_points": ["XGBoostModelEval"],
            "required_configs": [
                "DummyDataLoading_training",
                "DummyDataLoading_evaluation",
                "TabularPreprocessing_training",
                "TabularPreprocessing_evaluation",
                "XGBoostTraining",
                "XGBoostModelEval",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the XGBoost training with evaluation DAG using dummy data loading.

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
