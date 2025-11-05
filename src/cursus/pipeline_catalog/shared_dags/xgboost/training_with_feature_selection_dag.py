"""
Shared DAG definition for XGBoost Training with Evaluation Pipeline using Advanced Preprocessing and Feature Selection

This module provides the shared DAG definition for XGBoost training workflows
that include model evaluation with advanced preprocessing steps including missing value
imputation, risk table mapping, and feature selection. This DAG can be used by both regular and MODS
pipeline variants to ensure consistency.

The DAG includes:
1) Data Loading (training)
2) Preprocessing (training)
3) Missing Value Imputation (training)
4) Risk Table Mapping (training)
5) Feature Selection (training)
6) XGBoost Model Training
7) Data Loading (evaluation)
8) Preprocessing (evaluation)
9) Missing Value Imputation (evaluation)
10) Risk Table Mapping (evaluation)
11) Feature Selection (evaluation)
12) Model Evaluation
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_xgboost_training_with_feature_selection_dag() -> PipelineDAG:
    """
    Create a DAG for training and evaluating an XGBoost model with feature selection.

    This DAG represents a workflow that includes training an XGBoost model
    with advanced preprocessing (missing value imputation, risk table mapping, and feature selection)
    and then evaluating it with a separate evaluation dataset using the same
    preprocessing steps.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add nodes for training path
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "MissingValueImputation_training"
    )  # Missing value imputation for training
    dag.add_node("RiskTableMapping_training")  # Risk table mapping for training
    dag.add_node("FeatureSelection_training")  # Feature selection for training
    dag.add_node("XGBoostTraining")  # XGBoost training step

    # Add nodes for evaluation path
    dag.add_node("CradleDataLoading_evaluation")  # Data load for evaluation
    dag.add_node(
        "TabularPreprocessing_evaluation"
    )  # Tabular preprocessing for evaluation
    dag.add_node(
        "MissingValueImputation_evaluation"
    )  # Missing value imputation for evaluation
    dag.add_node("RiskTableMapping_evaluation")  # Risk table mapping for evaluation
    dag.add_node("FeatureSelection_evaluation")  # Feature selection for evaluation
    dag.add_node("XGBoostModelEval")  # Model evaluation step

    # Training preprocessing flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "MissingValueImputation_training")
    dag.add_edge("MissingValueImputation_training", "RiskTableMapping_training")
    dag.add_edge("RiskTableMapping_training", "FeatureSelection_training")
    dag.add_edge("FeatureSelection_training", "XGBoostTraining")

    # Evaluation preprocessing flow
    dag.add_edge("CradleDataLoading_evaluation", "TabularPreprocessing_evaluation")
    dag.add_edge("TabularPreprocessing_evaluation", "MissingValueImputation_evaluation")
    dag.add_edge("MissingValueImputation_evaluation", "RiskTableMapping_evaluation")
    dag.add_edge("RiskTableMapping_evaluation", "FeatureSelection_evaluation")
    dag.add_edge("FeatureSelection_evaluation", "XGBoostModelEval")

    # Cross-connections between training and evaluation preprocessing
    dag.add_edge("MissingValueImputation_training", "MissingValueImputation_evaluation")
    dag.add_edge("RiskTableMapping_training", "RiskTableMapping_evaluation")
    dag.add_edge(
        "FeatureSelection_training", "FeatureSelection_evaluation"
    )  # Feature selection artifacts

    # Model connection
    dag.add_edge("XGBoostTraining", "XGBoostModelEval")  # Model is input to evaluation

    logger.info(
        f"Created XGBoost training with feature selection DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the XGBoost training with evaluation DAG using feature selection.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="XGBoost training pipeline with feature selection and model evaluation",
        complexity="advanced",
        features=[
            "training",
            "evaluation",
            "data_loading",
            "preprocessing",
            "missing_value_imputation",
            "risk_table_mapping",
            "feature_selection",
        ],
        framework="xgboost",
        node_count=12,
        edge_count=14,
        extra_metadata={
            "name": "xgboost_training_with_feature_selection",
            "task_type": "training",
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_evaluation",
            ],
            "exit_points": ["XGBoostModelEval"],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_evaluation",
                "TabularPreprocessing_training",
                "TabularPreprocessing_evaluation",
                "MissingValueImputation_training",
                "MissingValueImputation_evaluation",
                "RiskTableMapping_training",
                "RiskTableMapping_evaluation",
                "FeatureSelection_training",
                "FeatureSelection_evaluation",
                "XGBoostTraining",
                "XGBoostModelEval",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the XGBoost training with evaluation DAG using feature selection.

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
        ("FeatureSelection_training", "FeatureSelection_evaluation"),
        ("RiskTableMapping_training", "FeatureSelection_training"),
        ("RiskTableMapping_evaluation", "FeatureSelection_evaluation"),
        ("FeatureSelection_training", "XGBoostTraining"),
        ("FeatureSelection_evaluation", "XGBoostModelEval"),
    ]

    for source, target in feature_selection_dependencies:
        if (source, target) not in dag.edges:
            validation_result["warnings"].append(
                f"Expected edge from {source} to {target} for feature selection workflow"
            )

    return validation_result
