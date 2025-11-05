"""
Shared DAG definition for XGBoost Training with Evaluation Pipeline using Advanced Preprocessing

This module provides the shared DAG definition for XGBoost training workflows
that include model evaluation with advanced preprocessing steps including missing value
imputation and risk table mapping. This DAG can be used by both regular and MODS
pipeline variants to ensure consistency.

The DAG includes:
1) Data Loading (training)
2) Preprocessing (training)
3) Missing Value Imputation (training)
4) Risk Table Mapping (training)
5) XGBoost Model Training
6) Data Loading (evaluation)
7) Preprocessing (evaluation)
8) Missing Value Imputation (evaluation)
9) Risk Table Mapping (evaluation)
10) Model Evaluation
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_xgboost_training_with_preprocessing_dag() -> PipelineDAG:
    """
    Create a DAG for training and evaluating an XGBoost model with advanced preprocessing.

    This DAG represents a workflow that includes training an XGBoost model
    with advanced preprocessing (missing value imputation and risk table mapping)
    and then evaluating it with a separate evaluation dataset using the same
    preprocessing steps.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add nodes
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "MissingValueImputation_training"
    )  # Missing value imputation for training
    dag.add_node("RiskTableMapping_training")  # Risk table mapping for training
    dag.add_node("XGBoostTraining")  # XGBoost training step
    dag.add_node("CradleDataLoading_evaluation")  # Data load for evaluation
    dag.add_node(
        "TabularPreprocessing_evaluation"
    )  # Tabular preprocessing for evaluation
    dag.add_node(
        "MissingValueImputation_evaluation"
    )  # Missing value imputation for evaluation
    dag.add_node("RiskTableMapping_evaluation")  # Risk table mapping for evaluation
    dag.add_node("XGBoostModelEval")  # Model evaluation step

    # Training preprocessing flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "MissingValueImputation_training")
    dag.add_edge("MissingValueImputation_training", "RiskTableMapping_training")
    dag.add_edge("RiskTableMapping_training", "XGBoostTraining")

    # Evaluation preprocessing flow
    dag.add_edge("CradleDataLoading_evaluation", "TabularPreprocessing_evaluation")
    dag.add_edge("TabularPreprocessing_evaluation", "MissingValueImputation_evaluation")
    dag.add_edge("MissingValueImputation_evaluation", "RiskTableMapping_evaluation")
    dag.add_edge("RiskTableMapping_evaluation", "XGBoostModelEval")

    # Cross-connections between training and evaluation preprocessing
    dag.add_edge("MissingValueImputation_training", "MissingValueImputation_evaluation")
    dag.add_edge("RiskTableMapping_training", "RiskTableMapping_evaluation")

    # Model connection
    dag.add_edge("XGBoostTraining", "XGBoostModelEval")  # Model is input to evaluation

    logger.info(
        f"Created XGBoost training with preprocessing DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the XGBoost training with evaluation DAG using advanced preprocessing.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="XGBoost training pipeline with advanced preprocessing (missing value imputation and risk table mapping) and model evaluation",
        complexity="advanced",
        features=[
            "training",
            "evaluation",
            "data_loading",
            "preprocessing",
            "missing_value_imputation",
            "risk_table_mapping",
        ],
        framework="xgboost",
        node_count=10,
        edge_count=11,
        extra_metadata={
            "name": "xgboost_training_with_preprocessing",
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
                "XGBoostTraining",
                "XGBoostModelEval",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the XGBoost training with evaluation DAG using advanced preprocessing.

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
