"""
Shared DAG definition for XGBoost simple pipeline.

This DAG definition can be used by both regular and MODS compilers,
ensuring consistency while avoiding code duplication.
"""

import logging
from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_xgboost_simple_dag() -> PipelineDAG:
    """
    Create a simple XGBoost training pipeline DAG.

    This DAG represents a basic XGBoost training workflow with separate paths
    for training and calibration data.

    Pipeline Steps:
    1) Data Loading (training)
    2) Preprocessing (training)
    3) XGBoost Model Training
    4) Data Loading (calibration)
    5) Preprocessing (calibration)

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add nodes
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("XGBoostTraining")  # XGBoost training step
    dag.add_node("CradleDataLoading_calibration")  # Data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration

    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")

    # Calibration flow (independent of training)
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")

    logger.debug(
        f"Created XGBoost simple DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the XGBoost simple DAG definition.

    Returns:
        DAGMetadata: Metadata including description, complexity, features
    """
    metadata = DAGMetadata(
        description="Simple XGBoost training pipeline with data loading and preprocessing",
        complexity="simple",
        features=["training", "data_loading", "preprocessing"],
        framework="xgboost",
        node_count=5,
        edge_count=3,
        extra_metadata={
            "name": "xgboost_simple",
            "task_type": "training",
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
            ],
            "exit_points": ["XGBoostTraining", "TabularPreprocessing_calibration"],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
                "TabularPreprocessing_training",
                "TabularPreprocessing_calibration",
                "XGBoostTraining",
            ],
        },
    )

    return metadata


def validate_dag() -> bool:
    """
    Validate the XGBoost simple DAG structure.

    Returns:
        bool: True if DAG is valid

    Raises:
        ValueError: If DAG structure is invalid
    """
    dag = create_xgboost_simple_dag()

    # Check expected nodes exist
    expected_nodes = {
        "CradleDataLoading_training",
        "TabularPreprocessing_training",
        "XGBoostTraining",
        "CradleDataLoading_calibration",
        "TabularPreprocessing_calibration",
    }

    if set(dag.nodes) != expected_nodes:
        raise ValueError(
            f"DAG nodes mismatch. Expected: {expected_nodes}, Got: {set(dag.nodes)}"
        )

    # Check expected edges exist
    expected_edges = {
        ("CradleDataLoading_training", "TabularPreprocessing_training"),
        ("TabularPreprocessing_training", "XGBoostTraining"),
        ("CradleDataLoading_calibration", "TabularPreprocessing_calibration"),
    }

    actual_edges = set(dag.edges)
    if actual_edges != expected_edges:
        raise ValueError(
            f"DAG edges mismatch. Expected: {expected_edges}, Got: {actual_edges}"
        )

    # Validate no cycles
    if dag.has_cycles():
        raise ValueError("DAG contains cycles")

    return True


if __name__ == "__main__":
    # Test the DAG creation
    dag = create_xgboost_simple_dag()
    metadata = get_dag_metadata()

    print(f"Created DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
    print(f"Metadata: {metadata}")

    # Validate
    try:
        validate_dag()
        print("DAG validation passed")
    except ValueError as e:
        print(f"DAG validation failed: {e}")
