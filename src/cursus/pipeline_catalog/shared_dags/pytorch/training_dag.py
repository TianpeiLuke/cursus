"""
Shared DAG definition for PyTorch training pipeline.

This DAG definition can be used by both regular and MODS compilers,
ensuring consistency while avoiding code duplication.
"""

import logging
from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_pytorch_training_dag() -> PipelineDAG:
    """
    Create a PyTorch training pipeline DAG.

    This DAG represents a workflow that includes training a PyTorch model
    and evaluating it with a validation dataset.

    Pipeline Steps:
    1) Data Loading (training)
    2) Preprocessing (training)
    3) PyTorch Model Training
    4) Data Loading (validation)
    5) Preprocessing (validation)
    6) Model Evaluation

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add nodes
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Preprocessing for training
    dag.add_node("PyTorchTraining")  # PyTorch training step
    dag.add_node("CradleDataLoading_validation")  # Data load for validation
    dag.add_node("TabularPreprocessing_validation")  # Preprocessing for validation
    dag.add_node("PyTorchModelEval")  # Model evaluation step

    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")

    # Evaluation flow
    dag.add_edge("CradleDataLoading_validation", "TabularPreprocessing_validation")
    dag.add_edge("TabularPreprocessing_validation", "PyTorchModelEval")
    dag.add_edge("PyTorchTraining", "PyTorchModelEval")  # Model is input to evaluation

    logger.debug(
        f"Created PyTorch training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the PyTorch training DAG definition.

    Returns:
        DAGMetadata: Metadata including description, complexity, features
    """
    metadata = DAGMetadata(
        description="PyTorch training pipeline with model evaluation",
        complexity="standard",
        features=["training", "evaluation", "data_loading", "preprocessing"],
        framework="pytorch",
        node_count=6,
        edge_count=5,
        extra_metadata={
            "name": "pytorch_training",
            "task_type": "training",
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_validation",
            ],
            "exit_points": ["PyTorchModelEval"],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_validation",
                "TabularPreprocessing_training",
                "TabularPreprocessing_validation",
                "PyTorchTraining",
                "PyTorchModelEval",
            ],
        },
    )

    return metadata


def validate_dag() -> bool:
    """
    Validate the PyTorch training DAG structure.

    Returns:
        bool: True if DAG is valid

    Raises:
        ValueError: If DAG structure is invalid
    """
    dag = create_pytorch_training_dag()

    # Check expected nodes exist
    expected_nodes = {
        "CradleDataLoading_training",
        "TabularPreprocessing_training",
        "PyTorchTraining",
        "CradleDataLoading_validation",
        "TabularPreprocessing_validation",
        "PyTorchModelEval",
    }

    if set(dag.nodes) != expected_nodes:
        raise ValueError(
            f"DAG nodes mismatch. Expected: {expected_nodes}, Got: {set(dag.nodes)}"
        )

    # Check expected edges exist
    expected_edges = {
        ("CradleDataLoading_training", "TabularPreprocessing_training"),
        ("TabularPreprocessing_training", "PyTorchTraining"),
        ("CradleDataLoading_validation", "TabularPreprocessing_validation"),
        ("TabularPreprocessing_validation", "PyTorchModelEval"),
        ("PyTorchTraining", "PyTorchModelEval"),
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
    dag = create_pytorch_training_dag()
    metadata = get_dag_metadata()

    print(f"Created DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
    print(f"Metadata: {metadata}")

    # Validate
    try:
        validate_dag()
        print("DAG validation passed")
    except ValueError as e:
        print(f"DAG validation failed: {e}")
