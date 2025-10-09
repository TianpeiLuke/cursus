"""
Shared DAG definition for CradleDataLoading singleton pipeline.

This DAG definition can be used by both regular and MODS compilers,
ensuring consistency while avoiding code duplication.
"""

import logging
from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_cradle_data_loading_singleton_dag() -> PipelineDAG:
    """
    Create a singleton CradleDataLoading pipeline DAG.

    This DAG represents a single-step pipeline containing only the
    CradleDataLoading_training step, following the Zettelkasten principle
    of atomicity where each pipeline represents one atomic concept.

    Pipeline Steps:
    1) CradleDataLoading_training (single step)

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add single node
    dag.add_node("CradleDataLoading_training")

    # No edges needed for singleton

    logger.debug(
        f"Created CradleDataLoading singleton DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the CradleDataLoading singleton DAG definition.

    Returns:
        DAGMetadata: Metadata including description, complexity, features
    """
    metadata = DAGMetadata(
        description="Singleton pipeline for CradleDataLoading training data step",
        complexity="simple",
        features=["data_loading", "training", "singleton"],
        framework="generic",
        node_count=1,
        edge_count=0,
        extra_metadata={
            "name": "cradle_data_loading_singleton",
            "task_type": "data_loading",
            "entry_points": ["CradleDataLoading_training"],
            "exit_points": ["CradleDataLoading_training"],
            "required_configs": ["CradleDataLoading_training"],
            "singleton": True,
            "atomic_concept": "data_loading_training",
        },
    )

    return metadata


def validate_dag() -> bool:
    """
    Validate the CradleDataLoading singleton DAG structure.

    Returns:
        bool: True if DAG is valid

    Raises:
        ValueError: If DAG structure is invalid
    """
    dag = create_cradle_data_loading_singleton_dag()

    # Check expected nodes exist
    expected_nodes = {"CradleDataLoading_training"}

    if set(dag.nodes) != expected_nodes:
        raise ValueError(
            f"DAG nodes mismatch. Expected: {expected_nodes}, Got: {set(dag.nodes)}"
        )

    # Check no edges exist (singleton)
    if len(dag.edges) != 0:
        raise ValueError(
            f"Singleton DAG should have no edges. Got: {len(dag.edges)} edges"
        )

    # Validate no cycles (trivially true for singleton - no edges means no cycles)
    # Note: PipelineDAG doesn't have has_cycles method, but singleton can't have cycles

    return True


if __name__ == "__main__":
    # Test the DAG creation
    dag = create_cradle_data_loading_singleton_dag()
    metadata = get_dag_metadata()

    print(f"Created singleton DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
    print(f"Metadata: {metadata}")

    # Validate
    try:
        validate_dag()
        print("DAG validation passed")
    except ValueError as e:
        print(f"DAG validation failed: {e}")
