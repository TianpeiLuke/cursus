"""
Shared DAG definition for Dummy End-to-End Basic Pipeline

This module provides the shared DAG definition for a basic dummy workflow
that includes training, packaging, payload preparation, and registration.
This DAG can be used by both regular and MODS pipeline variants to ensure consistency.

The DAG includes:
1) DummyTraining - Training step using a pretrained model
2) Package - Model packaging step
3) Payload - Payload testing step  
4) Registration - Model registration step

DAG Structure:
- DummyTraining -> Package
- DummyTraining -> Payload
- Package -> Registration
- Payload -> Registration
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_dummy_e2e_basic_dag() -> PipelineDAG:
    """
    Create a DAG for dummy end-to-end basic pipeline.

    This DAG represents a basic end-to-end workflow with dummy training,
    packaging, payload preparation, and registration steps.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes using proper step names from registry
    dag.add_node("DummyTraining")  # Dummy training step
    dag.add_node("Package")  # Package step
    dag.add_node("Payload")  # Payload step
    dag.add_node("Registration")  # Registration step

    # Add edges to create the diamond pattern:
    # DummyTraining -> Package, DummyTraining -> Payload
    # Package -> Registration, Payload -> Registration
    dag.add_edge("DummyTraining", "Package")
    dag.add_edge("DummyTraining", "Payload")
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created dummy E2E basic DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the dummy end-to-end basic DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Basic end-to-end pipeline with dummy training, packaging, payload preparation, and registration",
        complexity="simple",
        features=["end_to_end", "dummy", "testing", "packaging", "registration"],
        framework="dummy",
        node_count=4,
        edge_count=4,
        extra_metadata={
            "name": "dummy_e2e_basic",
            "task_type": "end_to_end",
            "entry_points": ["DummyTraining"],
            "exit_points": ["Registration"],
            "required_configs": ["DummyTraining", "Package", "Payload", "Registration"],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the dummy end-to-end basic DAG.

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
    missing_nodes = set(metadata.required_configs) - set(dag.nodes)
    if missing_nodes:
        validation_result["errors"].append(f"Missing required nodes: {missing_nodes}")
        validation_result["is_valid"] = False

    # Check entry points exist
    missing_entry_points = set(metadata.entry_points) - set(dag.nodes)
    if missing_entry_points:
        validation_result["errors"].append(
            f"Missing entry points: {missing_entry_points}"
        )
        validation_result["is_valid"] = False

    # Check exit points exist
    missing_exit_points = set(metadata.exit_points) - set(dag.nodes)
    if missing_exit_points:
        validation_result["errors"].append(
            f"Missing exit points: {missing_exit_points}"
        )
        validation_result["is_valid"] = False

    # Validate diamond structure
    # DummyTraining should have 2 outgoing edges
    dummy_training_edges = [edge for edge in dag.edges if edge[0] == "DummyTraining"]
    if len(dummy_training_edges) != 2:
        validation_result["errors"].append(
            f"DummyTraining should have 2 outgoing edges, found {len(dummy_training_edges)}"
        )
        validation_result["is_valid"] = False

    # Registration should have 2 incoming edges
    registration_edges = [edge for edge in dag.edges if edge[1] == "Registration"]
    if len(registration_edges) != 2:
        validation_result["errors"].append(
            f"Registration should have 2 incoming edges, found {len(registration_edges)}"
        )
        validation_result["is_valid"] = False

    return validation_result
