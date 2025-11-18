"""
Shared DAG definition for LightGBM Complete End-to-End Pipeline with Percentile Model Calibration and Testing Paths

This module provides the shared DAG definition for a complete LightGBM workflow
that includes training, percentile model calibration, testing (without calibration), packaging, registration,
evaluation with inference, metrics computation, and wiki generation.

The DAG includes:
1) Data Loading (training)
2) Preprocessing (training)
3) LightGBM Model Training
4) Percentile Model Calibration
5) Package Model
6) Payload Generation
7) Model Registration
8) Data Loading (calibration)
9) Preprocessing (calibration)
10) Model Evaluation (calibration)
11) Data Loading (testing)
12) Preprocessing (testing)
13) Model Inference (testing)
14) Model Metrics Computation (testing)
15) Model Wiki Generation

Key features:
- Calibration path: Uses PercentileModelCalibration instead of ModelCalibration
- Testing path: Skips PercentileModelCalibration entirely for faster evaluation
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_lightgbm_complete_e2e_with_percentile_calibration_dag() -> PipelineDAG:
    """
    Create a DAG for complete LightGBM E2E pipeline with percentile model calibration and testing paths.

    This DAG represents a complete end-to-end workflow including training,
    calibration path (with PercentileModelCalibration), testing path (without PercentileModelCalibration),
    packaging, registration, evaluation with inference, metrics computation, and wiki generation.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - Training
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("LightGBMTraining")  # LightGBM training step

    # Add all nodes - Calibration path (simplified with LightGBMModelEval)
    dag.add_node(
        "PercentileModelCalibration_calibration"
    )  # Percentile model calibration step with calibration variant
    dag.add_node("Package")  # Package step
    dag.add_node("Registration")  # MIMS registration step
    dag.add_node("Payload")  # Payload step
    dag.add_node("CradleDataLoading_calibration")  # Data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration
    dag.add_node(
        "LightGBMModelEval_calibration"
    )  # Model evaluation step (calibration) - combines inference, metrics, and wiki

    # Add all nodes - Testing path (no calibration)
    dag.add_node("CradleDataLoading_testing")  # Data load for testing
    dag.add_node("TabularPreprocessing_testing")  # Tabular preprocessing for testing
    dag.add_node("LightGBMModelInference_testing")  # Model inference step (testing)
    dag.add_node(
        "ModelMetricsComputation_testing"
    )  # Model metrics computation step (testing)
    dag.add_node("ModelWikiGenerator")  # Model wiki generator step (testing)

    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "LightGBMTraining")

    # Calibration flow
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")

    # Evaluation flow (calibration path)
    dag.add_edge("LightGBMTraining", "LightGBMModelEval_calibration")
    dag.add_edge("TabularPreprocessing_calibration", "LightGBMModelEval_calibration")

    # Percentile model calibration flow - depends on model evaluation
    dag.add_edge(
        "LightGBMModelEval_calibration", "PercentileModelCalibration_calibration"
    )

    # Testing flow (similar to calibration but skips PercentileModelCalibration)
    dag.add_edge("CradleDataLoading_testing", "TabularPreprocessing_testing")
    dag.add_edge("LightGBMTraining", "LightGBMModelInference_testing")
    dag.add_edge("TabularPreprocessing_testing", "LightGBMModelInference_testing")
    dag.add_edge("LightGBMModelInference_testing", "ModelMetricsComputation_testing")
    dag.add_edge("ModelMetricsComputation_testing", "ModelWikiGenerator")

    # Output flow (same as original complete_e2e_with_wiki_dag)
    dag.add_edge("PercentileModelCalibration_calibration", "Package")
    dag.add_edge("LightGBMTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("LightGBMTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created LightGBM complete E2E with percentile calibration and testing DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the LightGBM complete end-to-end DAG with percentile model calibration and testing paths.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Complete LightGBM end-to-end pipeline with training, percentile model calibration path, testing path (no calibration), packaging, registration, inference, metrics computation, and wiki generation",
        complexity="comprehensive",
        features=[
            "training",
            "percentile_calibration",
            "testing",
            "packaging",
            "registration",
            "inference",
            "metrics",
            "wiki_generation",
        ],
        framework="lightgbm",
        node_count=15,
        edge_count=16,
        extra_metadata={
            "name": "lightgbm_complete_e2e_with_percentile_calibration_and_testing",
            "task_type": "end_to_end",
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
                "CradleDataLoading_testing",
            ],
            "exit_points": [
                "Registration",
                "ModelWikiGenerator",
            ],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
                "CradleDataLoading_testing",
                "TabularPreprocessing_training",
                "TabularPreprocessing_calibration",
                "TabularPreprocessing_testing",
                "LightGBMTraining",
                "LightGBMModelEval_calibration",
                "LightGBMModelInference_testing",
                "ModelMetricsComputation_testing",
                "ModelWikiGenerator",
                "PercentileModelCalibration_calibration",
                "Package",
                "Payload",
                "Registration",
            ],
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the LightGBM complete end-to-end DAG with percentile model calibration and testing paths.

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
