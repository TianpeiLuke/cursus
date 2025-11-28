"""
Shared DAG definition for LightGBMMT End-to-End Pipeline with Label Ruleset

This module provides the shared DAG definition for a complete LightGBMMT workflow
that incorporates:
1. Label ruleset generation and execution for transparent, rule-based label transformation
2. Multi-task LightGBMMT training and evaluation
3. Model calibration, packaging, and registration

The DAG includes label ruleset steps between preprocessing and training/evaluation:

Training Flow:
1) Cradle Data Loading (training)
2) Tabular Preprocessing (training)
3) Label Ruleset Generation (shared) - generates rulesets for label transformation
4) Label Ruleset Execution (training) - applies rulesets to training data
5) LightGBMMT Model Training (multi-task)

Calibration Flow:
6) Cradle Data Loading (calibration)
7) Tabular Preprocessing (calibration)
8) Label Ruleset Execution (calibration) - applies rulesets to calibration data
9) LightGBMMT Model Evaluation (calibration, multi-task)

Final Steps:
10) Model Calibration
11) Package Model
12) Payload Generation
13) Model Registration

Key Features:
- Shared label ruleset generation for consistency across training and calibration
- Transparent rule-based label transformation with validation
- Multi-task learning support with multiple target labels
- Complete end-to-end workflow from data loading to model registration
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_lightgbmmt_with_label_ruleset_e2e_dag() -> PipelineDAG:
    """
    Create a DAG for LightGBMMT E2E pipeline with Label Ruleset steps.

    This DAG represents a complete end-to-end workflow that uses:
    1. Label ruleset generation and execution for transparent label transformation
    2. Multi-task LightGBMMT training and evaluation
    3. Model calibration, packaging, and registration

    The label ruleset steps sit between preprocessing and training/evaluation,
    providing transparent, rule-based label transformation that's easy to modify
    while supporting multiple task labels.

    Returns:
        PipelineDAG: The directed acyclic graph for the multi-task pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - incorporating label ruleset steps
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "LabelRulesetGeneration"
    )  # Label ruleset generation (shared for training and calibration)
    dag.add_node(
        "LabelRulesetExecution_training"
    )  # Label ruleset execution for training data
    dag.add_node("LightGBMMTTraining")  # LightGBMMT multi-task training step
    dag.add_node(
        "ModelCalibration_calibration"
    )  # Model calibration step with calibration variant
    dag.add_node("Package")  # Package step
    dag.add_node("Registration")  # MIMS registration step
    dag.add_node("Payload")  # Payload step
    dag.add_node("CradleDataLoading_calibration")  # Data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration
    dag.add_node(
        "LabelRulesetExecution_calibration"
    )  # Label ruleset execution for calibration data
    dag.add_node("LightGBMMTModelEval_calibration")  # Multi-task model evaluation step

    # Training flow with label ruleset integration
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")

    # Label ruleset execution for training - two inputs to LabelRulesetExecution_training
    dag.add_edge(
        "TabularPreprocessing_training", "LabelRulesetExecution_training"
    )  # Data input
    dag.add_edge(
        "LabelRulesetGeneration", "LabelRulesetExecution_training"
    )  # Ruleset input

    # Labeled data flows to LightGBMMT training
    dag.add_edge("LabelRulesetExecution_training", "LightGBMMTTraining")

    # Calibration flow with label ruleset integration
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")

    # Label ruleset execution for calibration - two inputs to LabelRulesetExecution_calibration
    dag.add_edge(
        "TabularPreprocessing_calibration", "LabelRulesetExecution_calibration"
    )  # Data input
    dag.add_edge(
        "LabelRulesetGeneration", "LabelRulesetExecution_calibration"
    )  # Ruleset input

    # Evaluation flow
    dag.add_edge("LightGBMMTTraining", "LightGBMMTModelEval_calibration")
    dag.add_edge(
        "LabelRulesetExecution_calibration", "LightGBMMTModelEval_calibration"
    )  # Use labeled calibration data

    # Model calibration flow - depends on model evaluation
    dag.add_edge("LightGBMMTModelEval_calibration", "ModelCalibration_calibration")

    # Output flow
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge(
        "LightGBMMTTraining", "Package"
    )  # Raw model is also input to packaging
    dag.add_edge("LightGBMMTTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created LightGBMMT with Label Ruleset E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the LightGBMMT with Label Ruleset end-to-end DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="LightGBMMT multi-task end-to-end pipeline with label ruleset generation/execution for transparent rule-based label transformation, training, calibration, packaging, and registration",
        complexity="comprehensive",
        features=[
            "label_ruleset_generation",
            "label_ruleset_execution",
            "transparent_labeling",
            "multi_task_training",
            "multi_task_evaluation",
            "calibration",
            "packaging",
            "registration",
        ],
        framework="lightgbmmt",
        node_count=13,
        edge_count=15,
        extra_metadata={
            "name": "lightgbmmt_with_label_ruleset_e2e",
            "task_type": "multi_task_end_to_end_with_label_ruleset",
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
                "LabelRulesetGeneration",
            ],
            "exit_points": ["Registration"],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_calibration",
                "TabularPreprocessing_training",
                "TabularPreprocessing_calibration",
                "LabelRulesetGeneration",
                "LabelRulesetExecution_training",
                "LabelRulesetExecution_calibration",
                "LightGBMMTTraining",
                "LightGBMMTModelEval_calibration",
                "ModelCalibration_calibration",
                "Package",
                "Payload",
                "Registration",
            ],
            "label_ruleset_integration": {
                "ruleset_generation": "LabelRulesetGeneration",
                "training_execution": "LabelRulesetExecution_training",
                "calibration_execution": "LabelRulesetExecution_calibration",
                "training_flow": {
                    "input_sources": [
                        "TabularPreprocessing_training",
                        "LabelRulesetGeneration",
                    ],
                    "output_target": "LightGBMMTTraining",
                },
                "calibration_flow": {
                    "input_sources": [
                        "TabularPreprocessing_calibration",
                        "LabelRulesetGeneration",
                    ],
                    "output_target": "LightGBMMTModelEval_calibration",
                },
                "key_features": {
                    "transparent_rules": "Easy-to-read format for rulesets",
                    "validation": "Field availability and label category validation",
                    "shared_rulesets": "Same ruleset used for training and calibration consistency",
                    "multi_task_support": "Supports multiple task labels for multi-task learning",
                    "format_preservation": "Maintains CSV/TSV/Parquet format through pipeline",
                },
            },
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the LightGBMMT with Label Ruleset end-to-end DAG.

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
