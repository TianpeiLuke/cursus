"""
Shared DAG definition for XGBoost Semi-Supervised Learning (SSL) Training Pipeline

This module provides the shared DAG definition for an XGBoost SSL workflow
that includes pretraining on labeled data, pseudo-labeling on unlabeled data,
active sample selection, and fine-tuning on combined labeled + pseudo-labeled data.

The DAG includes:
1) Data Preprocessing (training - small labeled dataset)
2) XGBoost Pretraining (on small labeled data)
3) Data Preprocessing (testing - large unlabeled dataset)
4) Model Inference (generate predictions on unlabeled data)
5) Active Sample Selection (select high-confidence pseudo-labels)
6) Pseudo Label Merge (combine labeled + pseudo-labeled data)
7) XGBoost Fine-tuning (on combined dataset)
8) Data Preprocessing (calibration)
9) Model Evaluation (on calibration data)
10) Model Calibration
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_xgboost_ssl_training_dag() -> PipelineDAG:
    """
    Create a DAG for XGBoost Semi-Supervised Learning training workflow.

    This DAG implements a complete SSL pipeline:
    1. Pretrain on small labeled dataset
    2. Generate pseudo-labels on unlabeled data
    3. Select high-confidence samples via active sampling
    4. Merge labeled + pseudo-labeled data
    5. Fine-tune model on combined dataset
    6. Evaluate and calibrate final model

    The workflow demonstrates:
    - Leveraging unlabeled data to improve model performance
    - Using confidence-based sample selection for quality pseudo-labels
    - Split-aware merge maintaining train/test/val boundaries
    - Auto-inferred split ratios for optimal data distribution

    Returns:
        PipelineDAG: The directed acyclic graph for the SSL pipeline
    """
    dag = PipelineDAG()

    # Add all nodes for SSL workflow
    # Phase 0: Data loading
    dag.add_node("CradleDataLoading_training")  # Load small labeled dataset
    dag.add_node("CradleDataLoading_testing")  # Load large unlabeled dataset
    dag.add_node("CradleDataLoading_calibration")  # Load calibration dataset

    # Phase 1: Data preparation
    dag.add_node("TabularPreprocessing_training")  # Small labeled dataset
    dag.add_node("TabularPreprocessing_testing")  # Large unlabeled dataset
    dag.add_node("TabularPreprocessing_calibration")  # Calibration dataset

    # Phase 2: Pretraining (on small labeled data)
    dag.add_node("XGBoostTraining_pretrain")  # Pretrain model on labeled data

    # Phase 3: Pseudo-labeling (on unlabeled data)
    dag.add_node(
        "XGBoostModelInference_testing"
    )  # Generate predictions on unlabeled data
    dag.add_node(
        "ActiveSampleSelection"
    )  # Select high-confidence pseudo-labeled samples

    # Phase 4: Data augmentation
    dag.add_node(
        "PseudoLabelMerge_training"
    )  # Merge labeled + pseudo-labeled data with split-aware logic

    # Phase 5: Fine-tuning (on combined data)
    dag.add_node("XGBoostTraining_finetune")  # Fine-tune on augmented dataset

    # Phase 6: Evaluation and calibration
    dag.add_node("XGBoostModelEval_calibration")  # Evaluate fine-tuned model
    dag.add_node("ModelCalibration_calibration")  # Calibrate model predictions

    # Phase 7: Model packaging and registration
    dag.add_node("Package")  # Package fine-tuned model
    dag.add_node("Payload")  # Generate payload for model testing
    dag.add_node("Registration")  # Register model in MIMS

    # ============================================================
    # Edge Definitions - SSL Training Flow
    # ============================================================

    # Phase 0 → Phase 1: Data loading flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("CradleDataLoading_testing", "TabularPreprocessing_testing")
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")

    # Phase 1 → Phase 2: Pretraining flow
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining_pretrain")

    # Phase 2 → Phase 3: Pseudo-labeling flow
    dag.add_edge("XGBoostTraining_pretrain", "XGBoostModelInference_testing")
    dag.add_edge("TabularPreprocessing_testing", "XGBoostModelInference_testing")

    # Phase 3 → Phase 4: Sample selection flow
    dag.add_edge("XGBoostModelInference_testing", "ActiveSampleSelection")

    # Phase 4 → Phase 4: Merge flow (dual inputs)
    dag.add_edge(
        "TabularPreprocessing_training", "PseudoLabelMerge_training"
    )  # Base labeled data
    dag.add_edge(
        "ActiveSampleSelection", "PseudoLabelMerge_training"
    )  # Pseudo-labeled augmentation

    # Phase 4 → Phase 5: Fine-tuning flow
    dag.add_edge("PseudoLabelMerge_training", "XGBoostTraining_finetune")

    # Phase 5 → Phase 6: Evaluation flow
    dag.add_edge("XGBoostTraining_finetune", "XGBoostModelEval_calibration")
    dag.add_edge("TabularPreprocessing_calibration", "XGBoostModelEval_calibration")

    # Phase 6: Calibration flow
    dag.add_edge("XGBoostModelEval_calibration", "ModelCalibration_calibration")

    # Phase 7: Model packaging and registration flow
    dag.add_edge(
        "ModelCalibration_calibration", "Package"
    )  # Calibrated model to package
    dag.add_edge(
        "XGBoostTraining_finetune", "Package"
    )  # Raw fine-tuned model to package
    dag.add_edge(
        "XGBoostTraining_finetune", "Payload"
    )  # Fine-tuned model for payload testing
    dag.add_edge("Package", "Registration")  # Packaged model to registration
    dag.add_edge("Payload", "Registration")  # Payload to registration

    logger.info(
        f"Created XGBoost SSL training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the XGBoost SSL training DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="XGBoost Semi-Supervised Learning pipeline with pretraining, pseudo-labeling, active sampling, merge, and fine-tuning",
        complexity="advanced",
        features=[
            "ssl",
            "pretraining",
            "pseudo_labeling",
            "active_sampling",
            "data_augmentation",
            "fine_tuning",
            "evaluation",
            "calibration",
        ],
        framework="xgboost",
        node_count=16,
        edge_count=19,
        extra_metadata={
            "name": "xgboost_ssl_training",
            "task_type": "semi_supervised_learning",
            "workflow_phases": [
                "data_loading",
                "data_preparation",
                "pretraining",
                "pseudo_labeling",
                "data_augmentation",
                "fine_tuning",
                "evaluation",
                "packaging",
                "registration",
            ],
            "entry_points": [
                "CradleDataLoading_training",
                "CradleDataLoading_testing",
                "CradleDataLoading_calibration",
            ],
            "exit_points": ["Registration"],
            "critical_steps": [
                "XGBoostTraining_pretrain",
                "ActiveSampleSelection",
                "PseudoLabelMerge_training",
                "XGBoostTraining_finetune",
            ],
            "required_configs": [
                "CradleDataLoading_training",
                "CradleDataLoading_testing",
                "CradleDataLoading_calibration",
                "TabularPreprocessing_training",
                "TabularPreprocessing_testing",
                "TabularPreprocessing_calibration",
                "XGBoostTraining_pretrain",
                "XGBoostModelInference_testing",
                "ActiveSampleSelection",
                "PseudoLabelMerge_training",
                "XGBoostTraining_finetune",
                "XGBoostModelEval_calibration",
                "ModelCalibration_calibration",
                "Package",
                "Payload",
                "Registration",
            ],
            "ssl_features": {
                "pretrain_model": "XGBoostTraining_pretrain",
                "inference_step": "XGBoostModelInference_testing",
                "selection_strategy": "confidence_threshold",  # Default, configurable
                "merge_strategy": "split_aware",  # Auto-inferred ratios
                "finetune_model": "XGBoostTraining_finetune",
            },
            "data_flow": {
                "labeled_data": "TabularPreprocessing_training → XGBoostTraining_pretrain + PseudoLabelMerge",
                "unlabeled_data": "TabularPreprocessing_testing → XGBoostModelInference_testing",
                "pseudo_labels": "ActiveSampleSelection → PseudoLabelMerge",
                "augmented_data": "PseudoLabelMerge → XGBoostTraining_finetune",
                "calibration_data": "TabularPreprocessing_calibration → XGBoostModelEval_calibration",
            },
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the XGBoost SSL training DAG.

    Performs comprehensive validation including:
    - Node and edge counts
    - Required step presence
    - Entry/exit point validation
    - Critical SSL workflow steps
    - Data flow integrity

    Args:
        dag: The DAG to validate

    Returns:
        Dict containing validation results with is_valid flag, errors, and warnings
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

    # Check critical SSL steps exist
    critical_steps = metadata.extra_metadata.get("critical_steps", [])
    missing_critical = set(critical_steps) - set(dag.nodes)
    if missing_critical:
        validation_result["errors"].append(
            f"Missing critical SSL steps: {missing_critical}"
        )
        validation_result["is_valid"] = False

    # Validate SSL-specific workflow integrity
    ssl_features = metadata.extra_metadata.get("ssl_features", {})

    # Check pretrain → inference connection
    pretrain_model = ssl_features.get("pretrain_model")
    inference_step = ssl_features.get("inference_step")
    if pretrain_model and inference_step:
        if not dag.has_edge(pretrain_model, inference_step):
            validation_result["errors"].append(
                f"Missing edge: {pretrain_model} → {inference_step} (pretrain to inference)"
            )
            validation_result["is_valid"] = False

    # Check selection → merge connection
    if not dag.has_edge("ActiveSampleSelection", "PseudoLabelMerge_training"):
        validation_result["errors"].append(
            "Missing edge: ActiveSampleSelection → PseudoLabelMerge_training"
        )
        validation_result["is_valid"] = False

    # Check merge → finetune connection
    finetune_model = ssl_features.get("finetune_model")
    if finetune_model:
        if not dag.has_edge("PseudoLabelMerge_training", finetune_model):
            validation_result["errors"].append(
                f"Missing edge: PseudoLabelMerge_training → {finetune_model} (merge to finetune)"
            )
            validation_result["is_valid"] = False

    # Check dual inputs to PseudoLabelMerge_training (base_data + augmentation_data)
    merge_predecessors = list(dag.get_predecessors("PseudoLabelMerge_training"))
    if len(merge_predecessors) != 2:
        validation_result["warnings"].append(
            f"PseudoLabelMerge_training expects 2 inputs (base_data + augmentation_data), found {len(merge_predecessors)}: {merge_predecessors}"
        )

    return validation_result
