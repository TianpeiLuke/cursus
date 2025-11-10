"""
Shared DAG definition for PyTorch End-to-End Pipeline with Bedrock Batch Processing and Label Ruleset

This module provides the shared DAG definition for a complete PyTorch workflow
that incorporates:
1. Bedrock prompt template generation and batch processing for cost-efficient LLM data processing
2. Label ruleset generation and execution for transparent, rule-based label transformation
3. PyTorch training and calibration

The DAG includes label ruleset steps between Bedrock processing and training/evaluation:

Training Flow:
1) Dummy Data Loading (training)
2) Tabular Preprocessing (training)
3) Bedrock Prompt Template Generation (shared)
4) Bedrock Batch Processing (training) - receives data + templates
5) Label Ruleset Generation (shared) - generates rulesets for label transformation
6) Label Ruleset Execution (training) - applies rulesets to training data
7) PyTorch Model Training

Calibration Flow:
8) Dummy Data Loading (calibration)
9) Tabular Preprocessing (calibration)
10) Bedrock Batch Processing (calibration) - receives data + templates (shared)
11) Label Ruleset Execution (calibration) - applies rulesets to calibration data
12) PyTorch Model Evaluation (calibration)

Final Steps:
13) Model Calibration
14) Package Model
15) Payload Generation
16) Model Registration

Key Features:
- Separate Bedrock batch processing for training and calibration data
- Shared prompt template generation and label ruleset generation for consistency
- Transparent rule-based label transformation with validation
- Cost-efficient LLM-enhanced data processing using AWS Bedrock batch inference
- Automatic fallback to real-time processing when batch processing is not suitable
- Up to 50% cost reduction for large datasets
- Job type variants (training vs calibration) for different processing behaviors
- Complete end-to-end workflow from data loading to model registration
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_bedrock_batch_pytorch_with_label_ruleset_e2e_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock Batch-enhanced PyTorch E2E pipeline with Label Ruleset steps.

    This DAG represents a complete end-to-end workflow that uses:
    1. Bedrock prompt template generation and batch processing for LLM-enhanced data
    2. Label ruleset generation and execution for transparent label transformation
    3. PyTorch training, followed by calibration, packaging, and registration

    The label ruleset steps sit between Bedrock processing and training/evaluation,
    providing transparent, rule-based label transformation that's easy to modify.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - incorporating Bedrock batch processing and label ruleset steps
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "BedrockPromptTemplateGeneration"
    )  # Bedrock prompt template generation (shared)
    dag.add_node(
        "BedrockBatchProcessing_training"
    )  # Bedrock batch processing step for training
    dag.add_node(
        "LabelRulesetGeneration"
    )  # Label ruleset generation (shared for training and calibration)
    dag.add_node(
        "LabelRulesetExecution_training"
    )  # Label ruleset execution for training data
    dag.add_node("PyTorchTraining")  # PyTorch training step
    dag.add_node(
        "ModelCalibration_calibration"
    )  # Model calibration step with calibration variant
    dag.add_node("Package")  # Package step
    dag.add_node("Registration")  # MIMS registration step
    dag.add_node("Payload")  # Payload step
    dag.add_node("DummyDataLoading_calibration")  # Dummy data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration
    dag.add_node(
        "BedrockBatchProcessing_calibration"
    )  # Bedrock batch processing step for calibration
    dag.add_node(
        "LabelRulesetExecution_calibration"
    )  # Label ruleset execution for calibration data
    dag.add_node("PyTorchModelEval_calibration")  # Model evaluation step

    # Training flow with Bedrock batch processing and label ruleset integration
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")

    # Bedrock batch processing flow for training - two inputs to BedrockBatchProcessing_training
    dag.add_edge(
        "TabularPreprocessing_training", "BedrockBatchProcessing_training"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockBatchProcessing_training"
    )  # Template input

    # Label ruleset execution for training - two inputs to LabelRulesetExecution_training
    dag.add_edge(
        "BedrockBatchProcessing_training", "LabelRulesetExecution_training"
    )  # Data input
    dag.add_edge(
        "LabelRulesetGeneration", "LabelRulesetExecution_training"
    )  # Ruleset input

    # Labeled data flows to PyTorch training
    dag.add_edge("LabelRulesetExecution_training", "PyTorchTraining")

    # Calibration flow with Bedrock batch processing and label ruleset integration
    dag.add_edge("DummyDataLoading_calibration", "TabularPreprocessing_calibration")

    # Bedrock batch processing flow for calibration - two inputs to BedrockBatchProcessing_calibration
    dag.add_edge(
        "TabularPreprocessing_calibration", "BedrockBatchProcessing_calibration"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockBatchProcessing_calibration"
    )  # Template input

    # Label ruleset execution for calibration - two inputs to LabelRulesetExecution_calibration
    dag.add_edge(
        "BedrockBatchProcessing_calibration", "LabelRulesetExecution_calibration"
    )  # Data input
    dag.add_edge(
        "LabelRulesetGeneration", "LabelRulesetExecution_calibration"
    )  # Ruleset input

    # Evaluation flow
    dag.add_edge("PyTorchTraining", "PyTorchModelEval_calibration")
    dag.add_edge(
        "LabelRulesetExecution_calibration", "PyTorchModelEval_calibration"
    )  # Use labeled calibration data

    # Model calibration flow - depends on model evaluation
    dag.add_edge("PyTorchModelEval_calibration", "ModelCalibration_calibration")

    # Output flow
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("PyTorchTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("PyTorchTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created Bedrock Batch-PyTorch with Label Ruleset E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock Batch-enhanced PyTorch with Label Ruleset end-to-end DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Bedrock Batch-enhanced PyTorch end-to-end pipeline with label ruleset generation/execution for transparent rule-based label transformation, training, calibration, packaging, and registration",
        complexity="comprehensive",
        features=[
            "dummy_data_loading",
            "bedrock_prompt_generation",
            "bedrock_batch_processing",
            "label_ruleset_generation",
            "label_ruleset_execution",
            "transparent_labeling",
            "cost_optimization",
            "training",
            "calibration",
            "packaging",
            "registration",
            "evaluation",
        ],
        framework="pytorch",
        node_count=16,
        edge_count=20,
        extra_metadata={
            "name": "bedrock_batch_pytorch_with_label_ruleset_e2e",
            "task_type": "end_to_end_with_batch_llm_and_label_ruleset",
            "entry_points": [
                "DummyDataLoading_training",
                "DummyDataLoading_calibration",
                "BedrockPromptTemplateGeneration",
                "LabelRulesetGeneration",
            ],
            "exit_points": ["Registration"],
            "required_configs": [
                "DummyDataLoading_training",
                "DummyDataLoading_calibration",
                "TabularPreprocessing_training",
                "TabularPreprocessing_calibration",
                "BedrockPromptTemplateGeneration",
                "BedrockBatchProcessing_training",
                "BedrockBatchProcessing_calibration",
                "LabelRulesetGeneration",
                "LabelRulesetExecution_training",
                "LabelRulesetExecution_calibration",
                "PyTorchTraining",
                "PyTorchModelEval_calibration",
                "ModelCalibration_calibration",
                "Package",
                "Payload",
                "Registration",
            ],
            "bedrock_batch_integration": {
                "template_generation": "BedrockPromptTemplateGeneration",
                "training_processing": "BedrockBatchProcessing_training",
                "calibration_processing": "BedrockBatchProcessing_calibration",
                "training_flow": {
                    "input_sources": [
                        "TabularPreprocessing_training",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": "LabelRulesetExecution_training",
                },
                "calibration_flow": {
                    "input_sources": [
                        "TabularPreprocessing_calibration",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": "LabelRulesetExecution_calibration",
                },
                "cost_optimization": {
                    "batch_processing_enabled": True,
                    "automatic_mode_selection": True,
                    "expected_cost_savings": "Up to 50% for large datasets",
                    "fallback_to_realtime": True,
                },
            },
            "label_ruleset_integration": {
                "ruleset_generation": "LabelRulesetGeneration",
                "training_execution": "LabelRulesetExecution_training",
                "calibration_execution": "LabelRulesetExecution_calibration",
                "training_flow": {
                    "input_sources": [
                        "BedrockBatchProcessing_training",
                        "LabelRulesetGeneration",
                    ],
                    "output_target": "PyTorchTraining",
                },
                "calibration_flow": {
                    "input_sources": [
                        "BedrockBatchProcessing_calibration",
                        "LabelRulesetGeneration",
                    ],
                    "output_target": "PyTorchModelEval_calibration",
                },
                "key_features": {
                    "transparent_rules": "Easy-to-read format for rulesets",
                    "validation": "Field availability and label category validation",
                    "shared_rulesets": "Same ruleset used for training and calibration consistency",
                    "format_preservation": "Maintains CSV/TSV/Parquet format through pipeline",
                },
            },
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the Bedrock Batch with Label Ruleset-enhanced PyTorch end-to-end DAG.

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

    # Validate Label Ruleset integration structure
    label_integration = metadata.extra_metadata.get("label_ruleset_integration", {})

    # Check that LabelRulesetExecution_training has the correct inputs
    label_training_node = "LabelRulesetExecution_training"
    if label_training_node in dag.nodes:
        label_predecessors = set()
        for edge in dag.edges:
            if edge[1] == label_training_node:
                label_predecessors.add(edge[0])

        expected_inputs = set(
            label_integration.get("training_flow", {}).get("input_sources", [])
        )
        if label_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"LabelRulesetExecution_training inputs mismatch. Expected: {expected_inputs}, Found: {label_predecessors}"
            )

    # Check that LabelRulesetExecution_calibration has the correct inputs
    label_calibration_node = "LabelRulesetExecution_calibration"
    if label_calibration_node in dag.nodes:
        label_predecessors = set()
        for edge in dag.edges:
            if edge[1] == label_calibration_node:
                label_predecessors.add(edge[0])

        expected_inputs = set(
            label_integration.get("calibration_flow", {}).get("input_sources", [])
        )
        if label_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"LabelRulesetExecution_calibration inputs mismatch. Expected: {expected_inputs}, Found: {label_predecessors}"
            )

    return validation_result


def get_label_ruleset_step_dependencies() -> Dict[str, Dict[str, Any]]:
    """
    Get the dependency specifications for Label Ruleset steps in this DAG.

    Returns:
        Dict mapping step names to their dependency specifications
    """
    return {
        "LabelRulesetGeneration": {
            "dependencies": {},  # No dependencies - can run independently
            "outputs": {
                "validated_ruleset": "Validated ruleset for label transformation",
                "ruleset_metadata": "Metadata about generated ruleset",
                "field_validation_schema": "Schema for validating field availability",
            },
            "description": "Generates transparent, rule-based label transformation rulesets",
        },
        "LabelRulesetExecution_training": {
            "dependencies": {
                "validated_ruleset": {
                    "source_step": "LabelRulesetGeneration",
                    "output_name": "validated_ruleset",
                    "required": True,
                },
                "input_data": {
                    "source_step": "BedrockBatchProcessing_training",
                    "output_name": "processed_data",
                    "required": True,
                },
            },
            "outputs": {
                "processed_data": "Data with rule-based labels applied (training)",
                "execution_report": "Statistics on rule matches and label distribution",
            },
            "description": "Applies validated rulesets to training data to generate labels",
        },
        "LabelRulesetExecution_calibration": {
            "dependencies": {
                "validated_ruleset": {
                    "source_step": "LabelRulesetGeneration",
                    "output_name": "validated_ruleset",
                    "required": True,
                },
                "input_data": {
                    "source_step": "BedrockBatchProcessing_calibration",
                    "output_name": "processed_data",
                    "required": True,
                },
            },
            "outputs": {
                "processed_data": "Data with rule-based labels applied (calibration)",
                "execution_report": "Statistics on rule matches and label distribution",
            },
            "description": "Applies validated rulesets to calibration data to generate labels",
        },
    }


def get_integration_notes() -> Dict[str, str]:
    """
    Get integration notes for implementing this DAG.

    Returns:
        Dict containing implementation notes and considerations
    """
    return {
        "bedrock_batch_setup": "Ensure Bedrock batch processing steps are configured with appropriate IAM role ARN for batch inference jobs",
        "cost_optimization": "Batch processing provides up to 50% cost savings for large datasets (>= 1000 records by default)",
        "automatic_fallback": "BedrockBatchProcessing automatically falls back to real-time processing when batch processing is not suitable",
        "label_ruleset_transparency": "Label rulesets use easy-to-read format for transparent and modifiable label transformation logic",
        "label_validation": "LabelRulesetGeneration validates field availability and label categories before execution",
        "shared_rulesets": "Same ruleset is used for both training and calibration to ensure consistency",
        "format_preservation": "Label ruleset execution maintains input data format (CSV/TSV/Parquet) through the pipeline",
        "data_flow": "Bedrock processing output must be compatible with LabelRulesetExecution input format expectations",
        "ruleset_structure": "Rulesets support priority-based evaluation, logical operators (all_of, any_of, none_of), and comprehensive validation",
        "parallel_execution": "BedrockPromptTemplateGeneration and LabelRulesetGeneration can run in parallel with data loading steps",
        "execution_statistics": "LabelRulesetExecution provides detailed statistics on rule matches and label distribution",
        "field_validation": "Execution-time validation ensures all fields referenced in rules exist in actual data",
        "monitoring": "Add monitoring for batch job status, ruleset execution success rates, and label distribution",
        "production_readiness": "Ensure fallback model is configured for Bedrock and test ruleset validation with production data schemas",
    }


def get_pipeline_benefits() -> Dict[str, Any]:
    """
    Get detailed information about benefits of this pipeline architecture.

    Returns:
        Dict containing pipeline benefits
    """
    return {
        "cost_optimization": {
            "bedrock_batch_savings": "Up to 50% reduction in Bedrock API costs for large datasets",
            "scalability": "No memory limits - can process millions of records",
            "efficiency": "AWS-managed batch infrastructure with optimal resource allocation",
        },
        "transparency_and_control": {
            "transparent_rules": "Easy-to-read ruleset format makes label logic clear and auditable",
            "easy_modification": "Rulesets can be modified without code changes",
            "validation_safety": "Comprehensive validation prevents runtime errors from invalid rules",
            "consistent_labeling": "Same ruleset ensures consistent labels across training and calibration",
        },
        "operational_benefits": {
            "format_preservation": "Maintains data format (CSV/TSV/Parquet) throughout entire pipeline",
            "execution_statistics": "Detailed reporting on rule matches and label distribution",
            "automatic_fallback": "Bedrock batch processing falls back to real-time when needed",
            "zero_configuration": "Works with existing pipeline configurations",
        },
        "production_readiness": {
            "field_validation": "Runtime validation ensures fields exist before rule evaluation",
            "error_handling": "Fail-safe approach continues processing even with individual rule errors",
            "monitoring_support": "Enhanced logging and statistics for production monitoring",
            "framework_integration": "Seamless integration with cursus S3 path patterns",
        },
        "recommended_use_cases": [
            "Large training datasets (>= 1000 records) requiring LLM enhancement",
            "Scenarios requiring transparent, auditable label transformation logic",
            "Pipelines where label rules need frequent modification",
            "Production workflows requiring consistent labeling across splits",
            "Cost-sensitive ML pipelines with batch inference workloads",
        ],
    }
