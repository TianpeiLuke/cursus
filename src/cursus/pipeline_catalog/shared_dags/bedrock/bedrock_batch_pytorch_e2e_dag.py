"""
Shared DAG definition for PyTorch End-to-End Pipeline with Bedrock Batch Processing

This module provides the shared DAG definition for a complete PyTorch workflow
that incorporates Bedrock prompt template generation and batch processing steps
for cost-efficient LLM-enhanced data processing before PyTorch training and calibration.

The DAG includes separate Bedrock batch processing paths for training and calibration:

Training Flow:
1) Dummy Data Loading (training)
2) Tabular Preprocessing (training)
3) Bedrock Prompt Template Generation (shared)
4) Bedrock Batch Processing (training) - receives data + templates
5) PyTorch Model Training

Calibration Flow:
6) Dummy Data Loading (calibration)
7) Tabular Preprocessing (calibration)
8) Bedrock Batch Processing (calibration) - receives data + templates (shared)
9) PyTorch Model Evaluation (calibration)

Final Steps:
10) Model Calibration
11) Package Model
12) Payload Generation
13) Model Registration

Key Features:
- Separate Bedrock batch processing for training and calibration data
- Shared prompt template generation for consistency
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


def create_bedrock_batch_pytorch_e2e_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock Batch-enhanced PyTorch E2E pipeline.

    This DAG represents a complete end-to-end workflow that uses Bedrock
    prompt template generation and batch processing to enhance data before
    PyTorch training, followed by calibration, packaging, registration,
    and evaluation. Provides cost-efficient processing for large datasets
    with automatic fallback to real-time processing.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - incorporating Bedrock batch processing steps with job type variants
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "BedrockPromptTemplateGeneration"
    )  # Bedrock prompt template generation
    dag.add_node(
        "BedrockBatchProcessing_training"
    )  # Bedrock batch processing step for training
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
    dag.add_node("PyTorchModelEval_calibration")  # Model evaluation step

    # Training flow with Bedrock batch processing integration
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")

    # Bedrock batch processing flow for training - two inputs to BedrockBatchProcessing_training
    dag.add_edge(
        "TabularPreprocessing_training", "BedrockBatchProcessing_training"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockBatchProcessing_training"
    )  # Template input

    # Enhanced data flows to PyTorch training
    dag.add_edge("BedrockBatchProcessing_training", "PyTorchTraining")

    # Calibration flow with Bedrock batch processing integration
    dag.add_edge("DummyDataLoading_calibration", "TabularPreprocessing_calibration")

    # Bedrock batch processing flow for calibration - two inputs to BedrockBatchProcessing_calibration
    dag.add_edge(
        "TabularPreprocessing_calibration", "BedrockBatchProcessing_calibration"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockBatchProcessing_calibration"
    )  # Template input

    # Evaluation flow
    dag.add_edge("PyTorchTraining", "PyTorchModelEval_calibration")
    dag.add_edge(
        "BedrockBatchProcessing_calibration", "PyTorchModelEval_calibration"
    )  # Use Bedrock batch-processed calibration data

    # Model calibration flow - depends on model evaluation
    dag.add_edge("PyTorchModelEval_calibration", "ModelCalibration_calibration")

    # Output flow
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("PyTorchTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("PyTorchTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created Bedrock Batch-PyTorch E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock Batch-enhanced PyTorch end-to-end DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Bedrock Batch-enhanced PyTorch end-to-end pipeline with cost-efficient LLM-based data processing, training, calibration, packaging, registration, and evaluation",
        complexity="comprehensive",
        features=[
            "dummy_data_loading",
            "bedrock_prompt_generation",
            "bedrock_batch_processing",
            "cost_optimization",
            "training",
            "calibration",
            "packaging",
            "registration",
            "evaluation",
        ],
        framework="pytorch",
        node_count=13,
        edge_count=15,
        extra_metadata={
            "name": "bedrock_batch_pytorch_e2e",
            "task_type": "end_to_end_with_batch_llm",
            "entry_points": [
                "DummyDataLoading_training",
                "DummyDataLoading_calibration",
                "BedrockPromptTemplateGeneration",
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
                    "output_target": "PyTorchTraining",
                },
                "calibration_flow": {
                    "input_sources": [
                        "TabularPreprocessing_calibration",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": "PyTorchModelEval_calibration",
                },
                "cost_optimization": {
                    "batch_processing_enabled": True,
                    "automatic_mode_selection": True,
                    "expected_cost_savings": "Up to 50% for large datasets",
                    "fallback_to_realtime": True,
                },
            },
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the Bedrock Batch-enhanced PyTorch end-to-end DAG.

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

    # Validate Bedrock batch integration structure
    bedrock_integration = metadata.extra_metadata.get("bedrock_batch_integration", {})

    # Check that BedrockBatchProcessing_training has the correct inputs
    bedrock_training_node = "BedrockBatchProcessing_training"
    if bedrock_training_node in dag.nodes:
        # Get predecessors of BedrockBatchProcessing_training
        bedrock_predecessors = set()
        for edge in dag.edges:
            if edge[1] == bedrock_training_node:
                bedrock_predecessors.add(edge[0])

        expected_inputs = set(
            bedrock_integration.get("training_flow", {}).get("input_sources", [])
        )
        if bedrock_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"BedrockBatchProcessing_training inputs mismatch. Expected: {expected_inputs}, Found: {bedrock_predecessors}"
            )

    # Check that BedrockBatchProcessing_calibration has the correct inputs
    bedrock_calibration_node = "BedrockBatchProcessing_calibration"
    if bedrock_calibration_node in dag.nodes:
        # Get predecessors of BedrockBatchProcessing_calibration
        bedrock_predecessors = set()
        for edge in dag.edges:
            if edge[1] == bedrock_calibration_node:
                bedrock_predecessors.add(edge[0])

        expected_inputs = set(
            bedrock_integration.get("calibration_flow", {}).get("input_sources", [])
        )
        if bedrock_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"BedrockBatchProcessing_calibration inputs mismatch. Expected: {expected_inputs}, Found: {bedrock_predecessors}"
            )

    # Check that BedrockBatchProcessing outputs to PyTorchTraining
    pytorch_training_node = bedrock_integration.get("training_flow", {}).get(
        "output_target"
    )
    if pytorch_training_node and pytorch_training_node in dag.nodes:
        bedrock_to_pytorch_edge = (bedrock_training_node, pytorch_training_node)
        if bedrock_to_pytorch_edge not in dag.edges:
            validation_result["errors"].append(
                f"Missing edge from {bedrock_training_node} to {pytorch_training_node}"
            )
            validation_result["is_valid"] = False

    return validation_result


def get_bedrock_batch_step_dependencies() -> Dict[str, Dict[str, Any]]:
    """
    Get the dependency specifications for Bedrock batch processing steps in this DAG.

    Returns:
        Dict mapping step names to their dependency specifications
    """
    return {
        "BedrockPromptTemplateGeneration": {
            "dependencies": {},  # No dependencies - can run independently
            "outputs": {
                "prompt_templates": "Templates for Bedrock batch processing",
                "template_metadata": "Metadata about generated templates",
                "validation_schema": "Schema for validating Bedrock responses",
            },
        },
        "BedrockBatchProcessing_training": {
            "dependencies": {
                "prompt_templates": {
                    "source_step": "BedrockPromptTemplateGeneration",
                    "output_name": "prompt_templates",
                    "required": True,
                },
                "validation_schema": {
                    "source_step": "BedrockPromptTemplateGeneration",
                    "output_name": "validation_schema",
                    "required": True,
                },
                "input_data": {
                    "source_step": "TabularPreprocessing_training",
                    "output_name": "processed_data",
                    "required": True,
                },
            },
            "outputs": {
                "processed_data": "LLM-enhanced processed data for training (batch processed)",
                "processing_metadata": "Metadata about Bedrock batch processing results",
            },
        },
        "BedrockBatchProcessing_calibration": {
            "dependencies": {
                "prompt_templates": {
                    "source_step": "BedrockPromptTemplateGeneration",
                    "output_name": "prompt_templates",
                    "required": True,
                },
                "validation_schema": {
                    "source_step": "BedrockPromptTemplateGeneration",
                    "output_name": "validation_schema",
                    "required": True,
                },
                "input_data": {
                    "source_step": "TabularPreprocessing_calibration",
                    "output_name": "processed_data",
                    "required": True,
                },
            },
            "outputs": {
                "processed_data": "LLM-enhanced processed data for calibration (batch processed)",
                "processing_metadata": "Metadata about Bedrock batch processing results",
            },
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
        "data_flow": "TabularPreprocessing output must be compatible with BedrockBatchProcessing input format expectations",
        "template_compatibility": "BedrockPromptTemplateGeneration outputs must match BedrockBatchProcessing input requirements for prompt_templates and validation_schema",
        "pytorch_integration": "BedrockBatchProcessing output format is identical to BedrockProcessing, ensuring compatibility with PyTorchTraining",
        "parallel_execution": "BedrockPromptTemplateGeneration can run in parallel with DummyDataLoading and TabularPreprocessing for better performance",
        "s3_integration": "BedrockBatchProcessing uses cursus framework patterns for S3 path management - no additional S3 configuration required",
        "batch_job_management": "Batch processing includes automatic job monitoring, result retrieval, and error handling",
        "processing_modes": "Supports three modes: 'auto' (intelligent selection), 'batch' (forced batch), 'realtime' (forced real-time)",
        "monitoring": "Add monitoring for Bedrock batch job status, processing latency, and cost savings achieved",
        "production_readiness": "Ensure fallback model is configured for production reliability and consider inference profile usage for Claude 4+ models",
    }


def get_cost_optimization_benefits() -> Dict[str, Any]:
    """
    Get detailed information about cost optimization benefits of this DAG.

    Returns:
        Dict containing cost optimization details
    """
    return {
        "batch_processing_advantages": {
            "cost_savings": "Up to 50% reduction in Bedrock API costs for large datasets",
            "scalability": "No memory limits - can process millions of records",
            "efficiency": "AWS-managed batch infrastructure with optimal resource allocation",
            "fault_tolerance": "Built-in retry and error recovery mechanisms",
        },
        "automatic_optimization": {
            "intelligent_selection": "Automatically chooses batch vs real-time based on data size",
            "threshold_based": "Default threshold of 1000 records (configurable)",
            "fallback_strategy": "Seamless fallback to real-time processing if batch fails",
            "zero_configuration": "Works with existing pipeline configurations",
        },
        "operational_benefits": {
            "monitoring": "Enhanced batch job status tracking and cost reporting",
            "reliability": "Automatic fallback ensures pipeline never fails due to batch issues",
            "compatibility": "Drop-in replacement for existing BedrockProcessing steps",
            "framework_integration": "Uses cursus S3 path patterns for seamless integration",
        },
        "recommended_use_cases": [
            "Large training datasets (>= 1000 records)",
            "Batch inference workloads",
            "Cost-sensitive production pipelines",
            "High-volume data processing scenarios",
            "Scenarios where processing latency is not critical",
        ],
    }
