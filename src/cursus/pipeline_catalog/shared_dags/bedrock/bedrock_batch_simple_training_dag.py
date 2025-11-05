"""
Shared DAG definition for Bedrock Batch-Enhanced Simple Training Pipeline

This module provides the shared DAG definition for a simple training workflow
that uses dummy data loading, Bedrock batch LLM enhancement, and PyTorch training.
This combines the simplicity of the basic training pipeline with the cost-efficient
power of Bedrock batch processing.

The DAG includes:
1) Dummy Data Loading (training)
2) Tabular Preprocessing (training)
3) Bedrock Prompt Template Generation
4) Bedrock Batch Processing (training) - cost-efficient LLM enhancement
5) PyTorch Model Training

Key Features:
- Simple training-focused workflow without calibration/packaging complexity
- Cost-efficient Bedrock batch processing with automatic fallback
- Up to 50% cost reduction for large datasets
- Intelligent processing mode selection (auto, batch, realtime)
- Seamless integration with existing training pipelines
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_bedrock_batch_simple_training_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock Batch-enhanced simple training pipeline.

    This DAG represents a simplified training workflow that includes
    cost-efficient Bedrock batch LLM enhancement between preprocessing
    and PyTorch training, without calibration, packaging, registration,
    or evaluation steps. Provides automatic fallback to real-time processing.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add core training nodes with Bedrock batch enhancement
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "BedrockPromptTemplateGeneration"
    )  # Bedrock prompt template generation
    dag.add_node(
        "BedrockBatchProcessing_training"
    )  # Bedrock batch processing step for training
    dag.add_node("PyTorchTraining")  # PyTorch training step

    # Training flow with Bedrock batch enhancement
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

    logger.info(
        f"Created Bedrock Batch-enhanced simple training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock Batch-enhanced simple training DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Bedrock Batch-enhanced simple training pipeline with dummy data loading, cost-efficient LLM processing, and PyTorch training",
        complexity="moderate",
        features=[
            "dummy_data_loading",
            "bedrock_prompt_generation",
            "bedrock_batch_processing",
            "cost_optimization",
            "training",
        ],
        framework="pytorch",
        node_count=5,
        edge_count=4,
        extra_metadata={
            "name": "bedrock_batch_simple_training",
            "task_type": "training_with_batch_llm",
            "entry_points": [
                "DummyDataLoading_training",
                "BedrockPromptTemplateGeneration",
            ],
            "exit_points": ["PyTorchTraining"],
            "required_configs": [
                "DummyDataLoading_training",
                "TabularPreprocessing_training",
                "BedrockPromptTemplateGeneration",
                "BedrockBatchProcessing_training",
                "PyTorchTraining",
            ],
            "bedrock_batch_integration": {
                "template_generation": "BedrockPromptTemplateGeneration",
                "training_processing": "BedrockBatchProcessing_training",
                "training_flow": {
                    "input_sources": [
                        "TabularPreprocessing_training",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": "PyTorchTraining",
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
    Validate the structure of the Bedrock Batch-enhanced simple training DAG.

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
    bedrock_processing_node = "BedrockBatchProcessing_training"
    if bedrock_processing_node in dag.nodes:
        # Get predecessors of BedrockBatchProcessing_training
        bedrock_predecessors = set()
        for edge in dag.edges:
            if edge[1] == bedrock_processing_node:
                bedrock_predecessors.add(edge[0])

        expected_inputs = set(
            bedrock_integration.get("training_flow", {}).get("input_sources", [])
        )
        if bedrock_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"BedrockBatchProcessing_training inputs mismatch. Expected: {expected_inputs}, Found: {bedrock_predecessors}"
            )

    # Check that BedrockBatchProcessing_training outputs to PyTorchTraining
    pytorch_training_node = bedrock_integration.get("training_flow", {}).get(
        "output_target"
    )
    if pytorch_training_node and pytorch_training_node in dag.nodes:
        bedrock_to_pytorch_edge = (bedrock_processing_node, pytorch_training_node)
        if bedrock_to_pytorch_edge not in dag.edges:
            validation_result["errors"].append(
                f"Missing edge from {bedrock_processing_node} to {pytorch_training_node}"
            )
            validation_result["is_valid"] = False

    # Validate expected edges
    expected_edges = [
        ("DummyDataLoading_training", "TabularPreprocessing_training"),
        ("TabularPreprocessing_training", "BedrockBatchProcessing_training"),
        ("BedrockPromptTemplateGeneration", "BedrockBatchProcessing_training"),
        ("BedrockBatchProcessing_training", "PyTorchTraining"),
    ]

    for edge in expected_edges:
        if edge not in dag.edges:
            validation_result["errors"].append(f"Missing expected edge: {edge}")
            validation_result["is_valid"] = False

    return validation_result


def get_training_flow_info() -> Dict[str, Any]:
    """
    Get information about the Bedrock Batch-enhanced training flow in this DAG.

    Returns:
        Dict containing training flow details
    """
    return {
        "flow_type": "bedrock_batch_enhanced_training",
        "steps": [
            {
                "step": "DummyDataLoading_training",
                "purpose": "Load dummy training data",
                "output": "Raw training dataset",
            },
            {
                "step": "TabularPreprocessing_training",
                "purpose": "Preprocess training data with train/val/test splits",
                "output": "Processed training data with splits",
            },
            {
                "step": "BedrockPromptTemplateGeneration",
                "purpose": "Generate LLM prompt templates and validation schemas",
                "output": "Prompt templates and validation schemas",
            },
            {
                "step": "BedrockBatchProcessing_training",
                "purpose": "Cost-efficiently enhance training data with LLM-generated insights using batch processing",
                "output": "LLM-enhanced training data with preserved splits (batch processed)",
            },
            {
                "step": "PyTorchTraining",
                "purpose": "Train PyTorch model using LLM-enhanced data",
                "output": "Trained PyTorch model artifacts",
            },
        ],
        "data_flow": "DummyDataLoading_training → TabularPreprocessing_training → BedrockBatchProcessing_training → PyTorchTraining",
        "template_flow": "BedrockPromptTemplateGeneration → BedrockBatchProcessing_training",
        "characteristics": {
            "simple": True,
            "training_only": True,
            "dummy_data": True,
            "llm_enhanced": True,
            "bedrock_batch_integration": True,
            "cost_optimized": True,
            "automatic_fallback": True,
            "no_calibration": True,
            "no_packaging": True,
            "no_registration": True,
            "no_evaluation": True,
        },
        "bedrock_batch_features": {
            "prompt_template_generation": True,
            "batch_llm_data_enhancement": True,
            "cost_optimization": True,
            "intelligent_mode_selection": True,
            "automatic_fallback": True,
            "train_val_test_preservation": True,
            "field_preservation": True,
            "validation_schema_support": True,
            "s3_integration": True,
            "batch_job_management": True,
        },
    }


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
    }


def get_integration_notes() -> Dict[str, str]:
    """
    Get integration notes for implementing this DAG.

    Returns:
        Dict containing implementation notes and considerations
    """
    return {
        "bedrock_batch_setup": "Ensure Bedrock batch processing step is configured with appropriate IAM role ARN for batch inference jobs",
        "cost_optimization": "Batch processing provides up to 50% cost savings for large datasets (>= 1000 records by default)",
        "automatic_fallback": "BedrockBatchProcessing automatically falls back to real-time processing when batch processing is not suitable",
        "data_flow": "TabularPreprocessing_training output must be compatible with BedrockBatchProcessing_training input format expectations",
        "template_compatibility": "BedrockPromptTemplateGeneration outputs must match BedrockBatchProcessing_training input requirements for prompt_templates and validation_schema",
        "pytorch_integration": "BedrockBatchProcessing_training output format is identical to BedrockProcessing, ensuring compatibility with PyTorchTraining",
        "parallel_execution": "BedrockPromptTemplateGeneration can run in parallel with DummyDataLoading_training and TabularPreprocessing_training for better performance",
        "s3_integration": "BedrockBatchProcessing uses cursus framework patterns for S3 path management - no additional S3 configuration required",
        "batch_job_management": "Batch processing includes automatic job monitoring, result retrieval, and error handling",
        "processing_modes": "Supports three modes: 'auto' (intelligent selection), 'batch' (forced batch), 'realtime' (forced real-time)",
        "monitoring": "Add monitoring for Bedrock batch job status, processing latency, and cost savings achieved",
        "production_readiness": "Ensure fallback model is configured for production reliability and consider inference profile usage for Claude 4+ models",
        "simplicity": "This DAG focuses on training-only workflow - no calibration, packaging, or registration complexity while providing cost optimization",
    }


def get_cost_optimization_benefits() -> Dict[str, Any]:
    """
    Get detailed information about cost optimization benefits of this simple training DAG.

    Returns:
        Dict containing cost optimization details
    """
    return {
        "batch_processing_advantages": {
            "cost_savings": "Up to 50% reduction in Bedrock API costs for large training datasets",
            "scalability": "No memory limits - can process millions of training records",
            "efficiency": "AWS-managed batch infrastructure with optimal resource allocation",
            "fault_tolerance": "Built-in retry and error recovery mechanisms",
        },
        "automatic_optimization": {
            "intelligent_selection": "Automatically chooses batch vs real-time based on training data size",
            "threshold_based": "Default threshold of 1000 records (configurable)",
            "fallback_strategy": "Seamless fallback to real-time processing if batch fails",
            "zero_configuration": "Works with existing training pipeline configurations",
        },
        "operational_benefits": {
            "monitoring": "Enhanced batch job status tracking and cost reporting",
            "reliability": "Automatic fallback ensures training pipeline never fails due to batch issues",
            "compatibility": "Drop-in replacement for existing BedrockProcessing steps",
            "framework_integration": "Uses cursus S3 path patterns for seamless integration",
            "simplicity": "Maintains simple training workflow while adding cost optimization",
        },
        "recommended_use_cases": [
            "Large training datasets (>= 1000 records)",
            "Cost-sensitive training pipelines",
            "High-volume training data processing scenarios",
            "Training scenarios where processing latency is not critical",
            "Simple training workflows without calibration/packaging complexity",
        ],
        "training_specific_benefits": {
            "enhanced_training_data": "LLM-enhanced features improve model training quality",
            "preserved_splits": "Maintains train/val/test splits throughout batch processing",
            "field_preservation": "All original data fields preserved alongside LLM enhancements",
            "validation_support": "Pydantic validation ensures data quality for training",
            "cost_efficient_enhancement": "Significant cost savings for large training datasets",
        },
    }


def get_simple_training_comparison() -> Dict[str, Any]:
    """
    Compare this batch-enhanced simple training DAG with the original simple training DAG.

    Returns:
        Dict containing comparison details
    """
    return {
        "structural_differences": {
            "nodes": "Same 5 nodes, BedrockBatchProcessing_training replaces BedrockProcessing_training",
            "edges": "Same 4 edges, identical flow structure",
            "complexity": "Same moderate complexity level",
        },
        "functional_differences": {
            "processing_mode": "Batch processing with automatic fallback vs real-time only",
            "cost_optimization": "Up to 50% cost savings vs standard pricing",
            "scalability": "No memory limits vs processing instance constraints",
            "reliability": "Enhanced with automatic fallback vs single processing mode",
        },
        "compatibility": {
            "input_format": "Identical - same TabularPreprocessing output requirements",
            "output_format": "Identical - same PyTorchTraining input compatibility",
            "configuration": "Drop-in replacement with additional batch-specific options",
            "dependencies": "Same template and data dependencies",
        },
        "when_to_use_batch_version": [
            "Training datasets >= 1000 records",
            "Cost optimization is a priority",
            "Processing latency is not critical",
            "Large-scale training scenarios",
            "Production training pipelines with cost constraints",
        ],
        "when_to_use_original_version": [
            "Small training datasets < 1000 records",
            "Low latency requirements",
            "Real-time training scenarios",
            "Development/testing with small datasets",
        ],
    }
