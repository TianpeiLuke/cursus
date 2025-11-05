"""
Shared DAG definition for Bedrock-Enhanced Simple Training Pipeline

This module provides the shared DAG definition for a simple training workflow
that uses dummy data loading, Bedrock LLM enhancement, and PyTorch training.
This combines the simplicity of the basic training pipeline with the power
of Bedrock LLM processing.

The DAG includes:
1) Dummy Data Loading (training)
2) Tabular Preprocessing (training)
3) Bedrock Prompt Template Generation
4) Bedrock Processing (training)
5) PyTorch Model Training
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_bedrock_simple_training_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock-enhanced simple training pipeline.

    This DAG represents a simplified training workflow that includes
    Bedrock LLM enhancement between preprocessing and PyTorch training,
    without calibration, packaging, registration, or evaluation steps.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add core training nodes with Bedrock enhancement
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "BedrockPromptTemplateGeneration"
    )  # Bedrock prompt template generation
    dag.add_node("BedrockProcessing_training")  # Bedrock processing step for training
    dag.add_node("PyTorchTraining")  # PyTorch training step

    # Training flow with Bedrock enhancement
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")

    # Bedrock processing flow for training - two inputs to BedrockProcessing_training
    dag.add_edge(
        "TabularPreprocessing_training", "BedrockProcessing_training"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockProcessing_training"
    )  # Template input

    # Enhanced data flows to PyTorch training
    dag.add_edge("BedrockProcessing_training", "PyTorchTraining")

    logger.info(
        f"Created Bedrock-enhanced simple training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock-enhanced simple training DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Bedrock-enhanced simple training pipeline with dummy data loading, LLM processing, and PyTorch training",
        complexity="moderate",
        features=[
            "dummy_data_loading",
            "bedrock_prompt_generation",
            "bedrock_processing",
            "training",
        ],
        framework="pytorch",
        node_count=5,
        edge_count=4,
        extra_metadata={
            "name": "bedrock_simple_training",
            "task_type": "training_with_llm",
            "entry_points": [
                "DummyDataLoading_training",
                "BedrockPromptTemplateGeneration",
            ],
            "exit_points": ["PyTorchTraining"],
            "required_configs": [
                "DummyDataLoading_training",
                "TabularPreprocessing_training",
                "BedrockPromptTemplateGeneration",
                "BedrockProcessing_training",
                "PyTorchTraining",
            ],
            "bedrock_integration": {
                "template_generation": "BedrockPromptTemplateGeneration",
                "training_processing": "BedrockProcessing_training",
                "training_flow": {
                    "input_sources": [
                        "TabularPreprocessing_training",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": "PyTorchTraining",
                },
            },
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the Bedrock-enhanced simple training DAG.

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

    # Validate Bedrock integration structure
    bedrock_integration = metadata.extra_metadata.get("bedrock_integration", {})

    # Check that BedrockProcessing_training has the correct inputs
    bedrock_processing_node = "BedrockProcessing_training"
    if bedrock_processing_node in dag.nodes:
        # Get predecessors of BedrockProcessing_training
        bedrock_predecessors = set()
        for edge in dag.edges:
            if edge[1] == bedrock_processing_node:
                bedrock_predecessors.add(edge[0])

        expected_inputs = set(
            bedrock_integration.get("training_flow", {}).get("input_sources", [])
        )
        if bedrock_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"BedrockProcessing_training inputs mismatch. Expected: {expected_inputs}, Found: {bedrock_predecessors}"
            )

    # Check that BedrockProcessing_training outputs to PyTorchTraining
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
        ("TabularPreprocessing_training", "BedrockProcessing_training"),
        ("BedrockPromptTemplateGeneration", "BedrockProcessing_training"),
        ("BedrockProcessing_training", "PyTorchTraining"),
    ]

    for edge in expected_edges:
        if edge not in dag.edges:
            validation_result["errors"].append(f"Missing expected edge: {edge}")
            validation_result["is_valid"] = False

    return validation_result


def get_training_flow_info() -> Dict[str, Any]:
    """
    Get information about the Bedrock-enhanced training flow in this DAG.

    Returns:
        Dict containing training flow details
    """
    return {
        "flow_type": "bedrock_enhanced_training",
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
                "step": "BedrockProcessing_training",
                "purpose": "Enhance training data with LLM-generated insights",
                "output": "LLM-enhanced training data with preserved splits",
            },
            {
                "step": "PyTorchTraining",
                "purpose": "Train PyTorch model using LLM-enhanced data",
                "output": "Trained PyTorch model artifacts",
            },
        ],
        "data_flow": "DummyDataLoading_training → TabularPreprocessing_training → BedrockProcessing_training → PyTorchTraining",
        "template_flow": "BedrockPromptTemplateGeneration → BedrockProcessing_training",
        "characteristics": {
            "simple": True,
            "training_only": True,
            "dummy_data": True,
            "llm_enhanced": True,
            "bedrock_integration": True,
            "no_calibration": True,
            "no_packaging": True,
            "no_registration": True,
            "no_evaluation": True,
        },
        "bedrock_features": {
            "prompt_template_generation": True,
            "llm_data_enhancement": True,
            "train_val_test_preservation": True,
            "field_preservation": True,
            "validation_schema_support": True,
        },
    }


def get_bedrock_step_dependencies() -> Dict[str, Dict[str, Any]]:
    """
    Get the dependency specifications for Bedrock steps in this DAG.

    Returns:
        Dict mapping step names to their dependency specifications
    """
    return {
        "BedrockPromptTemplateGeneration": {
            "dependencies": {},  # No dependencies - can run independently
            "outputs": {
                "prompt_templates": "Templates for Bedrock processing",
                "template_metadata": "Metadata about generated templates",
                "validation_schema": "Schema for validating Bedrock responses",
            },
        },
        "BedrockProcessing_training": {
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
                "processed_data": "LLM-enhanced processed data for training",
                "processing_metadata": "Metadata about Bedrock processing results",
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
        "bedrock_setup": "Ensure Bedrock prompt template generation step is configured with appropriate category definitions and output format specifications",
        "data_flow": "TabularPreprocessing_training output must be compatible with BedrockProcessing_training input format expectations",
        "template_compatibility": "BedrockPromptTemplateGeneration outputs must match BedrockProcessing_training input requirements for prompt_templates and validation_schema",
        "pytorch_integration": "BedrockProcessing_training output format must be compatible with PyTorchTraining input data expectations",
        "parallel_execution": "BedrockPromptTemplateGeneration can run in parallel with DummyDataLoading_training and TabularPreprocessing_training for better performance",
        "error_handling": "Consider implementing fallback mechanisms if Bedrock processing fails - potentially bypass to direct PyTorch training",
        "monitoring": "Add monitoring for Bedrock API usage, response quality, and processing latency",
        "cost_optimization": "Monitor Bedrock usage costs and consider batching strategies for large datasets",
        "simplicity": "This DAG focuses on training-only workflow - no calibration, packaging, or registration complexity",
    }
