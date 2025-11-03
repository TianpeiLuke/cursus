"""
Shared DAG definition for PyTorch End-to-End Pipeline with Bedrock Processing

This module provides the shared DAG definition for a complete PyTorch workflow
that incorporates Bedrock prompt template generation and processing steps
for LLM-enhanced data processing before PyTorch training and calibration.

The DAG includes separate Bedrock processing paths for training and calibration:

Training Flow:
1) Dummy Data Loading (training)
2) Tabular Preprocessing (training) 
3) Bedrock Prompt Template Generation (shared)
4) Bedrock Processing (training) - receives data + templates
5) PyTorch Model Training

Calibration Flow:
6) Dummy Data Loading (calibration)
7) Tabular Preprocessing (calibration)
8) Bedrock Processing (calibration) - receives data + templates (shared)
9) PyTorch Model Evaluation (calibration)

Final Steps:
10) Model Calibration
11) Package Model
12) Payload Generation  
13) Model Registration

Key Features:
- Separate Bedrock processing for training and calibration data
- Shared prompt template generation for consistency
- LLM-enhanced data processing using AWS Bedrock
- Job type variants (training vs calibration) for different processing behaviors
- Complete end-to-end workflow from data loading to model registration
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_bedrock_pytorch_e2e_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock-enhanced PyTorch E2E pipeline.

    This DAG represents a complete end-to-end workflow that uses Bedrock
    prompt template generation and processing to enhance data before
    PyTorch training, followed by calibration, packaging, registration,
    and evaluation.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - incorporating Bedrock steps with job type variants
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("BedrockPromptTemplateGeneration")  # Bedrock prompt template generation
    dag.add_node("BedrockProcessing_training")  # Bedrock processing step for training
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
    dag.add_node("BedrockProcessing_calibration")  # Bedrock processing step for calibration
    dag.add_node("PyTorchModelEval_calibration")  # Model evaluation step

    # Training flow with Bedrock integration
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")
    
    # Bedrock processing flow for training - two inputs to BedrockProcessing_training
    dag.add_edge("TabularPreprocessing_training", "BedrockProcessing_training")  # Data input
    dag.add_edge("BedrockPromptTemplateGeneration", "BedrockProcessing_training")  # Template input
    
    # Enhanced data flows to PyTorch training
    dag.add_edge("BedrockProcessing_training", "PyTorchTraining")

    # Calibration flow with Bedrock integration
    dag.add_edge("DummyDataLoading_calibration", "TabularPreprocessing_calibration")
    
    # Bedrock processing flow for calibration - two inputs to BedrockProcessing_calibration
    dag.add_edge("TabularPreprocessing_calibration", "BedrockProcessing_calibration")  # Data input
    dag.add_edge("BedrockPromptTemplateGeneration", "BedrockProcessing_calibration")  # Template input

    # Evaluation flow
    dag.add_edge("PyTorchTraining", "PyTorchModelEval_calibration")
    dag.add_edge("BedrockProcessing_calibration", "PyTorchModelEval_calibration")  # Use Bedrock-processed calibration data

    # Model calibration flow - depends on model evaluation
    dag.add_edge("PyTorchModelEval_calibration", "ModelCalibration_calibration")

    # Output flow
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("PyTorchTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("PyTorchTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created Bedrock-PyTorch E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock-enhanced PyTorch end-to-end DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Bedrock-enhanced PyTorch end-to-end pipeline with LLM-based data processing, training, calibration, packaging, registration, and evaluation",
        complexity="comprehensive",
        features=[
            "dummy_data_loading", 
            "bedrock_prompt_generation", 
            "bedrock_processing", 
            "training", 
            "calibration", 
            "packaging", 
            "registration", 
            "evaluation"
        ],
        framework="pytorch",
        node_count=13,
        edge_count=15,
        extra_metadata={
            "name": "bedrock_pytorch_e2e",
            "task_type": "end_to_end_with_llm",
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
                "BedrockProcessing_training",
                "BedrockProcessing_calibration",
                "PyTorchTraining",
                "PyTorchModelEval_calibration",
                "ModelCalibration_calibration",
                "Package",
                "Payload",
                "Registration",
            ],
            "bedrock_integration": {
                "template_generation": "BedrockPromptTemplateGeneration",
                "training_processing": "BedrockProcessing_training",
                "calibration_processing": "BedrockProcessing_calibration",
                "training_flow": {
                    "input_sources": ["TabularPreprocessing_training", "BedrockPromptTemplateGeneration"],
                    "output_target": "PyTorchTraining"
                },
                "calibration_flow": {
                    "input_sources": ["TabularPreprocessing_calibration", "BedrockPromptTemplateGeneration"],
                    "output_target": "PyTorchModelEval_calibration"
                }
            }
        },
    )


def validate_dag_structure(dag: PipelineDAG) -> Dict[str, Any]:
    """
    Validate the structure of the Bedrock-enhanced PyTorch end-to-end DAG.

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
    
    # Check that BedrockProcessing has the correct inputs
    bedrock_processing_node = "BedrockProcessing"
    if bedrock_processing_node in dag.nodes:
        # Get predecessors of BedrockProcessing
        bedrock_predecessors = set()
        for edge in dag.edges:
            if edge[1] == bedrock_processing_node:
                bedrock_predecessors.add(edge[0])
        
        expected_inputs = set(bedrock_integration.get("input_sources", []))
        if bedrock_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"BedrockProcessing inputs mismatch. Expected: {expected_inputs}, Found: {bedrock_predecessors}"
            )
    
    # Check that BedrockProcessing outputs to PyTorchTraining
    pytorch_training_node = bedrock_integration.get("output_target")
    if pytorch_training_node and pytorch_training_node in dag.nodes:
        bedrock_to_pytorch_edge = (bedrock_processing_node, pytorch_training_node)
        if bedrock_to_pytorch_edge not in dag.edges:
            validation_result["errors"].append(
                f"Missing edge from {bedrock_processing_node} to {pytorch_training_node}"
            )
            validation_result["is_valid"] = False

    return validation_result


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
                "validation_schema": "Schema for validating Bedrock responses"
            }
        },
        "BedrockProcessing": {
            "dependencies": {
                "prompt_templates": {
                    "source_step": "BedrockPromptTemplateGeneration",
                    "output_name": "prompt_templates",
                    "required": True
                },
                "validation_schema": {
                    "source_step": "BedrockPromptTemplateGeneration", 
                    "output_name": "validation_schema",
                    "required": True
                },
                "input_data": {
                    "source_step": "TabularPreprocessing_training",
                    "output_name": "processed_data",  # Assuming this is the output name
                    "required": True
                }
            },
            "outputs": {
                "processed_data": "LLM-enhanced processed data for training",
                "processing_metadata": "Metadata about Bedrock processing results"
            }
        }
    }


def get_integration_notes() -> Dict[str, str]:
    """
    Get integration notes for implementing this DAG.
    
    Returns:
        Dict containing implementation notes and considerations
    """
    return {
        "bedrock_setup": "Ensure Bedrock prompt template generation step is configured with appropriate category definitions and output format specifications",
        "data_flow": "TabularPreprocessing_training output must be compatible with BedrockProcessing input format expectations",
        "template_compatibility": "BedrockPromptTemplateGeneration outputs must match BedrockProcessing input requirements for prompt_templates and validation_schema",
        "pytorch_integration": "BedrockProcessing output format must be compatible with PyTorchTraining input data expectations",
        "parallel_execution": "BedrockPromptTemplateGeneration can run in parallel with DummyDataLoading_training and TabularPreprocessing_training for better performance",
        "error_handling": "Consider implementing fallback mechanisms if Bedrock processing fails - potentially bypass to direct PyTorch training",
        "monitoring": "Add monitoring for Bedrock API usage, response quality, and processing latency",
        "cost_optimization": "Monitor Bedrock usage costs and consider batching strategies for large datasets"
    }
