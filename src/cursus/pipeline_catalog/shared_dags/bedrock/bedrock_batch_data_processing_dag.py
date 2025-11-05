"""
Shared DAG definition for Bedrock Batch Data Processing Pipeline

This module provides the shared DAG definition for a pure data processing workflow
that uses dummy data loading, Bedrock batch LLM enhancement, without any training.
This is the simplest possible pipeline focused purely on cost-efficient data enhancement
using Bedrock batch processing.

The DAG includes:
1) Dummy Data Loading
2) Tabular Preprocessing
3) Bedrock Prompt Template Generation
4) Bedrock Batch Processing - cost-efficient LLM data enhancement

Key Features:
- Pure data processing workflow without any training/modeling complexity
- Cost-efficient Bedrock batch processing with automatic fallback
- Up to 50% cost reduction for large datasets
- Intelligent processing mode selection (auto, batch, realtime)
- Perfect for data enhancement, annotation, and enrichment workflows
- Minimal pipeline for maximum cost efficiency
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_bedrock_batch_data_processing_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock Batch data processing pipeline.

    This DAG represents the simplest possible workflow that includes
    cost-efficient Bedrock batch LLM enhancement for pure data processing
    without any training, calibration, packaging, registration, or evaluation steps.
    Perfect for data enhancement and annotation workflows.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add minimal data processing nodes with Bedrock batch enhancement
    dag.add_node("DummyDataLoading")  # Dummy data load
    dag.add_node("TabularPreprocessing")  # Tabular preprocessing
    dag.add_node(
        "BedrockPromptTemplateGeneration"
    )  # Bedrock prompt template generation
    dag.add_node("BedrockBatchProcessing")  # Bedrock batch processing step

    # Simple data processing flow with Bedrock batch enhancement
    dag.add_edge("DummyDataLoading", "TabularPreprocessing")

    # Bedrock batch processing flow - two inputs to BedrockBatchProcessing
    dag.add_edge("TabularPreprocessing", "BedrockBatchProcessing")  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockBatchProcessing"
    )  # Template input

    logger.info(
        f"Created Bedrock Batch data processing DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock Batch data processing DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Bedrock Batch data processing pipeline for cost-efficient LLM-based data enhancement and annotation",
        complexity="simple",
        features=[
            "dummy_data_loading",
            "bedrock_prompt_generation",
            "bedrock_batch_processing",
            "cost_optimization",
            "data_processing",
        ],
        framework="data_processing",
        node_count=4,
        edge_count=3,
        extra_metadata={
            "name": "bedrock_batch_data_processing",
            "task_type": "data_processing_with_batch_llm",
            "entry_points": [
                "DummyDataLoading",
                "BedrockPromptTemplateGeneration",
            ],
            "exit_points": ["BedrockBatchProcessing"],
            "required_configs": [
                "DummyDataLoading",
                "TabularPreprocessing",
                "BedrockPromptTemplateGeneration",
                "BedrockBatchProcessing",
            ],
            "bedrock_batch_integration": {
                "template_generation": "BedrockPromptTemplateGeneration",
                "data_processing": "BedrockBatchProcessing",
                "data_flow": {
                    "input_sources": [
                        "TabularPreprocessing",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": None,  # No downstream processing - pure data enhancement
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
    Validate the structure of the Bedrock Batch data processing DAG.

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

    # Check that BedrockBatchProcessing has the correct inputs
    bedrock_processing_node = "BedrockBatchProcessing"
    if bedrock_processing_node in dag.nodes:
        # Get predecessors of BedrockBatchProcessing
        bedrock_predecessors = set()
        for edge in dag.edges:
            if edge[1] == bedrock_processing_node:
                bedrock_predecessors.add(edge[0])

        expected_inputs = set(
            bedrock_integration.get("data_flow", {}).get("input_sources", [])
        )
        if bedrock_predecessors != expected_inputs:
            validation_result["warnings"].append(
                f"BedrockBatchProcessing inputs mismatch. Expected: {expected_inputs}, Found: {bedrock_predecessors}"
            )

    # Validate expected edges
    expected_edges = [
        ("DummyDataLoading", "TabularPreprocessing"),
        ("TabularPreprocessing", "BedrockBatchProcessing"),
        ("BedrockPromptTemplateGeneration", "BedrockBatchProcessing"),
    ]

    for edge in expected_edges:
        if edge not in dag.edges:
            validation_result["errors"].append(f"Missing expected edge: {edge}")
            validation_result["is_valid"] = False

    return validation_result


def get_data_processing_flow_info() -> Dict[str, Any]:
    """
    Get information about the Bedrock Batch data processing flow in this DAG.

    Returns:
        Dict containing data processing flow details
    """
    return {
        "flow_type": "bedrock_batch_data_processing",
        "steps": [
            {
                "step": "DummyDataLoading",
                "purpose": "Load dummy data for processing",
                "output": "Raw dataset",
            },
            {
                "step": "TabularPreprocessing",
                "purpose": "Preprocess data with standardization and formatting",
                "output": "Processed data ready for LLM enhancement",
            },
            {
                "step": "BedrockPromptTemplateGeneration",
                "purpose": "Generate LLM prompt templates and validation schemas",
                "output": "Prompt templates and validation schemas",
            },
            {
                "step": "BedrockBatchProcessing",
                "purpose": "Cost-efficiently enhance data with LLM-generated insights using batch processing",
                "output": "LLM-enhanced data with annotations, classifications, or enrichments",
            },
        ],
        "data_flow": "DummyDataLoading → TabularPreprocessing → BedrockBatchProcessing",
        "template_flow": "BedrockPromptTemplateGeneration → BedrockBatchProcessing",
        "characteristics": {
            "minimal": True,
            "data_processing_only": True,
            "dummy_data": True,
            "llm_enhanced": True,
            "bedrock_batch_integration": True,
            "cost_optimized": True,
            "automatic_fallback": True,
            "no_training": True,
            "no_calibration": True,
            "no_packaging": True,
            "no_registration": True,
            "no_evaluation": True,
            "pure_data_enhancement": True,
        },
        "bedrock_batch_features": {
            "prompt_template_generation": True,
            "batch_llm_data_enhancement": True,
            "cost_optimization": True,
            "intelligent_mode_selection": True,
            "automatic_fallback": True,
            "field_preservation": True,
            "validation_schema_support": True,
            "s3_integration": True,
            "batch_job_management": True,
            "data_annotation": True,
            "data_enrichment": True,
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
        "BedrockBatchProcessing": {
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
                    "source_step": "TabularPreprocessing",
                    "output_name": "processed_data",
                    "required": True,
                },
            },
            "outputs": {
                "processed_data": "LLM-enhanced processed data (batch processed)",
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
        "data_flow": "TabularPreprocessing output must be compatible with BedrockBatchProcessing input format expectations",
        "template_compatibility": "BedrockPromptTemplateGeneration outputs must match BedrockBatchProcessing input requirements for prompt_templates and validation_schema",
        "output_usage": "BedrockBatchProcessing output can be used for downstream analysis, model training, or data export",
        "parallel_execution": "BedrockPromptTemplateGeneration can run in parallel with DummyDataLoading and TabularPreprocessing for better performance",
        "s3_integration": "BedrockBatchProcessing uses cursus framework patterns for S3 path management - no additional S3 configuration required",
        "batch_job_management": "Batch processing includes automatic job monitoring, result retrieval, and error handling",
        "processing_modes": "Supports three modes: 'auto' (intelligent selection), 'batch' (forced batch), 'realtime' (forced real-time)",
        "monitoring": "Add monitoring for Bedrock batch job status, processing latency, and cost savings achieved",
        "production_readiness": "Ensure fallback model is configured for production reliability and consider inference profile usage for Claude 4+ models",
        "simplicity": "This DAG focuses purely on data processing - no training, modeling, or deployment complexity",
        "data_export": "Enhanced data can be exported to various formats for downstream use in other systems or workflows",
    }


def get_cost_optimization_benefits() -> Dict[str, Any]:
    """
    Get detailed information about cost optimization benefits of this data processing DAG.

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
            "zero_configuration": "Works with existing data processing configurations",
        },
        "operational_benefits": {
            "monitoring": "Enhanced batch job status tracking and cost reporting",
            "reliability": "Automatic fallback ensures data processing never fails due to batch issues",
            "compatibility": "Drop-in replacement for existing BedrockProcessing steps",
            "framework_integration": "Uses cursus S3 path patterns for seamless integration",
            "simplicity": "Minimal pipeline complexity while maximizing cost optimization",
        },
        "recommended_use_cases": [
            "Large datasets requiring LLM enhancement (>= 1000 records)",
            "Data annotation and labeling workflows",
            "Data enrichment and augmentation projects",
            "Content analysis and classification tasks",
            "Cost-sensitive data processing scenarios",
            "Batch data preparation for downstream ML workflows",
        ],
        "data_processing_specific_benefits": {
            "enhanced_data_quality": "LLM-generated annotations and classifications improve data quality",
            "field_preservation": "All original data fields preserved alongside LLM enhancements",
            "validation_support": "Pydantic validation ensures data quality and consistency",
            "cost_efficient_enhancement": "Significant cost savings for large-scale data processing",
            "flexible_output": "Enhanced data ready for export to various downstream systems",
            "no_training_overhead": "Pure data processing without model training complexity",
        },
    }


def get_use_case_examples() -> Dict[str, Any]:
    """
    Get examples of use cases for this data processing pipeline.

    Returns:
        Dict containing use case examples and scenarios
    """
    return {
        "data_annotation_scenarios": [
            "Sentiment analysis of customer reviews",
            "Content categorization and tagging",
            "Entity extraction from text documents",
            "Quality assessment of user-generated content",
            "Language detection and translation preparation",
        ],
        "data_enrichment_scenarios": [
            "Product description enhancement",
            "Customer profile augmentation",
            "Document summarization and key point extraction",
            "Data standardization and normalization",
            "Missing field completion using LLM inference",
        ],
        "analysis_preparation_scenarios": [
            "Preparing data for downstream ML model training",
            "Creating labeled datasets for supervised learning",
            "Generating features for predictive modeling",
            "Data quality assessment and scoring",
            "Compliance and content moderation preparation",
        ],
        "workflow_integration": {
            "standalone_processing": "Use as independent data enhancement pipeline",
            "preprocessing_for_training": "Prepare enhanced data for subsequent training pipelines",
            "batch_analysis": "Process large datasets for business intelligence and reporting",
            "data_pipeline_component": "Integrate as part of larger ETL/ELT workflows",
            "research_and_development": "Explore data characteristics and LLM capabilities",
        },
        "output_formats": [
            "Enhanced CSV files with LLM-generated columns",
            "Parquet files for efficient downstream processing",
            "JSON files with structured LLM responses",
            "Metadata files with processing statistics and quality metrics",
        ],
    }


def get_pipeline_comparison() -> Dict[str, Any]:
    """
    Compare this data processing DAG with other Bedrock batch pipelines.

    Returns:
        Dict containing comparison details
    """
    return {
        "vs_simple_training_dag": {
            "nodes": "4 nodes vs 5 nodes (removed PyTorchTraining)",
            "edges": "3 edges vs 4 edges (removed training connection)",
            "complexity": "Simple vs Moderate (reduced complexity)",
            "purpose": "Pure data processing vs Training preparation",
            "output": "Enhanced data vs Trained model",
            "use_case": "Data enhancement vs Model development",
        },
        "vs_e2e_dag": {
            "nodes": "4 nodes vs 13 nodes (minimal vs comprehensive)",
            "edges": "3 edges vs 15 edges (simple vs complex)",
            "complexity": "Simple vs Comprehensive (maximum simplification)",
            "purpose": "Data processing only vs Full ML lifecycle",
            "output": "Enhanced data vs Deployed model",
            "use_case": "Data enhancement vs Production ML system",
        },
        "advantages_of_data_processing_dag": [
            "Minimal complexity - easiest to understand and maintain",
            "Fastest execution - no training or evaluation overhead",
            "Maximum cost efficiency - only pay for data processing",
            "Flexible output - enhanced data can be used anywhere",
            "No ML expertise required - pure data enhancement workflow",
            "Ideal for data exploration and preparation",
        ],
        "when_to_use_data_processing_dag": [
            "Pure data enhancement and annotation needs",
            "Preparing data for external ML systems",
            "Data quality improvement projects",
            "Content analysis and classification tasks",
            "Research and data exploration scenarios",
            "Cost-sensitive data processing requirements",
        ],
    }
