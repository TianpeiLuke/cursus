"""
Bedrock Batch Processing Step Specification.

This module defines the declarative specification for Bedrock batch processing steps,
including their dependencies and outputs. This step processes input data through
AWS Bedrock models using batch inference capabilities with automatic fallback to
real-time processing. Integrates with generated prompt templates and validation schemas
from the Bedrock Prompt Template Generation step.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.bedrock_batch_processing_contract import (
        BEDROCK_BATCH_PROCESSING_CONTRACT,
    )


# Import the contract at runtime to avoid circular imports
def _get_bedrock_batch_processing_contract():
    from ..contracts.bedrock_batch_processing_contract import (
        BEDROCK_BATCH_PROCESSING_CONTRACT,
    )

    return BEDROCK_BATCH_PROCESSING_CONTRACT


# Bedrock Batch Processing Step Specification
BEDROCK_BATCH_PROCESSING_SPEC = StepSpecification(
    step_type=get_spec_step_type("BedrockBatchProcessing"),
    node_type=NodeType.INTERNAL,  # INTERNAL node with dependencies
    script_contract=_get_bedrock_batch_processing_contract(),  # Add reference to the script contract
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "DummyDataLoading",
                "CradleDataLoading",
                "TabularPreprocessing",
                "TemporalSequenceNormalization",
                "TemporalFeatureEngineering",
                "StratifiedSampling",
                "MissingValueImputation",
                "FeatureSelection",
                "CurrencyConversion",
            ],
            semantic_keywords=[
                "data",
                "dataset",
                "input",
                "processed_data",
                "features",
            ],
            data_type="S3Uri",
            description="Input data to be processed through Bedrock models with batch inference (CSV or Parquet format)",
        ),
        DependencySpec(
            logical_name="prompt_templates",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["BedrockPromptTemplateGeneration"],
            semantic_keywords=[
                "templates",
                "prompts",
                "prompt_templates",
                "bedrock_templates",
            ],
            data_type="S3Uri",
            description="Prompt templates from Bedrock Prompt Template Generation step (prompts.json)",
        ),
        DependencySpec(
            logical_name="validation_schema",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["BedrockPromptTemplateGeneration"],
            semantic_keywords=[
                "schema",
                "validation",
                "validation_schema",
                "response_schema",
            ],
            data_type="S3Uri",
            description="Validation schemas from Bedrock Prompt Template Generation step (validation_schema_*.json)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "input_path",
                "training_data",
                "model_input_data",
                "input_data",
            ],  # Added aliases matching TabularPreprocessing for better compatibility
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed data with Bedrock responses and validation results (batch or real-time processed)",
        ),
        OutputSpec(
            logical_name="analysis_summary",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['analysis_summary'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processing statistics, success rates, batch processing usage, and comprehensive analysis metadata",
        ),
    ],
)
