"""
Bedrock Prompt Template Generation Step Specification.

This module defines the declarative specification for the Bedrock prompt template generation step,
including its dependencies and outputs based on the actual implementation.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type


# Import the contract at runtime to avoid circular imports
def _get_bedrock_prompt_template_generation_contract():
    from ..contracts.bedrock_prompt_template_generation_contract import (
        BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT,
    )

    return BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT


# Bedrock Prompt Template Generation Step Specification
BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("BedrockPromptTemplateGeneration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_bedrock_prompt_template_generation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="prompt_configs",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=[
                "PromptConfiguration",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "prompt_configs",
                "prompt_configuration",
                "bedrock_configs",
                "template_configs",
                "system_prompt_config",
                "output_format_config",
                "instruction_config",
                "category_definitions",
                "classification_config",
                "llm_configs",
                "bedrock_prompt_configs",
            ],
            data_type="S3Uri",
            description="Prompt configuration directory containing JSON files: system_prompt.json, output_format.json, instruction.json, and category_definitions.json for template generation",
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="prompt_templates",
            aliases=[
                "templates",
                "bedrock_templates",
                "prompt_template",
                "generated_templates",
                "classification_templates",
                "llm_templates",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['prompt_templates'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated prompt templates with system prompt and user prompt template for Bedrock processing",
        ),
        OutputSpec(
            logical_name="template_metadata",
            aliases=[
                "metadata",
                "generation_metadata",
                "template_info",
                "validation_results",
                "quality_metrics",
                "generation_results",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['template_metadata'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Template generation metadata including validation results, quality scores, and configuration details",
        ),
        OutputSpec(
            logical_name="validation_schema",
            aliases=[
                "schema",
                "output_validation_schema",
                "bedrock_validation_schema",
                "response_schema",
                "classification_schema",
                "downstream_schema",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['validation_schema'].S3Output.S3Uri",
            data_type="S3Uri",
            description="JSON validation schemas for validating Bedrock responses in downstream processing steps",
        ),
    ],
)
