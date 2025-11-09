"""
Label Ruleset Generation Step Specification.

This module defines the declarative specification for the label ruleset generation step,
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
def _get_label_ruleset_generation_contract():
    from ..contracts.label_ruleset_generation_contract import (
        LABEL_RULESET_GENERATION_CONTRACT,
    )

    return LABEL_RULESET_GENERATION_CONTRACT


# Label Ruleset Generation Step Specification
LABEL_RULESET_GENERATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("LabelRulesetGeneration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_label_ruleset_generation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="ruleset_configs",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=[
                "RulesetConfiguration",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "ruleset_configs",
                "ruleset_configuration",
                "rule_configs",
                "classification_rules",
                "label_rules",
                "label_config",
                "field_config",
                "rule_definitions",
                "ruleset",
                "rules",
                "label_mapping_rules",
                "classification_config",
            ],
            data_type="S3Uri",
            description="Ruleset configuration directory containing JSON files: label_config.json (label definitions), field_config.json (field schema), and ruleset.json (rule definitions) for validation and optimization",
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="validated_ruleset",
            aliases=[
                "ruleset",
                "optimized_ruleset",
                "validated_rules",
                "classification_ruleset",
                "label_ruleset",
                "generated_ruleset",
                "rules",
                "label_mapping_rules",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['validated_ruleset'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Validated and optimized ruleset with label configuration, field configuration, and priority-ordered rules for execution",
        ),
        OutputSpec(
            logical_name="validation_report",
            aliases=[
                "report",
                "validation_results",
                "ruleset_metadata",
                "validation_metadata",
                "quality_report",
                "diagnostics",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['validation_report'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Detailed validation report including field validation, label validation, logic validation results, and optimization metadata",
        ),
    ],
)
