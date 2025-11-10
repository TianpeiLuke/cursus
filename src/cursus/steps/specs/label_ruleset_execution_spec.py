"""
Label Ruleset Execution Step Specification.

This module defines the declarative specification for label ruleset execution steps,
including their dependencies and outputs. This step applies validated rulesets to
processed data to generate classification labels using priority-based rule evaluation.
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
    from ..contracts.label_ruleset_execution_contract import (
        LABEL_RULESET_EXECUTION_CONTRACT,
    )


# Import the contract at runtime to avoid circular imports
def _get_label_ruleset_execution_contract():
    from ..contracts.label_ruleset_execution_contract import (
        LABEL_RULESET_EXECUTION_CONTRACT,
    )

    return LABEL_RULESET_EXECUTION_CONTRACT


# Label Ruleset Execution Step Specification
LABEL_RULESET_EXECUTION_SPEC = StepSpecification(
    step_type=get_spec_step_type("LabelRulesetExecution"),
    node_type=NodeType.INTERNAL,  # INTERNAL node with dependencies
    script_contract=_get_label_ruleset_execution_contract(),  # Add reference to the script contract
    dependencies=[
        DependencySpec(
            logical_name="validated_ruleset",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["LabelRulesetGeneration"],
            semantic_keywords=[
                "ruleset",
                "validated_ruleset",
                "rules",
                "label_rules",
            ],
            data_type="S3Uri",
            description="Validated ruleset from Label Ruleset Generation step (validated_ruleset.json)",
        ),
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "BedrockProcessing",
                "BedrockBatchProcessing",
                "TemporalSequenceNormalization",
                "TemporalFeatureEngineering",
                "StratifiedSampling",
                "MissingValueImputation",
                "FeatureSelection",
                "CurrencyConversion",
                "RiskTableMapping",
            ],
            semantic_keywords=[
                "data",
                "dataset",
                "input",
                "input_data",
                "processed_data",
                "features",
            ],
            data_type="S3Uri",
            description="Processed data from preprocessing steps to apply ruleset labels (supports CSV, TSV, Parquet formats)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "input_data",
                "input_path",
                "training_data",
                "model_input_data",
                "labeled_data",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Labeled data with ruleset-generated classification labels added",
        ),
        OutputSpec(
            logical_name="execution_report",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['execution_report'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Execution statistics including rule match counts, label distribution, and validation results",
        ),
    ],
)
