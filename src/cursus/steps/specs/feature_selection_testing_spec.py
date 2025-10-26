"""
Feature Selection Testing Step Specification.

This module defines the declarative specification for feature selection steps
specifically for testing data, including their dependencies and outputs.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type_with_job_type


# Import the contract at runtime to avoid circular imports
def _get_feature_selection_contract():
    from ..contracts.feature_selection_contract import FEATURE_SELECTION_CONTRACT

    return FEATURE_SELECTION_CONTRACT


# Feature Selection Testing Step Specification
FEATURE_SELECTION_TESTING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("FeatureSelection", "testing"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_feature_selection_contract(),
    dependencies=[
        DependencySpec(
            logical_name="processed_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling", 
                "RiskTableMapping",
                "MissingValueImputation",
                "ProcessingStep"
            ],
            semantic_keywords=[
                "testing",
                "test",
                "processed_data",
                "preprocessed",
                "cleaned",
                "tabular",
                "data",
                "input",
                "dataset",
            ],
            data_type="S3Uri",
            description="Processed testing data from preprocessing steps for feature selection",
        ),
        DependencySpec(
            logical_name="selected_features",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["FeatureSelection_Training"],
            semantic_keywords=[
                "selected_features",
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "training_artifacts",
            ],
            data_type="S3Uri",
            description="Selected features metadata and scores from training step",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "selected_data",
                "feature_selected_data",
                "testing_data",
                "model_testing_data",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Testing data with selected features applied using pre-computed selection",
        ),
        OutputSpec(
            logical_name="selected_features",
            aliases=[
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "selection_results",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['selected_features'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Selected features metadata and scores (passthrough from training)",
        ),
    ],
)
