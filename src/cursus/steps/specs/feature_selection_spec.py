"""
Feature Selection Step Specification.

This module defines the general declarative specification for feature selection steps,
including their dependencies and outputs. This serves as the base specification
that can be used when job type is not specified.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)


# Import the contract at runtime to avoid circular imports
def _get_feature_selection_contract():
    from ..contracts.feature_selection_contract import FEATURE_SELECTION_CONTRACT

    return FEATURE_SELECTION_CONTRACT


# General Feature Selection Step Specification
FEATURE_SELECTION_SPEC = StepSpecification(
    step_type="FeatureSelection",
    node_type=NodeType.INTERNAL,
    script_contract=_get_feature_selection_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "RiskTableMapping",
                "MissingValueImputation",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "processed_data",
                "preprocessed",
                "cleaned",
                "tabular",
                "data",
                "input",
                "dataset",
                "splits",
                "feature_engineering",
                "training",
                "validation",
                "testing",
                "calibration",
            ],
            data_type="S3Uri",
            description="Processed data from preprocessing steps for feature selection",
        ),
        # Selected features dependency - optional for training mode, required for others
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=[
                "FeatureSelection_Training",
                "FeatureSelection",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "selected_features",
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "training_artifacts",
                "model_artifacts",
            ],
            data_type="S3Uri",
            description="Selected features metadata and scores (required for non-training modes)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "selected_data",
                "feature_selected_data",
                "training_data",
                "validation_data",
                "testing_data",
                "calibration_data",
                "model_input_data",
                "input_path",
                "input_data",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Data with selected features applied using statistical and ML methods",
        ),
        OutputSpec(
            logical_name="model_artifacts_output",
            aliases=[
                "selected_features",
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "selection_results",
                "training_artifacts",
                "model_artifacts",
                "model_artifacts_input",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['model_artifacts_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Selected features metadata and scores for downstream consumption",
        ),
    ],
)
